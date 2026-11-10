# rag.py

import os
import re
import uuid
from dataclasses import dataclass, field

from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata

DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# How chunks are re-ordered after the vector search:
#   cross-encoder: a small model scores every (question, chunk) pair at once. Fast.
#   llm:           the chat model rates each chunk 1-10, one call per chunk. Slow.
#   none:          keep the vector search order.
RERANKERS = ("cross-encoder", "llm", "none")

# How the RETRIEVE_K candidate chunks are found before reranking:
#   vector: embedding similarity only.
#   hybrid: embedding similarity + BM25 keyword search, merged with reciprocal rank fusion.
#           Helps with exact terms like "BLEU", "P100" or "warmup_steps".
SEARCH_MODES = ("vector", "hybrid")
RRF_K = 60  # standard reciprocal rank fusion constant

RETRIEVE_K = 20
TOP_N = 5
HISTORY_TURNS = 3  # past question/answer pairs used to rewrite a follow-up

NO_DOCUMENT_MESSAGE = "Please upload and ingest a PDF document first."


# Prompt to score each chunk (llm reranker)
SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rate the passage for how useful it is for answering the question (1-10). Only output a single number."),
    ("human", "Question: {question}\n\nPassage: {passage}")
])

# Final prompt to answer the question, grounded in the retrieved context
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about the user's documents. "
     "Use only the context provided. If the context does not contain the answer, "
     "reply: \"I couldn't find that in the document.\" Do not make up facts."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# Prompt to turn a follow-up ("what about page 2?") into a standalone question
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's follow-up question so it can be understood without the conversation. "
     "Keep it short. If it is already standalone, return it unchanged. Output only the question."),
    ("human", "Conversation:\n{history}\n\nFollow-up question: {question}")
])


@dataclass
class Source:
    file_name: str
    page: int  # 1-based
    text: str
    score: float


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    question: str = ""  # the standalone question actually searched for


def format_history(history):
    """history: list of (question, answer) pairs, oldest first."""
    lines = []
    for question, answer in history[-HISTORY_TURNS:]:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


def to_source(score, doc) -> Source:
    meta = doc.metadata
    return Source(
        file_name=meta.get("file_name") or os.path.basename(meta.get("source", "")),
        page=int(meta.get("page", 0)) + 1,  # PyPDFLoader pages are 0-based
        text=doc.page_content,
        score=score,
    )


TABLE_MIN_ROWS = 3        # number-heavy lines in a row before it counts as a table
TABLE_NUMERIC_RATIO = 0.25  # share of a line's words that contain a digit
TABLE_MAX_GAP = 15        # short non-numeric lines like "(A)" may sit inside a table
TABLE_CHUNK_CHARS = 1000  # tables get bigger chunks than prose
CAPTION_RE = re.compile(r"^\s*Table\s+\d+[:.]", re.I)


def is_table_row(line):
    words = line.split()
    numeric = sum(any(ch.isdigit() for ch in w) for w in words)
    return numeric >= 2 and numeric / len(words) >= TABLE_NUMERIC_RATIO


def split_tables(text):
    """Split page text into (prose, tables).

    A table is a run of number-heavy lines. Plain chunking cuts tables into pieces
    that lose the caption and column headers ("Table 2 ... BLEU EN-DE EN-FR"), so a
    row like "Transformer (base model) 27.3 38.1" can't be found or read. Here each
    table becomes its own chunks, and every chunk repeats the caption and header.
    """
    lines = text.split("\n")
    tables, used = [], set()
    i = 0
    while i < len(lines):
        if not is_table_row(lines[i]):
            i += 1
            continue
        # extend the run, allowing short label lines such as "(A)" in between
        end, rows = i, 0
        j = i
        while j < len(lines):
            if is_table_row(lines[j]):
                rows, end = rows + 1, j
            elif len(lines[j].strip()) >= TABLE_MAX_GAP:
                break
            j += 1
        if rows < TABLE_MIN_ROWS:
            i += 1
            continue

        # caption + header: back up to a "Table N:" line within 8 lines, else 2 lines of header
        start = i
        for k in range(i - 1, max(-1, i - 9), -1):
            if k in used:
                break
            if CAPTION_RE.match(lines[k]):
                start = k
                break
        else:
            start = max(i - 2, 0)
            while start in used:
                start += 1
        prefix = "\n".join(lines[start:i]).strip()
        body = lines[i:end + 1]

        # pack rows into chunks, each starting with the caption + header
        chunk, size = [], len(prefix)
        for row in body:
            if chunk and size + len(row) + 1 > TABLE_CHUNK_CHARS:
                tables.append("\n".join([prefix] + chunk).strip())
                chunk, size = [], len(prefix)
            chunk.append(row)
            size += len(row) + 1
        tables.append("\n".join([prefix] + chunk).strip())

        used.update(range(start, end + 1))
        i = end + 1

    prose = "\n".join(line for n, line in enumerate(lines) if n not in used)
    return prose, tables


def tokenize(text):
    """Lowercase words for BM25. Keeps decimals like 28.4 as one token."""
    return re.findall(r"\w+(?:\.\d+)?", text.lower())


def chunk_key(doc):
    return (doc.metadata.get("file_name"), doc.metadata.get("page"), doc.page_content)


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Merge several best-first lists of docs into one best-first list."""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def parse_score(text: str) -> int:
    """Pull the first integer out of the model's reply ("8", "8.", "Score: 8/10")."""
    match = re.search(r"\d+", text)
    if not match:
        return 0
    return max(0, min(10, int(match.group())))


class PDFRAG:
    def __init__(self, llm_model=None, embedding_model=None, llm=None,
                 reranker="cross-encoder", search="vector"):
        if reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {RERANKERS}, got {reranker!r}")
        if search not in SEARCH_MODES:
            raise ValueError(f"search must be one of {SEARCH_MODES}, got {search!r}")
        self.search = search
        self.chunks = []  # every ingested chunk, for BM25
        self._bm25 = None  # rebuilt lazily after ingest
        llm_model = llm_model or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )

        self.reranker = reranker
        self.embedding = FastEmbedEmbeddings(model_name=embedding_model)
        # `llm` lets tests (or other backends) pass in any LangChain chat model.
        self.model = llm or ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        self.vector_store = None
        self._cross_encoder = None  # loaded on first use

    def use_model(self, llm_model):
        """Switch the Ollama chat model (the sidebar setting)."""
        self.model = ChatOllama(model=llm_model)

    def _chain(self, prompt):
        return prompt | self.model | StrOutputParser()

    @property
    def scoring_chain(self):
        return self._chain(SCORING_PROMPT)

    @property
    def answer_chain(self):
        return self._chain(ANSWER_PROMPT)

    @property
    def rewrite_chain(self):
        return self._chain(REWRITE_PROMPT)

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = TextCrossEncoder(DEFAULT_RERANKER_MODEL)
        return self._cross_encoder

    def ingest(self, pdf_path: str, file_name: str = None):
        """Add a PDF to the store. `file_name` is what citations show (defaults to the path's name)."""
        docs = PyPDFLoader(file_path=pdf_path).load()
        file_name = file_name or os.path.basename(pdf_path)
        chunks = []
        for doc in docs:
            doc.metadata["file_name"] = file_name
            prose, tables = split_tables(doc.page_content)
            chunks += [Document(page_content=t, metadata={**doc.metadata, "table": True})
                       for t in tables]
            chunks += self.text_splitter.split_documents(
                [Document(page_content=prose, metadata=doc.metadata)]
            )
        chunks = filter_complex_metadata(chunks)
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._bm25 = None

        if self.vector_store is None:
            # In-memory store with a unique collection, so nothing piles up on disk
            # and separate sessions never share chunks.
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                collection_name=f"chatpdf-{uuid.uuid4().hex}",
            )
        else:
            # Further PDFs are added to the same store instead of replacing it.
            self.vector_store.add_documents(chunks)

    def rerank(self, query, docs):
        """Return (score, doc) pairs, best first."""
        if self.reranker == "cross-encoder":
            scores = self.cross_encoder.rerank(query, [d.page_content for d in docs])
        elif self.reranker == "llm":
            scores = [self._llm_score(query, d.page_content) for d in docs]
        else:
            scores = [-i for i in range(len(docs))]  # keep vector search order
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [(float(s), d) for s, d in ranked]

    def _llm_score(self, query, passage):
        try:
            return parse_score(self.scoring_chain.invoke({"question": query, "passage": passage}))
        except Exception:
            return 0

    def keyword_search(self, query, k=RETRIEVE_K):
        """BM25 over every ingested chunk, best first."""
        if not self.chunks:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])
        scores = self._bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in best if scores[i] > 0]

    def retrieve(self, query):
        """Vector search for RETRIEVE_K chunks, rerank, keep the best TOP_N."""
        docs = self.vector_store.similarity_search(query, k=RETRIEVE_K)
        if self.search == "hybrid":
            docs = reciprocal_rank_fusion([docs, self.keyword_search(query)])[:RETRIEVE_K]
        return self.rerank(query, docs)[:TOP_N]

    def standalone_question(self, query, history=None):
        """Rewrite a follow-up using recent history. The first question is used as is."""
        if not history:
            return query
        rewritten = self.rewrite_chain.invoke(
            {"history": format_history(history), "question": query}
        ).strip()
        return rewritten or query

    def _prepare(self, query, history):
        """Rewrite the question, retrieve sources, and build the answer-prompt inputs."""
        question = self.standalone_question(query, history)
        top = self.retrieve(question)
        context = "\n\n".join(doc.page_content for _, doc in top)
        sources = [to_source(score, doc) for score, doc in top]
        return question, sources, {"context": context, "question": question}

    def ask(self, query: str, history=None) -> Answer:
        if not self.vector_store:
            return Answer(NO_DOCUMENT_MESSAGE, question=query)
        question, sources, inputs = self._prepare(query, history)
        return Answer(self.answer_chain.invoke(inputs), sources, question)

    def stream(self, query: str, history=None):
        """Like ask(), but returns (Answer without text, iterator of answer tokens)."""
        if not self.vector_store:
            return Answer("", question=query), iter([NO_DOCUMENT_MESSAGE])
        question, sources, inputs = self._prepare(query, history)
        return Answer("", sources, question), self.answer_chain.stream(inputs)

    def clear(self):
        if self.vector_store is not None:
            self.vector_store.delete_collection()
        self.vector_store = None
        self.chunks = []
        self._bm25 = None
