# rag.py

import os
import re
import uuid
from dataclasses import dataclass, field

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
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


def parse_score(text: str) -> int:
    """Pull the first integer out of the model's reply ("8", "8.", "Score: 8/10")."""
    match = re.search(r"\d+", text)
    if not match:
        return 0
    return max(0, min(10, int(match.group())))


class PDFRAG:
    def __init__(self, llm_model=None, embedding_model=None, llm=None, reranker="cross-encoder"):
        if reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {RERANKERS}, got {reranker!r}")
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
        for doc in docs:
            doc.metadata["file_name"] = file_name
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        if not chunks:
            return

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

    def retrieve(self, query):
        """Vector search for RETRIEVE_K chunks, rerank, keep the best TOP_N."""
        docs = self.vector_store.similarity_search(query, k=RETRIEVE_K)
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
