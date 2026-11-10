# rag.py

import os
import re
import uuid

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

RETRIEVE_K = 20
TOP_N = 5


def parse_score(text: str) -> int:
    """Pull the first integer out of the model's reply ("8", "8.", "Score: 8/10")."""
    match = re.search(r"\d+", text)
    if not match:
        return 0
    return max(0, min(10, int(match.group())))


class PDFRAG:
    def __init__(self, llm_model=None, embedding_model=None, llm=None):
        llm_model = llm_model or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )

        self.embedding = FastEmbedEmbeddings(model_name=embedding_model)
        # `llm` lets tests (or other backends) pass in any LangChain chat model.
        self.model = llm or ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        self.vector_store = None

        # Prompt to score each chunk
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rate the passage for how useful it is for answering the question (1-10). Only output a single number."),
            ("human", "Question: {question}\n\nPassage: {passage}")
        ])
        self.scoring_chain = self.scoring_prompt | self.model | StrOutputParser()

        # Final prompt to answer the question, grounded in the retrieved context
        self.final_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You answer questions about the user's documents. "
             "Use only the context provided. If the context does not contain the answer, "
             "say you could not find it in the document. Do not make up facts."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        self.answer_chain = self.final_prompt | self.model | StrOutputParser()

    def ingest(self, pdf_path: str):
        docs = PyPDFLoader(file_path=pdf_path).load()
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

    def ask(self, query: str):
        if not self.vector_store:
            return "Please upload and ingest a PDF document first."

        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": RETRIEVE_K}
        )
        docs = retriever.invoke(query)

        # Re-rank documents using LLM scoring
        scored_chunks = []
        for doc in docs:
            try:
                reply = self.scoring_chain.invoke(
                    {"question": query, "passage": doc.page_content}
                )
                score = parse_score(reply)
            except Exception:
                score = 0
            scored_chunks.append((score, doc.page_content))

        # Sort and select the highest scoring chunks
        top_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:TOP_N]
        refined_context = "\n\n".join(chunk for _, chunk in top_chunks)

        return self.answer_chain.invoke({"context": refined_context, "question": query})

    def clear(self):
        if self.vector_store is not None:
            self.vector_store.delete_collection()
        self.vector_store = None
