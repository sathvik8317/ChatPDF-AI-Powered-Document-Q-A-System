# rag.py

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import shutil
import time


class ChatPDFAlternative4_v2:
    def __init__(self, llm_model="qwen2.5:7b"):
        self.embedding = FastEmbedEmbeddings(model="answerdotai/answerai-colbert-small-v1")
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        self.vector_store = None

        # Prompt to score each chunk
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rate the passage for how useful it is for answering the question (1-10). Only output a single number."),
            ("human", "Question: {question}\n\nPassage: {passage}")
        ])

        # Final prompt to answer the question
        self.final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable assistant."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

    def ingest(self, pdf_path: str):
        timestamp = str(int(time.time()))
        unique_dir = f"chroma_alt4_{timestamp}"

        if os.path.exists(unique_dir):
            shutil.rmtree(unique_dir)

        docs = PyPDFLoader(file_path=pdf_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=unique_dir,
        )

    def ask(self, query: str):
        if not self.vector_store:
            return "Please upload and ingest a PDF document first."

        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        docs = retriever.invoke(query)

        # Re-rank documents using LLM scoring
        scored_chunks = []
        for doc in docs:
            scoring_chain = (
                RunnableLambda(lambda _: {"question": query, "passage": doc.page_content})
                | self.scoring_prompt
                | self.model
                | StrOutputParser()
            )

            try:
                score = int(scoring_chain.invoke({}).strip())
            except:
                score = 0

            scored_chunks.append((score, doc.page_content))

        # Sort and select top 5 highest scoring
        top_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:5]
        refined_context = "\n\n".join(chunk for _, chunk in top_chunks)

        # Final answer chain
        answer_chain = (
            {"context": lambda _: refined_context, "question": RunnablePassthrough()}
            | self.final_prompt
            | self.model
            | StrOutputParser()
        )

        return answer_chain.invoke(query)

    def clear(self):
        self.vector_store = None