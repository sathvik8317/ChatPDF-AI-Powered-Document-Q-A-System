#!/usr/bin/env python3
import os
import tempfile
import time

import streamlit as st  # type: ignore

from rag import DEFAULT_LLM_MODEL, RERANKERS, SEARCH_MODES, PDFRAG

st.set_page_config(page_title="ChatPDF", page_icon="📄")

RERANKER_LABELS = {
    "cross-encoder": "Cross-encoder (fast)",
    "llm": "LLM scoring (slow)",
    "none": "None (vector order)",
}


def get_assistant():
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = PDFRAG()
        st.session_state["messages"] = []
        st.session_state["ingested"] = []
    return st.session_state["assistant"]


def ingest_uploads():
    assistant = get_assistant()
    assistant.clear()
    st.session_state["messages"] = []
    st.session_state["ingested"] = []

    for file in st.session_state["uploads"] or []:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            path = tf.name
        try:
            t0 = time.time()
            assistant.ingest(path, file_name=file.name)
            st.session_state["ingested"].append((file.name, time.time() - t0))
        finally:
            os.remove(path)


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s.file_name}**, page {s.page} · score {s.score:.2f}")
            st.caption(s.text)


def history_pairs(messages):
    """Turn the chat log into (question, answer) pairs for follow-up rewriting."""
    pairs, question = [], None
    for m in messages:
        if m["role"] == "user":
            question = m["content"]
        elif question is not None:
            pairs.append((question, m["content"]))
            question = None
    return pairs


def sidebar(assistant):
    with st.sidebar:
        st.header("Documents")
        st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            key="uploads",
            accept_multiple_files=True,
            on_change=ingest_uploads,
        )
        for name, seconds in st.session_state["ingested"]:
            st.caption(f"✓ {name} ({seconds:.1f}s)")

        st.header("Settings")
        model = st.text_input("Ollama model", os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
        if model and model != getattr(assistant.model, "model", model):
            assistant.use_model(model)
        assistant.reranker = st.radio(
            "Reranker",
            RERANKERS,
            format_func=RERANKER_LABELS.get,
            help="Re-orders the 20 retrieved chunks before the best 5 go to the model.",
        )
        assistant.search = st.radio(
            "Search",
            SEARCH_MODES,
            format_func={"vector": "Vector", "hybrid": "Hybrid (vector + BM25 keywords)"}.get,
            help="Hybrid also matches exact words like model names or numbers.",
        )
        if st.button("Clear chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()


def main():
    assistant = get_assistant()
    sidebar(assistant)

    st.title("ChatPDF")
    if not st.session_state["ingested"]:
        st.info("Upload one or more PDFs in the sidebar, then ask questions about them.")

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources"))

    query = st.chat_input("Ask about your documents")
    if not query:
        return

    history = history_pairs(st.session_state["messages"])
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            answer, tokens = assistant.stream(query, history)
        if answer.question != query:
            st.caption(f"Searched for: {answer.question}")
        text = st.write_stream(tokens)
        show_sources(answer.sources)

    st.session_state["messages"].append(
        {"role": "assistant", "content": text, "sources": answer.sources}
    )


if __name__ == "__main__":
    main()
