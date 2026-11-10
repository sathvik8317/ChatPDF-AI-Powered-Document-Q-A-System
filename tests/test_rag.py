import pytest

import rag


@pytest.mark.parametrize(
    "reply, expected",
    [("8", 8), ("8.", 8), ("Score: 7/10", 7), ("no idea", 0), ("42", 10)],
)
def test_parse_score(reply, expected):
    assert rag.parse_score(reply) == expected


def test_ask_before_ingest(make_rag):
    assert "upload" in make_rag([]).ask("anything").text.lower()


def test_multiple_pdfs_are_all_searchable(make_rag, pdfs):
    r = make_rag([])
    for path in pdfs:
        r.ingest(path)
    sources = {d.metadata["source"] for d in r.vector_store.similarity_search("x", k=10)}
    assert sources == set(pdfs)


def test_ask_returns_model_answer(make_rag, pdfs):
    r = make_rag(["Queen Maris"])
    for path in pdfs:
        r.ingest(path)
    assert r.ask("Who founded Coralport?").text == "Queen Maris"


def test_cross_encoder_puts_relevant_chunk_first(make_rag, pdfs):
    r = make_rag([])
    for path in pdfs:
        r.ingest(path)
    _, best = r.retrieve("How do zebras sleep?")[0]
    assert "Zebras" in best.page_content


def test_llm_reranker_orders_by_model_scores(make_rag, pdfs):
    atlantis, _ = pdfs
    r = make_rag(["Score: 2", "9"], reranker="llm")
    r.ingest(atlantis)
    scores = [s for s, _ in r.retrieve("Who founded Coralport?")]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 9


def test_unknown_reranker_is_rejected(make_rag):
    with pytest.raises(ValueError):
        make_rag([], reranker="magic")


def test_clear_drops_old_documents(make_rag, pdfs):
    atlantis, zebras = pdfs
    r = make_rag([])
    r.ingest(atlantis)
    r.clear()
    assert r.vector_store is None
    r.ingest(zebras)
    sources = {d.metadata["source"] for d in r.vector_store.similarity_search("x", k=10)}
    assert sources == {zebras}


def test_answer_cites_file_name_and_page(make_rag, pdfs):
    atlantis, _ = pdfs
    r = make_rag(["Queen Maris"])
    r.ingest(atlantis, file_name="Atlantis Guide.pdf")
    top = r.ask("Who founded Coralport?").sources[0]
    assert top.file_name == "Atlantis Guide.pdf"
    assert top.page == 2
    assert "1312" in top.text


def test_first_question_is_not_rewritten(make_rag, pdfs):
    r = make_rag(["Queen Maris"])  # only one LLM call: the answer
    r.ingest(pdfs[0])
    answer = r.ask("Who founded Coralport?", history=[])
    assert answer.question == "Who founded Coralport?"
    assert answer.text == "Queen Maris"


def test_follow_up_is_rewritten_and_searched(make_rag, pdfs):
    r = make_rag(["How do zebras sleep?", "Standing up"])
    for path in pdfs:
        r.ingest(path)
    history = [("Tell me about zebras", "They live in the savanna.")]
    answer = r.ask("and how do they sleep?", history=history)
    assert answer.question == "How do zebras sleep?"
    assert answer.text == "Standing up"
    assert "Zebras" in answer.sources[0].text


def test_history_keeps_only_recent_turns():
    history = [(f"q{i}", f"a{i}") for i in range(10)]
    text = rag.format_history(history)
    assert "q6" not in text
    assert "q7" in text and "a9" in text
