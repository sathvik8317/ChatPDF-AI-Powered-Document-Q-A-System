import pytest

import rag


@pytest.mark.parametrize(
    "reply, expected",
    [("8", 8), ("8.", 8), ("Score: 7/10", 7), ("no idea", 0), ("42", 10)],
)
def test_parse_score(reply, expected):
    assert rag.parse_score(reply) == expected


def test_ask_before_ingest(make_rag):
    assert "upload" in make_rag([]).ask("anything").lower()


def test_multiple_pdfs_are_all_searchable(make_rag, pdfs):
    r = make_rag([])
    for path in pdfs:
        r.ingest(path)
    sources = {d.metadata["source"] for d in r.vector_store.similarity_search("x", k=10)}
    assert sources == set(pdfs)


def test_ask_returns_model_answer(make_rag, pdfs):
    # 3 chunks get scored, then the final answer is generated.
    r = make_rag(["9", "Score: 2", "5", "Queen Maris"])
    for path in pdfs:
        r.ingest(path)
    assert r.ask("Who founded Coralport?") == "Queen Maris"


def test_clear_drops_old_documents(make_rag, pdfs):
    atlantis, zebras = pdfs
    r = make_rag([])
    r.ingest(atlantis)
    r.clear()
    assert r.vector_store is None
    r.ingest(zebras)
    sources = {d.metadata["source"] for d in r.vector_store.similarity_search("x", k=10)}
    assert sources == {zebras}
