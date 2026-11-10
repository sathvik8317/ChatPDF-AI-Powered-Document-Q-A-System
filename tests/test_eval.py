import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import run_eval  # noqa: E402
from rag import Source  # noqa: E402


def test_numbers_match_whole_words_only():
    item = {"expected": ["4|four"]}
    assert run_eval.is_correct(item, "They used a beam size of 4.")
    assert not run_eval.is_correct(item, "They used 24 beams.")


def test_all_expected_phrases_must_appear():
    item = {"expected": ["sine", "cosine"]}
    assert run_eval.is_correct(item, "Sine and cosine functions.")
    assert not run_eval.is_correct(item, "Only sine.")


def test_unanswerable_needs_a_not_found_reply():
    item = {"unanswerable": True}
    assert run_eval.is_correct(item, "I couldn't find that in the document.")
    assert not run_eval.is_correct(item, "GPT-3 has 175 billion parameters.")


def test_hit_checks_source_pages():
    item = {"pages": [7]}
    assert run_eval.is_hit(item, [Source("a.pdf", 3, "", 1.0), Source("a.pdf", 7, "", 0.5)])
    assert not run_eval.is_hit(item, [Source("a.pdf", 3, "", 1.0)])


def test_question_file_is_well_formed():
    with open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    assert len(questions) >= 20
    for q in questions:
        assert q["question"].endswith("?")
        assert q.get("unanswerable") or (q["expected"] and q["pages"])
