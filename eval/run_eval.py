"""Measure retrieval and answers on "Attention Is All You Need".

    python eval/run_eval.py                          # every search mode x reranker
    python eval/run_eval.py --search hybrid --rerankers cross-encoder none

Needs Ollama running with the model in LLM_MODEL (default qwen2.5:7b).
Writes eval/results.md (table for the README) and eval/results.json (every answer).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import rag  # noqa: E402

PDF_URL = "https://arxiv.org/pdf/1706.03762v7"
PDF_PATH = os.path.join(HERE, ".cache", "attention.pdf")
NOT_FOUND = re.compile(r"(couldn't|could not|cannot|can't) find|not (mentioned|provided|contain)", re.I)


def download_pdf():
    if not os.path.exists(PDF_PATH):
        os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
        print(f"Downloading {PDF_URL}")
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    return PDF_PATH


def phrase_in(phrase, text):
    """Case-insensitive match on word boundaries, so "4" does not match "24"."""
    return re.search(r"(?<![\w.])" + re.escape(phrase) + r"(?![\w])", text, re.I) is not None


def is_correct(item, answer):
    if item.get("unanswerable"):
        return bool(NOT_FOUND.search(answer))
    # every expected entry must match; "a|b" means either a or b
    return all(any(phrase_in(alt, answer) for alt in entry.split("|")) for entry in item["expected"])


def is_hit(item, sources):
    return any(s.page in item["pages"] for s in sources)


def run(assistant, questions):
    rows = []
    for i, item in enumerate(questions, 1):
        t0 = time.time()
        answer = assistant.ask(item["question"])
        seconds = time.time() - t0
        row = {
            "question": item["question"],
            "answer": answer.text.strip(),
            "pages": [s.page for s in answer.sources],
            "correct": is_correct(item, answer.text),
            "hit": None if item.get("unanswerable") else is_hit(item, answer.sources),
            "seconds": round(seconds, 2),
        }
        rows.append(row)
        mark = "✓" if row["correct"] else "✗"
        print(f"  [{i:2}/{len(questions)}] {mark} {seconds:5.1f}s  {item['question']}")
    return rows


def summarize(rows):
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    return {
        "hit_rate": sum(hits) / len(hits),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "seconds": sum(r["seconds"] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerankers", nargs="+", default=list(rag.RERANKERS), choices=rag.RERANKERS)
    parser.add_argument("--search", nargs="+", default=list(rag.SEARCH_MODES), choices=rag.SEARCH_MODES)
    args = parser.parse_args()

    with open(os.path.join(HERE, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)

    assistant = rag.PDFRAG()
    assistant.ingest(download_pdf(), file_name="attention.pdf")
    assistant.ask("warm up")  # load the LLM and reranker so the first timing is fair

    # Results from earlier runs are kept, so slow configs don't have to be re-run.
    results_path = os.path.join(HERE, "results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

    for search in args.search:
        for reranker in args.rerankers:
            name = f"{search} + {reranker}"
            print(f"\n== {name}")
            assistant.search, assistant.reranker = search, reranker
            rows = run(assistant, questions)
            results[name] = {"search": search, "reranker": reranker,
                             "summary": summarize(rows), "rows": rows}
            s = results[name]["summary"]
            print(f"   hit rate {s['hit_rate']:.0%}, accuracy {s['accuracy']:.0%}, {s['seconds']:.1f}s/question")
            # save after every configuration, so a crash halfway keeps what finished
            write_results(results, questions, getattr(assistant.model, "model", "?"), results_path)


def write_results(results, questions, model, results_path):
    answerable = sum(not q.get("unanswerable") for q in questions)
    lines = [
        f"Model `{model}`, {len(questions)} questions ({answerable} answerable, "
        f"{len(questions) - answerable} not in the paper).",
        "",
        "| Search | Reranker | Retrieval hit rate | Answer accuracy | Seconds / question |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results.values(), key=lambda r: (r["search"], r["reranker"])):
        s = r["summary"]
        lines.append(f"| {r['search']} | {r['reranker']} | {s['hit_rate']:.0%} "
                     f"| {s['accuracy']:.0%} | {s['seconds']:.1f} |")

    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
