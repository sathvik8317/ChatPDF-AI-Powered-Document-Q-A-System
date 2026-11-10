import os
import sys

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag  # noqa: E402


def write_pdf(path, pages):
    """Write a minimal PDF, one string per page; newlines become separate lines (no extra dependency)."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None]
    font_id = 3
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for text in pages:
        shown = []
        for line in text.split("\n"):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            shown.append(f"({safe}) Tj")
        stream = ("BT /F1 10 Tf 12 TL 40 760 Td " + " T* ".join(shown) + " ET").encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
            b"/Resources << /Font << /F1 %d 0 R >> >> >>" % (content_id, font_id)
        )
        page_ids.append(len(objects))
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    objects[1] = b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % len(page_ids)

    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)
    return str(path)


@pytest.fixture(scope="session")
def make_rag():
    """Build a PDFRAG whose LLM is a scripted fake, so tests never need Ollama."""

    def _make(responses, **kwargs):
        return rag.PDFRAG(llm=FakeListChatModel(responses=responses), **kwargs)

    return _make


@pytest.fixture
def pdfs(tmp_path):
    atlantis = write_pdf(
        tmp_path / "atlantis.pdf",
        ["The capital of Atlantis is Coralport.", "Coralport was founded in 1312 by Queen Maris."],
    )
    zebras = write_pdf(tmp_path / "zebras.pdf", ["Zebras sleep standing up in the savanna."])
    return atlantis, zebras
