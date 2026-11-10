import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_starts_and_asks_for_a_pdf():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "Upload one or more PDFs" in at.info[0].value
    assert at.sidebar.radio[0].value == "cross-encoder"


def test_question_without_pdf_gets_a_hint():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert "upload" in at.session_state["messages"][-1]["content"].lower()
