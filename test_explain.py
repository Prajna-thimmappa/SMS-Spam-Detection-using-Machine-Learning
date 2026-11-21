import os
import joblib

from explain import explain_prediction, format_explanation


def _pipe():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spam_pipeline.joblib")
    return joblib.load(path)


def test_explain_returns_list():
    pipe = _pipe()
    result = explain_prediction(pipe, "WIN cash now!!!")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple) and len(item) == 2
        assert isinstance(item[0], str) and isinstance(item[1], float)


def test_explain_spam_words_score_positive():
    pipe = _pipe()
    result = explain_prediction(pipe, "WIN cash now!!! free prize")
    top_word = result[0][0] if result else ""
    assert any(w in {"win", "cash", "free", "prize"} for w, _ in result)
    # Highest-scoring present word should be a spam indicator
    assert top_word in {"win", "cash", "free", "prize"}


def test_explain_respects_top_n():
    pipe = _pipe()
    result = explain_prediction(pipe, "WIN cash now!!! free prize click link call", top_n=3)
    assert len(result) <= 3


def test_format_explanation_nonempty():
    pipe = _pipe()
    result = explain_prediction(pipe, "WIN cash now!!!")
    text = format_explanation(result)
    assert "Top spam indicators:" in text
