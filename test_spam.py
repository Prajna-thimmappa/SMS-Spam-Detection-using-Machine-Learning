import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
from preprocess import clean_text


def test_clean_text_basic():
    assert clean_text("WIN cash now!!!") == "win cash"
    assert clean_text("Hello, meeting today?") == "hello meeting today"


def test_clean_text_removes_urls_numbers():
    assert "http" not in clean_text("Click http://example.com now 123")
    assert "123" not in clean_text("Call 555-1234 now")


def test_pipeline_loads_and_predicts():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spam_pipeline.joblib")
    pipe = joblib.load(model_path)
    pred = pipe.predict(["WIN cash now!!!", "See you tomorrow at lunch"])
    assert list(pred) == [1, 0]
    proba = pipe.predict_proba(["WIN cash now!!!"])
    assert proba.shape == (1, 2)
    assert pipe.classes_.tolist() == [0, 1]
