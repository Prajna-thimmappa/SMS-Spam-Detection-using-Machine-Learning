# backend.py - FastAPI backend for SMS spam prediction
# pip install fastapi uvicorn joblib
# run: uvicorn backend:app --reload
import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from explain import explain_prediction, format_explanation

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "spam_pipeline.joblib")

app = FastAPI(title="SMS Spam Classifier API")
pipe = joblib.load(MODEL_PATH)

class Message(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(msg: Message):
    pred = int(pipe.predict([msg.text])[0])
    proba = pipe.predict_proba([msg.text])[0].tolist()
    return {"label": "spam" if pred == 1 else "ham", "spam_prob": proba[1], "ham_prob": proba[0]}

@app.post("/explain")
def explain(msg: Message):
    pred = int(pipe.predict([msg.text])[0])
    explanation = explain_prediction(pipe, msg.text)
    return {
        "label": "spam" if pred == 1 else "ham",
        "top_indicators": [{"word": w, "score": round(s, 4)} for w, s in explanation],
        "summary": format_explanation(explanation),
    }
