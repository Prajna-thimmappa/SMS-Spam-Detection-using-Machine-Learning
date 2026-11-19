# Step 9: Simple Streamlit app (save as app.py)
# pip install streamlit joblib
# streamlit run app.py

import os
import streamlit as st
import joblib

from explain import explain_prediction

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "spam_pipeline.joblib")

st.title("SMS Spam Classifier")
pipe = joblib.load(MODEL_PATH)

msg = st.text_area("Enter an SMS message")
if st.button("Predict"):
    if msg.strip():
        pred = pipe.predict([msg])[0]
        proba = pipe.predict_proba([msg])[0]
        label = "SPAM" if pred == 1 else "HAM"
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: spam={proba[1]:.2f}, ham={proba[0]:.2f}")

        explanation = explain_prediction(pipe, msg)
        if explanation:
            st.write("Top spam indicators:")
            st.write(", ".join(f"{w} ({s:+.2f})" for w, s in explanation))
    else:
        st.warning("Please enter a message.")
