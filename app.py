# Step 9: Simple Streamlit app (save as app.py)
# pip install streamlit joblib
# streamlit run app.py

import streamlit as st
import joblib

st.title("SMS Spam Classifier")
pipe = joblib.load("spam_pipeline.joblib")

msg = st.text_area("Enter an SMS message")
if st.button("Predict"):
    if msg.strip():
        pred = pipe.predict([msg])[0]
        proba = pipe.predict_proba([msg])[0]
        label = "SPAM" if pred == 1 else "HAM"
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: spam={proba[1]:.2f}, ham={proba[0]:.2f}")
    else:
        st.warning("Please enter a message.")
