# Step 8: Test with custom messages
import joblib

if __name__ == "__main__":
    pipe = joblib.load("spam_pipeline.joblib")
    msgs = ["WIN cash now!!!", "See you tomorrow at lunch", "Free entry in 2 a wkly comp to win FA Cup final tkts"]
    for msg in msgs:
        pred = pipe.predict([msg])[0]
        proba = pipe.predict_proba([msg])[0]
        label = "SPAM" if pred == 1 else "HAM"
        print(f"[{label}] spam={proba[1]:.2f} ham={proba[0]:.2f} :: {msg}")
