# Step 7: Build a scikit-learn Pipeline to avoid data leakage
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    X = df["message"]
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0))

    joblib.dump(pipe, "spam_pipeline.joblib")
    print("Saved pipeline to spam_pipeline.joblib")
    print("Demo:", list(pipe.predict(["Free prize win now", "See you at 5"])))
