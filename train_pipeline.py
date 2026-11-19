# End-to-end SMS spam training pipeline (Steps 1-8)
# Run: python train_pipeline.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from preprocess import clean_text

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "spam.csv")
MODEL = os.path.join(BASE, "spam_pipeline.joblib")


def main():
    df = pd.read_csv(DATA, encoding="utf-8")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df = df.dropna(subset=["message"])

    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train spam ratio:", round(y_train.mean(), 3),
          "Test spam ratio:", round(y_test.mean(), 3))

    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(),
    }

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text,
                tokenizer=str.split,
                max_features=3000,
                ngram_range=(1, 2),
            )),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        report = classification_report(y_test, pred, target_names=["ham", "spam"], output_dict=True)
        spam_f1 = report["spam"]["f1-score"]
        results[name] = (pipe, spam_f1)
        print(f"\n== {name}")
        print(classification_report(y_test, pred, target_names=["ham", "spam"]))

    # Choose best model by spam F1 that supports predict_proba (needed by app/backend)
    proba_models = {n: (p, f1) for n, (p, f1) in results.items()
                    if hasattr(p, "predict_proba")}
    best_name = max(proba_models, key=lambda n: proba_models[n][1])
    best_pipe, best_f1 = proba_models[best_name]
    print(f"\nBest model: {best_name} (spam F1={best_f1:.4f})")

    joblib.dump(best_pipe, MODEL)
    print("Saved:", MODEL)

    # Sanity check on custom messages
    sample = ["WIN cash now!!!", "See you tomorrow at lunch",
              "Congratulations! You won a free iPhone. Click here"]
    print("pred:", best_pipe.predict(sample).tolist())
    print("proba:", best_pipe.predict_proba(sample).tolist())


if __name__ == "__main__":
    main()
