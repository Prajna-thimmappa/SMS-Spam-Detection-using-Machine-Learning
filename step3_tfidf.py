# Step 3: Convert text to numbers using TF-IDF
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    df = joblib.load("df_clean.joblib")
    y = (df["label"] == "spam").astype(int)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean"])   # learn vocab + transform

    joblib.dump((X, y, vectorizer), "tfidf.joblib")
    print("Shape:", X.shape)
    print("Top 20 features:", list(vectorizer.get_feature_names_out()[:20]))
    print("Saved TF-IDF features + labels + vectorizer to tfidf.joblib")
