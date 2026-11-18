# Step 7: Build a scikit-learn Pipeline to avoid data leakage
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000)),
])

# Fit on raw text (cleaning can be a custom transformer or done in tfidf preprocessor)
# pipe.fit(X_train_text, y_train)
# pipe.predict(["Free prize, call now!"])
# joblib.dump(pipe, "spam_pipeline.joblib")
print("Pipeline: tfidf -> classifier; fit on raw text, predict on raw text.")
