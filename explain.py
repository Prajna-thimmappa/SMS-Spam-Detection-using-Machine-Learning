"""Model explainability: top words driving a spam/ham prediction."""

import numpy as np


def explain_prediction(pipe, message, top_n=10):
    """Return top contributing words for a spam prediction.

    Works with a scikit-learn Pipeline that has a ``tfidf`` vectorizer step
    and a classifier step exposing either ``coef_`` (LinearSVC/LogisticRegression)
    or ``feature_log_prob_`` (MultinomialNB).
    """
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    feature_names = np.array(tfidf.get_feature_names_out())

    # Transform the message into a TF-IDF vector (sparse).
    vec = tfidf.transform([message])
    indices = vec.indices
    words_present = feature_names[indices]

    if hasattr(clf, "coef_"):
        # Linear model: weight per class. Use spam class (index 1) weights.
        weights = clf.coef_[1] if clf.coef_.ndim == 2 else clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        # MultinomialNB: log P(word|spam) - log P(word|ham).
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return []

    # Score each word present in the message by its class weight.
    scores = []
    for word in words_present:
        idx = np.where(feature_names == word)[0]
        if len(idx) == 0:
            continue
        scores.append((word, float(weights[idx[0]])))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def format_explanation(explanation):
    """Human-readable string for a list of (word, score) tuples."""
    if not explanation:
        return "No explanation available."
    parts = [f"{word} ({score:+.3f})" for word, score in explanation]
    return "Top spam indicators: " + ", ".join(parts)
