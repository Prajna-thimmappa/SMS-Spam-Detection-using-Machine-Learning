# Step 5: Train Naive Bayes, Logistic Regression, and Linear SVM
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Assume X_train, y_train ready
# nb = MultinomialNB(); nb.fit(X_train, y_train)
# lr = LogisticRegression(max_iter=1000); lr.fit(X_train, y_train)
# svm = LinearSVC(); svm.fit(X_train, y_train)

# MultinomialNB needs non-negative counts/TF-IDF -> ok with TF-IDF.
# LinearSVC has no predict_proba; use decision_function if needed.
print("Models: MultinomialNB, LogisticRegression, LinearSVC")
