# Step 5: Train Naive Bayes, Logistic Regression, and Linear SVM
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = joblib.load("split.joblib")

    models = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=1000),
        "svm": LinearSVC(max_iter=1000),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"model_{name}.joblib")
        print(f"Trained and saved {name} -> model_{name}.joblib")
