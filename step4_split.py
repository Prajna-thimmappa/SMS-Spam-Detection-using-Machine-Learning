# Step 4: Stratified train/test split
import joblib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y, vectorizer = joblib.load("tfidf.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    joblib.dump((X_train, X_test, y_train, y_test), "split.joblib")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train spam ratio:", round(y_train.mean(), 4), "Test spam ratio:", round(y_test.mean(), 4))
    print("Saved split to split.joblib")
