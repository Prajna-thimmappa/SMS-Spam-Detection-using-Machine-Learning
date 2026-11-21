# Step 6: Evaluate with precision, recall, F1, confusion matrix
import joblib
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = joblib.load("split.joblib")

    for name in ["nb", "lr", "svm"]:
        model = joblib.load(f"model_{name}.joblib")
        pred = model.predict(X_test)
        print("==", name.upper())
        print(classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0))
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot()
        plt.title(name.upper()); plt.show()
