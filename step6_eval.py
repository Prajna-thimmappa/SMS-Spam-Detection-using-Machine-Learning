# Step 6: Evaluate with precision, recall, F1, confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# for name, model in [("NB", nb), ("LR", lr), ("SVM", svm)]:
#     pred = model.predict(X_test)
#     print(name)
#     print(classification_report(y_test, pred, target_names=["ham", "spam"]))
#     cm = confusion_matrix(y_test, pred)
#     ConfusionMatrixDisplay(cm, display_labels=["ham","spam"]).plot()
#     plt.title(name); plt.show()

print("Use classification_report + confusion_matrix; focus on spam recall/precision.")
