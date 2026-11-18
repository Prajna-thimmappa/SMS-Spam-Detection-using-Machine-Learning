# Step 4: Stratified train/test split
from sklearn.model_selection import train_test_split

# X = tfidf features, y = labels (0=ham,1=spam)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# print("Train:", X_train.shape, "Test:", X_test.shape)
# print("Train spam ratio:", y_train.mean(), "Test spam ratio:", y_test.mean())

# Demo with dummy labels
import numpy as np
y = np.array([0]*87 + [1]*13)
X = np.arange(100).reshape(100,1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("train spam ratio:", ytr.mean(), "test spam ratio:", yte.mean())
