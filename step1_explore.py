# Step 1: Load and explore the SMS Spam Collection data
# Run: python step1_explore.py (after: pip install pandas matplotlib seaborn)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load data. The UCI/Kaggle file is usually named "spam.csv"
#    It has columns: v1 (label: ham/spam), v2 (message), plus 3 empty columns.
df = pd.read_csv("spam.csv", encoding="utf-8")

# Keep only the useful columns and rename them
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# 2) Basic info
print("Shape:", df.shape)
print("\nLabel counts:\n", df["label"].value_counts())
print("\nClass proportions:\n", df["label"].value_counts(normalize=True))

# 3) Message length (number of characters)
df["length"] = df["message"].apply(len)

# 4) Plot class balance
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=df)
plt.title("Class Balance: ham vs spam")
plt.show()

# 5) Plot message length distribution by class
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="length", hue="label", bins=50, kde=True)
plt.title("Message Length by Class")
plt.show()

# 6) Average length per class
print("\nAverage message length by class:\n", df.groupby("label")["length"].mean())

# 7) Most common words (quick view, before cleaning)
from collections import Counter
all_words = " ".join(df["message"]).lower().split()
print("\nTop 15 raw words:", Counter(all_words).most_common(15))
