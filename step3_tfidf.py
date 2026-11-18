# Step 3: Convert text to numbers using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Assume df has a "clean" column from Step 2
# vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
# X = vectorizer.fit_transform(df["clean"])   # learn vocab + transform
# y = (df["label"] == "spam").astype(int)    # 1=spam, 0=ham
# print("Shape:", X.shape)
# print("Top features:", vectorizer.get_feature_names_out()[:20])

# Demo
corpus = ["free prize win now", "hello how are you", "win cash prize today"]
v = TfidfVectorizer()
X = v.fit_transform(corpus)
print(v.get_feature_names_out())
print(X.toarray())
