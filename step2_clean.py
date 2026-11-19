# Step 2: Clean the text
# pip install nltk  (then run once: python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')")

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # 1) lowercase
    text = text.lower()
    # 2) remove URLs and numbers
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    # 3) remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 4) tokenize
    tokens = text.split()
    # 5) remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Apply to the dataframe from Step 1
# df["clean"] = df["message"].apply(clean_text)
# print(df[["message", "clean"]].head())

# Demo on a single message
sample = "WINNER!! You have won a FREE prize. Call 0906... now!!!"
print("Original:", sample)
print("Cleaned :", clean_text(sample))
