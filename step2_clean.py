# Step 2: Clean the text
# pip install nltk  (then run once: python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')")

import re
import string
import joblib
import pandas as pd
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

if __name__ == "__main__":
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["clean"] = df["message"].apply(clean_text)
    joblib.dump(df, "df_clean.joblib")
    print(df[["message", "clean"]].head())
    print("Saved cleaned dataframe to df_clean.joblib")
