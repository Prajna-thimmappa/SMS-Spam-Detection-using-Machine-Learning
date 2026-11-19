"""Text preprocessing used across the SMS spam project."""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _res in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Lowercase, strip URLs/numbers/punctuation, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words]
    return " ".join(tokens)
