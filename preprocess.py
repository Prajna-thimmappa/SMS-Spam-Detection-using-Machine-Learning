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

# Common Hindi function words written in Latin script (Hinglish / code-mixed).
_HINGLISH_STOP_WORDS = {
    "hai", "hain", "ho", "hun", "hua", "hui", "hue", "hoga", "hoge", "hogi",
    "tha", "the", "thi", "thee", "raha", "rahe", "rahi", "rahte", "rah",
    "aap", "tum", "main", "mera", "meri", "mere", "tumhara", "tumhare",
    "uska", "uske", "uski", "yah", "wah", "ye", "ve", "kya", "kaise", "kab",
    "kahan", "kahaan", "kyun", "kyunki", "lekin", "par", "aur", "bhi",
    "sirf", "na", "nhi", "nahi", "mat", "jo", "ki", "ke", "ka", "ko", "ne",
    "se", "mein", "pe", "pehle", "baad", "abhi", "ab", "tab", "pahle", "phir",
    "ek", "do", "teen", "chaar", "hoon", "namaste", "dhanyavad", "shukriya",
}


def is_hinglish(text):
    """Return True if the text contains Hinglish (code-mixed Hindi) stop words."""
    tokens = set(str(text).lower().split())
    return bool(tokens & _HINGLISH_STOP_WORDS)


def clean_text(text, remove_hinglish=False):
    """Lowercase, strip URLs/numbers/punctuation, remove stopwords, lemmatize.

    When ``remove_hinglish`` is True, also drop common Hinglish stop words so
    code-mixed Hindi-English messages are normalized the same way as English.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    stop = _stop_words
    if remove_hinglish:
        stop = stop | _HINGLISH_STOP_WORDS
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w not in stop]
    return " ".join(tokens)
