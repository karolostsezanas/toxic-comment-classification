"""Text preprocessing and token-frequency helpers for toxic comment analysis."""

import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[A-Za-z']+")

def basic_clean(text: str) -> str:
    """Lowercase text and normalize URLs, numbers, and punctuation."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphabetic tokens with apostrophes kept."""
    return TOKEN_RE.findall(str(text).lower())

def useful_tokens(text: str) -> list[str]:
    """Filter tokens to non-trivial words outside the English stopword list."""
    tokens = tokenize(text)
    return [
        token for token in tokens
        if len(token) > 2 and token not in ENGLISH_STOP_WORDS
    ]

def top_words(texts, n: int = 25) -> list[tuple[str, int]]:
    """Return the most common useful tokens across an iterable of texts."""
    counter = Counter()
    for text in texts:
        counter.update(useful_tokens(text))
    return counter.most_common(n)
