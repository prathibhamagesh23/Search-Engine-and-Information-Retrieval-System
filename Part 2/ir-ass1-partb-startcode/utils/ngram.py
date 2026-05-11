"""
Word- and char-level n-gram (Task A-2).

- make_ngrams_tokens: word n-grams with <s>/</s> padding for phrase/proximity.
- make_ngrams_chars: character n-grams with '$' word boundaries for wildcarding.

Both functions are O(L) with a single sliding window; outputs are deterministic.
"""
from typing import List, Tuple

__all__ = ["make_ngrams_tokens", "make_ngrams_chars"]

def make_ngrams_tokens(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    #Build word-level n-grams with <s> and </s> padding (document boundaries).
    if n < 1:
        raise ValueError("n must be >= 1")
    seq = ["<s>"] + (tokens or []) + ["</s>"] # pad with sentence/document markers
    return [tuple(seq[i:i+n]) for i in range(0, max(0, len(seq) - n + 1))]

def make_ngrams_chars(text: str, n: int) -> List[str]:
    """Return character n-grams with per-word '$' boundaries.

    We split on whitespace and emit n-grams inside each word separately:
    - Prefix: '$' + word[:n-1]
    - Suffix: word[-(n-1):] + '$'
    No cross-word n-grams are produced. Useful for wildcard and fuzzy matching.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not text:
        return []
    grams: List[str] = []
    for word in text.split():
        padded = f"${word}$"
        L = len(padded)
        if L >= n:
            grams.extend(padded[i:i+n] for i in range(0, L - n + 1))
    return grams

