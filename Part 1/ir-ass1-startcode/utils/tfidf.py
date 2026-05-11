"""TF-IDF variants (Task A-3)
IDF = ln(N / df) with natural log. TF modes:
- 'raw'  : length-normalised counts tf/|d| (baseline; required for correctness)
- 'log'  : 1 + ln(tf) for tf>0; 0 otherwise (damps repetition)
- 'bm25' : ((k+1)*tf)/(k+tf), k=1.2 (bounded saturation)

Returns (matrix [N×V], vocab {term→col}). Empty inputs yield valid zero shapes.
"""

import numpy as np
from typing import List, Dict, Tuple

__all__ = ["tfidf_variants"]

def _build_vocab(docs: List[List[str]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for doc in docs:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def _df(docs: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    V = len(vocab)
    df = np.zeros(V, dtype=np.int32)
    for doc in docs:
        seen = set()
        for tok in doc:
            j = vocab[tok]
            if j not in seen:
                df[j] += 1
                seen.add(j)
    return df

def _idf(docs: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    N = len(docs)
    V = len(vocab)
    if N == 0 or V == 0:
        return np.zeros((V,), dtype=np.float32)
    df = _df(docs, vocab)
    # Natural log; df >= 1 for terms in vocab, but clamp for safety.
    return np.log(N / np.maximum(1, df)).astype(np.float32)

def _tf_row(doc: List[str], vocab: Dict[str, int], mode: str, k: float) -> np.ndarray:
    V = len(vocab)
    tf = np.zeros(V, dtype=np.float32)
    if V == 0:
        return tf
    if not doc:
        return tf

    # raw counts
    for tok in doc:
        tf[vocab[tok]] += 1.0

    if mode == "raw":
        # length-normalised raw TF: tf/|d|
        L = float(len(doc))
        if L > 0:
            tf = tf / L
        else:
            tf[:] = 0.0
    elif mode == "log":
        # avoid computing log(0) at all
        cnt = tf  # counts vector from above
        tf = np.zeros_like(cnt, dtype=np.float32)
        mask = cnt > 0
        tf[mask] = 1.0 + np.log(cnt[mask])
    elif mode == "bm25":
        # BM25-style saturation (no doc-length norm here):
        # ((k+1)*tf) / (k + tf)
        tf = np.where(tf > 0, ((k + 1.0) * tf) / (k + tf), 0.0)
    else:
        raise ValueError("tf_mode must be one of {'raw','log','bm25'}")

    return tf

def tfidf_variants(
    docs: List[List[str]],
    tf_mode: str = "raw",
    k: float = 1.2
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a TF-IDF matrix with selectable TF formulation.
    Rationale:
    - Natural-log IDF penalises corpus-common terms (df≈N) toward 0.
    - 'raw' guards against length bias; 'log' and 'bm25' control term burstiness.
    - No IDF smoothing 
    Returns a float32 matrix for efficient downstream cosine similarity.
    """

    if not isinstance(docs, list):
        raise TypeError("docs must be a list of token lists")

    vocab = _build_vocab(docs)
    idf = _idf(docs, vocab)  # [V]
    N, V = len(docs), len(vocab)

    if N == 0 or V == 0:
        return np.zeros((N, V), dtype=np.float32), vocab

    rows = []
    for doc in docs:
        tf = _tf_row(doc, vocab, tf_mode, k)  # [V]
        rows.append(tf * idf)                 # TF-IDF = TF * IDF

    return np.vstack(rows).astype(np.float32), vocab

