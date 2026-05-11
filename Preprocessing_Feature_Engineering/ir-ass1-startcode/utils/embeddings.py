"""Light-weight GloVe loader & document embedding aggregation (Task A-4).
Implements:
- Baselines: 'mean', 'max', 'sum'
- Advanced:  'tfidf_weighted'  
- Advanced:  'meanmax' 
"""

from __future__ import annotations

import io
import zipfile
import urllib.request
import pathlib
from typing import Dict, List

import numpy as np

__all__ = ["semantic_vector"]

# ----------------------------------------------------------------------
# 1) Load a 100-d slice of GloVe (or from cache) and add a deterministic <unk>
# ----------------------------------------------------------------------
_GLOVE_URL = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.100d.txt")
_CACHE = pathlib.Path.home() / ".cache" / "ir_glove_100.txt"


def _ensure_glove() -> Dict[str, np.ndarray]:
    """Fetch glove.6B.100d and cache a plain-text copy locally if missing."""
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _CACHE.exists():
        url, fname = _GLOVE_URL
        # Download the zip in-memory, then extract the 100d file
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            text = zf.read(fname).decode("utf-8")
        _CACHE.write_text(text, encoding="utf-8")

    vocab: Dict[str, np.ndarray] = {}
    with _CACHE.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if not parts or len(parts) < 2:
                continue
            word, *vec = parts
            try:
                vocab[word] = np.asarray(vec, dtype=np.float32)
            except ValueError:
                # Skip malformed lines defensively
                continue

    # Ensure deterministic <unk> exists
    if "<unk>" not in vocab:
        rng = np.random.default_rng(seed=42)
        dim = len(next(iter(vocab.values())))
        vocab["<unk>"] = rng.normal(0.0, 0.05, size=dim).astype(np.float32)

    return vocab


# Global embedding table
_WORD_VEC: Dict[str, np.ndarray] = _ensure_glove()
_DIM: int = int(next(iter(_WORD_VEC.values())).shape[0])


# ----------------------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------------------
def _key(tok: str) -> str:
    """Return token if known, else '<unk>'."""
    return tok if tok in _WORD_VEC else "<unk>"


def _doc_matrix(tokens: List[str]) -> np.ndarray:
    """Stack embeddings for each token occurrence (OOV mapped to <unk>)."""
    if not tokens:
        return np.zeros((0, _DIM), dtype=np.float32)
    rows = [_WORD_VEC[_key(t)] for t in tokens]
    return np.vstack(rows).astype(np.float32)


# ----------------------------------------------------------------------
# 3) Main entry
# ----------------------------------------------------------------------
def semantic_vector(docs: List[List[str]], method: str = "mean") -> np.ndarray:
    """
    Aggregate token embeddings into document vectors.
    For 'mean'/'max'/'sum'/'tfidf_weighted': shape [N, D]
    For 'meanmax':                           shape [N, 2D]
    Weighted averages divide by sum of weights to stabilise vector magnitudes
            """
    docs = [([] if d is None else list(d)) for d in (docs or [])]
    N = len(docs)

    # -------- Baselines: mean / max / sum / meanmax --------
    if method in {"mean", "max", "sum", "meanmax"}:
        out: List[np.ndarray] = []
        for tokens in docs:
            M = _doc_matrix(tokens)  # [L, D]
            if M.shape[0] == 0:
                mean = np.zeros(_DIM, dtype=np.float32)
                mmax = np.zeros(_DIM, dtype=np.float32)
                summ = np.zeros(_DIM, dtype=np.float32)
            else:
                mean = M.mean(axis=0)
                mmax = M.max(axis=0)
                summ = M.sum(axis=0)
            if method == "mean":
                out.append(mean)
            elif method == "max":
                out.append(mmax)
            elif method == "sum":
                out.append(summ)
            else:  # meanmax
                out.append(np.concatenate([mean, mmax], axis=0).astype(np.float32))
        if not out:
            return np.zeros((0, _DIM if method != "meanmax" else 2 * _DIM), dtype=np.float32)
        return np.vstack(out)

# -------- Advanced: TF-IDF weighted average (uses RAW TF-IDF from Task 3) --------
    if method == "tfidf_weighted":
        mapped_docs: List[List[str]] = [[_key(t) for t in tokens] for tokens in docs]

        try:
            from utils.tfidf import tfidf_variants      
        except ModuleNotFoundError:
            from tfidf import tfidf_variants            

        tfm, vocab = tfidf_variants(mapped_docs, tf_mode="raw")  # [N, V]

        N, V = tfm.shape
        if N == 0 or V == 0:
            return np.zeros((N, _DIM), dtype=np.float32)

        E = np.zeros((V, _DIM), dtype=np.float32)
        for term, j in vocab.items():
            E[j] = _WORD_VEC[term]   # <unk> guaranteed

        out = []
        for i in range(N):
            w = tfm[i]; s = float(w.sum())
            out.append((w @ E)/s if s>0 else np.zeros(_DIM, np.float32))
        return np.vstack(out)


