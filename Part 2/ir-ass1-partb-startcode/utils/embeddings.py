"""Light-weight GloVe loader & semantic aggregation (Task A-4)."""
import io, zipfile, urllib.request, pathlib
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# 1.  Load a 200-d slice of GloVe (or from cache) and add a random <unk>
# ----------------------------------------------------------------------
_GLOVE_URL  = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.200d.txt")
_CACHE      = pathlib.Path.home() / ".cache" / "ir_glove_200.txt"


def _ensure_glove() -> Dict[str, np.ndarray]:
    if not _CACHE.exists():
        url, fname = _GLOVE_URL
        with urllib.request.urlopen(url) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                _CACHE.write_text(zf.read(fname).decode("utf-8"), encoding="utf-8", newline="\n")

    vocab = {}
    with _CACHE.open("r", encoding="utf-8", newline="\n") as f:
        for line in f:
            word, *vec = line.strip().split()
            vocab[word] = np.asarray(vec, dtype=float)

    # deterministic random <unk> – keeps load fast
    if "<unk>" not in vocab:
        rng = np.random.default_rng(seed=42)
        dim = len(next(iter(vocab.values())))
        vocab["<unk>"] = rng.normal(0.0, 0.05, size=dim)

    return vocab


_WORD_VEC: Dict[str, np.ndarray] = _ensure_glove()
_DIM: int = next(iter(_WORD_VEC.values())).shape[0]


# ----------------------------------------------------------------------
# 2.  Public helper
# ----------------------------------------------------------------------
def _key(tok: str) -> str:
    """Return token itself if in vocab, else '<unk>'."""
    return tok if tok in _WORD_VEC else "<unk>"


# ----------------------------------------------------------------------
# 3.  Main entry
# ----------------------------------------------------------------------
def semantic_vector(docs: List[List[str]], method: str = "mean") -> np.ndarray:
    """
    Parameters
    ----------
    docs   : list of token lists
    method : "mean" | "max" | "sum" | "tfidf_weighted" | "meanmax"
    """
    pass