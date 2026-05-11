"""TF-IDF variants (Task A-3)."""
import math, numpy as np
from typing import List, Dict, Tuple

def _idf(df: int, N: int) -> float:
    return math.log((N) / df)

def tfidf_variants(
        docs: List[List[str]],
        tf_mode: str = "raw",
        k: float = 1.2
) -> Tuple[np.ndarray, Dict[str, int]]:
    pass
