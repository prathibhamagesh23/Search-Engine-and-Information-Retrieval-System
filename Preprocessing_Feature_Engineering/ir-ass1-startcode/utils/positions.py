"""Token-position mapping (Task A-2)
index starting positions of unigrams or n-grams to support phrase and
proximity queries. Values are sorted, allowing linear-time merge joins later.
"""
from collections import defaultdict
from typing import Dict, List, Union, Tuple

def make_positions(tokens: List[str], n: int = 1) -> Dict[Union[str, Tuple[str, ...]], List[int]]:
    """Map each unique n-gram to a list of its 0-based start positions.
    Unigrams are emitted as strings; n>1 are tuples for hashable keys.
    Overlapping n-grams are indexed (e.g., bigrams at i and i+1 both present).
    Complexity: O(L) time, O(U) space where U is number of unique n-grams.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    positions: Dict[Union[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    L = len(tokens)

    if n == 1:
        for i, tok in enumerate(tokens):
            positions[tok].append(i)
    else:
        for i in range(0, max(0, L - n + 1)):
            gram = tuple(tokens[i:i+n])
            positions[gram].append(i)

    return dict(positions)


