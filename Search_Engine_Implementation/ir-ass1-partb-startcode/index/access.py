"""
Unified Index Package Access Functions - Task 1
Provides O(1) access to all three sub-indexes from a single package.
"""

from typing import List, Union, Tuple, Dict, Any
from .io import load

# Global cache to store loaded packages for O(1) access
_package_cache: Dict[str, Dict[str, Any]] = {}

def _load_package(index_path: str) -> Dict[str, Any]:
    """Load and cache the unified index package."""
    if index_path not in _package_cache:
        _package_cache[index_path] = load(index_path)
    return _package_cache[index_path]

def clear_cache(index_path: str | None = None) -> None:
    """Clear cached package(s). If index_path is None, clear all."""
    if index_path is None:
        _package_cache.clear()
    else:
        _package_cache.pop(index_path, None)

def get_posting_list(term: Union[str, Tuple[str, ...]], index_path: str) -> List[int]:
    """
    Returns the posting list for a unigram or n-gram from the unified sub-index.

    Args:
        term: A string (unigram) or tuple of strings (n-gram)
        index_path: Path to the unified index package

    Returns:
        List of document IDs (sorted, deduplicated) containing the term
    """
    package = _load_package(index_path)
    unified_index = package.get("unified", {})
    # O(1) average-case dict lookup
    postings = unified_index.get(term)
    return postings[:] if postings is not None else []

def find_wildcard_matches(ngram: str, index_path: str) -> List[str]:
    """
    Returns the terms for a character n-gram (with $ boundaries), up to length 3.

    Args:
        ngram: an n-gram (e.g., "$cl", "on$", "mat")
        index_path: Path to the unified index package

    Returns:
        List of matching terms (sorted lexicographically, deduplicated)
    """
    # exclude the boundary-only unigram "$"
    if ngram == "$":
        return []

    package = _load_package(index_path)
    wildcard_index = package.get("wildcard", {})
    terms = wildcard_index.get(ngram)
    return terms[:] if terms is not None else []

def get_term_positions(term: Union[str, Tuple[str, ...]], doc_id: int, index_path: str) -> List[int]:
    """
    Returns the position list for a unigram or n-gram in a specific document.

    Args:
        term: A string (unigram) or tuple of strings (n-gram)
        doc_id: Document ID
        index_path: Path to the unified index package

    Returns:
        List of positions (0-based, sorted, deduplicated) where the term appears in the document
    """
    package = _load_package(index_path)
    proximity_index = package.get("proximity", {})
    # Two O(1) lookups: first for term, then for the document
    doc_map = proximity_index.get(term)
    if not doc_map:
        return []
    positions = doc_map.get(doc_id)
    return positions[:] if positions is not None else []
