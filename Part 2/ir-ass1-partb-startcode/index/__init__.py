"""
Unified Index Package - Task 1
Single package containing unified term/n-gram, wildcard, and proximity indexes.
"""

from .builders import create_all_indexes
from .access import get_posting_list, find_wildcard_matches, get_term_positions
from .io import dump, load

__all__ = [
    "create_all_indexes",
    "get_posting_list", 
    "find_wildcard_matches",
    "get_term_positions",
    "dump",
    "load"
]
