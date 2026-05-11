from typing import Set
from .detection import detect_query_type
from .boolean import process_boolean_query
from .wildcard import process_wildcard_query
from .proximity import process_proximity_query

def convert_natural_language(nl_query: str) -> str:
    """
    Convert already-cleaned natural language to a Boolean-OR query.

    Input is assumed pre-cleaned/lowercased (do not preprocess here).
    Split on whitespace and join tokens with ' OR '.
    If empty/whitespace, return ''.
    """
    if not nl_query or not nl_query.strip():
        return ""
    tokens = nl_query.split()
    return " OR ".join(tokens)

def process_query(query: str, index_path: str) -> Set[int]:
    """
    Main query processing with automatic type detection.

    Dispatch rules:
      - "boolean"   -> process_boolean_query
      - "wildcard"  -> process_wildcard_query
      - "proximity" -> process_proximity_query
      - else        -> convert to OR-string, then boolean
    """
    qtype = detect_query_type(query)

    if qtype == "boolean":
        return process_boolean_query(query, index_path)
    elif qtype == "wildcard":
        return process_wildcard_query(query, index_path)
    elif qtype == "proximity":
        return process_proximity_query(query, index_path)
    else:  # "natural_language"
        boolean_query = convert_natural_language(query)
        return process_boolean_query(boolean_query, index_path) if boolean_query else set()
