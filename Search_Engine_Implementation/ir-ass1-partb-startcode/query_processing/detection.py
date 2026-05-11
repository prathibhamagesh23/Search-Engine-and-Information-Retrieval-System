import re
from typing import Set

# Standalone operators (avoid substring false positives like sANDman)
_OP_RE = re.compile(r'(?<!\w)(AND|OR|NOT)(?!\w)')

# Proximity regexes
_NEAR_TOKEN_RE  = re.compile(r'\bNEAR\b')          # any appearance of token NEAR
_NEAR_STRICT_RE = re.compile(r'\bNEAR/([0-9]+)\b') # strict: NEAR/<int> (no spaces)
_QUOTE_RE       = re.compile(r'"([^"]*)"')

OPS: Set[str] = {"AND", "OR", "NOT"}

def detect_query_type(query: str) -> str:
    """
    Return one of {"proximity", "wildcard", "boolean", "natural_language"}.

    Rules (case-sensitive):
      - Proximity: requires exactly one strict NEAR/<int> (no spaces), no AND/OR/NOT, no '*'
      - Wildcard: single token containing exactly one '*', no whitespace/quotes/parens
      - Boolean: presence of standalone AND/OR/NOT or matched quotes (phrases)
      - Else: natural language

    Malformed structured inputs -> raise ValueError.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    # ---------- Proximity (strict NEAR/<int>) ----------
    # If NEAR token appears at all, require the strict NEAR/<int> form; otherwise it's malformed.
    if _NEAR_TOKEN_RE.search(query):
        matches = _NEAR_STRICT_RE.findall(query)
        if not matches:
            raise ValueError("Malformed: invalid NEAR/k form (must be NEAR/<integer> with no spaces).")
        if len(matches) != 1:
            raise ValueError("Malformed: more than one NEAR/k.")
        # do not mix with boolean operators or wildcard
        if _OP_RE.search(query):
            raise ValueError("Malformed: proximity combined with Boolean.")
        if "*" in query:
            raise ValueError("Malformed: proximity combined with wildcard.")
        # require non-empty operands around NEAR/<int>
        # split on the strict match; result is [left, k, right]
        parts = _NEAR_STRICT_RE.split(query)
        if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
            raise ValueError("Malformed: NEAR/k missing operands.")
        return "proximity"

    # ---------- Wildcard ----------
    if "*" in query:
        # must be a single token: no spaces, quotes, or parentheses
        if any(ch.isspace() for ch in query) or '"' in query or "(" in query or ")" in query:
            raise ValueError("Malformed wildcard: must be a single token without spaces/quotes/parentheses.")
        if query.count("*") != 1:
            raise ValueError("Malformed wildcard: multiple '*' not allowed.")
        if all(ch == "*" for ch in query):
            raise ValueError("Malformed wildcard: pattern cannot be only '*'.")
        return "wildcard"

    # ---------- Boolean ----------
    has_ops = bool(_OP_RE.search(query))
    has_quotes = ('"' in query)

    if has_ops or has_quotes:
        # Quotes must be matched; phrases non-empty and ≤ 3 words
        if query.count('"') % 2 != 0:
            raise ValueError("Malformed: unmatched quotes.")
        if '""' in query:
            raise ValueError("Malformed: empty phrase.")
        for m in _QUOTE_RE.finditer(query):
            content = m.group(1)
            if content.strip() == "":
                raise ValueError("Malformed: empty phrase.")
            if len(content.split()) > 3:
                raise ValueError("Malformed: phrase longer than 3 words.")
        return "boolean"

    # ---------- Natural language ----------
    return "natural_language"
