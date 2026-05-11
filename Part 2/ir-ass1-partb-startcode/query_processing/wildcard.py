from typing import Set, Iterable, List
import re
from index.access import find_wildcard_matches, get_posting_list

def _glob_to_regex(pattern: str) -> re.Pattern:
    esc = re.escape(pattern)
    return re.compile("^" + esc.replace(r"\*", ".*") + "$")

def _grams_1_3(text: str) -> Iterable[str]:
    L = len(text)
    for n in (1, 2, 3):
        if L >= n:
            for i in range(0, L - n + 1):
                yield text[i:i+n]

def _required_grams(pattern: str) -> List[str]:
    """
    Build necessary char n-grams (1..3) with $ boundaries from fixed parts of the pattern.
      prefix*  -> grams from "$prefix"
      *suffix  -> grams from "suffix$"
      pre*suf  -> grams from "$pre" + "suf$"
    Excludes the meaningless unigram "$".
    """
    if pattern.count("*") != 1:
        raise ValueError("Malformed wildcard: only one '*' supported.")
    left, right = pattern.split("*", 1)
    grams: List[str] = []

    if left:
        grams.extend(_grams_1_3("$" + left))
    if right:
        grams.extend(_grams_1_3(right + "$"))

    grams = [g for g in grams if not (len(g) == 1 and g == "$")]
    # dedup, preserve order
    seen, out = set(), []
    for g in grams:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out

def process_wildcard_query(pattern: str, index_path: str) -> Set[int]:
    """
    Expand wildcard pattern to candidate terms using the wildcard sub-index, then
    union their postings from the unified sub-index.
    """
    # validation : single token, contains '*', at least one non-* char, no mixing
    if any(ch.isspace() for ch in pattern) or '"' in pattern or "(" in pattern or ")" in pattern:
        raise ValueError("Malformed wildcard: must be a single token without spaces/quotes/parentheses.")
    if "*" not in pattern or all(ch == "*" for ch in pattern):
        raise ValueError("Malformed wildcard: must contain '*' and at least one non-* char.")
    if pattern.count("*") != 1:
        raise ValueError("Malformed wildcard: only one '*' supported.")
    # Do NOT reject operator substrings like 'sAND*' or 'CNOT*' — they are valid tokens.

    grams = _required_grams(pattern)
    if not grams:
        return set()

    # intersect candidate terms across grams
    candidates = set(find_wildcard_matches(grams[0], index_path))
    for g in grams[1:]:
        candidates &= set(find_wildcard_matches(g, index_path))
        if not candidates:
            break

    # final exact filter with regex
    rx = _glob_to_regex(pattern)
    terms = [t for t in candidates if rx.match(t)]

    # union postings for all matching terms
    docs: Set[int] = set()
    for t in terms:
        docs |= set(get_posting_list(t, index_path))
    return docs
