from typing import Set, List, Tuple, Union
import re
from index.access import get_term_positions, get_posting_list

_NEAR_STRICT_RE = re.compile(r"NEAR/([0-9]+)")  # strict (no spaces)

Operand = Union[str, Tuple[str, ...]]

def _parse_proximity(query: str) -> Tuple[Operand, int, Operand]:
    m = _NEAR_STRICT_RE.search(query)
    if not m:
        raise ValueError("Malformed: NEAR/k not found.")
    k = int(m.group(1))

    parts = _NEAR_STRICT_RE.split(query)  # [left, k, right]
    if len(parts) != 3:
        raise ValueError("Malformed NEAR/k structure.")
    left_raw, _, right_raw = parts
    left_raw, right_raw = left_raw.strip(), right_raw.strip()
    if not left_raw or not right_raw:
        raise ValueError("Malformed: NEAR/k missing operands.")

    def parse_operand(s: str) -> Operand:
        s = s.strip()
        # Parentheses are not permitted as NEAR operands in this spec
        if s.startswith("(") or s.endswith(")"):
            raise ValueError("Malformed: parentheses not allowed as NEAR operands.")
        # Wildcards are not permitted inside NEAR operands
        if "*" in s:
            raise ValueError("Malformed: wildcard not allowed in NEAR operands.")
        # Quoted phrase (≤3 words)
        if s.startswith('"') and s.endswith('"'):
            inner = s[1:-1].strip()
            words = inner.split()
            if not words or len(words) > 3:
                raise ValueError("Malformed phrase operand.")
            return tuple(words)
        # Must be a single term otherwise
        if '"' in s or " " in s:
            raise ValueError("Malformed: NEAR operand must be a term or quoted phrase.")
        return s

    return parse_operand(left_raw), k, parse_operand(right_raw)

def _spans(operand: Operand, doc_id: int, index_path: str) -> List[Tuple[int, int]]:
    """term -> [(p,p)]; phrase of length m -> [(s, s+m-1)]"""
    if isinstance(operand, tuple):
        starts = get_term_positions(operand, doc_id, index_path)
        m = len(operand)
        return [(s, s + m - 1) for s in starts]
    else:
        starts = get_term_positions(operand, doc_id, index_path)
        return [(s, s) for s in starts]

def process_proximity_query(query: str, index_path: str) -> Set[int]:
    """
    Process proximity queries with NEAR/k semantics (edge-to-edge, order-insensitive).
    D = min(|qstart - pend|, |pstart - qend|) and match iff D <= k.
    The same exact token span cannot satisfy both operands (so t NEAR/0 t is false).
    """
    # must contain exactly one NEAR/k
    if query.count("NEAR/") != 1:
        raise ValueError("Malformed: must contain exactly one NEAR/k.")

    left, k, right = _parse_proximity(query)

    # quick candidate docs = intersection of postings of both operands
    def postings(op: Operand) -> Set[int]:
        return set(get_posting_list(op, index_path))
    candidates = postings(left) & postings(right)
    if not candidates:
        return set()

    # special-case identical single-token with k == 0 can never match
    if k == 0 and isinstance(left, str) and isinstance(right, str) and left == right:
        return set()

    result: Set[int] = set()
    for did in candidates:
        L = _spans(left, did, index_path)
        R = _spans(right, did, index_path)
        if not L or not R:
            continue

        i = j = 0
        matched = False
        while i < len(L) and j < len(R):
            ls, le = L[i]
            rs, re = R[j]

            # Overlap handling
            if not (le < rs or re < ls):
                # identical spans cannot satisfy both operands
                if ls == rs and le == re:
                    # advance one pointer deterministically to avoid infinite loop
                    j += 1
                    continue
                # overlapping but not identical -> distance 0
                matched = (0 <= k)
                if matched:
                    break
            else:
                # edge to edge distance when non overlapping
                D = min(abs(rs - le), abs(ls - re))
                if D <= k:
                    matched = True
                    break

            # advance the pointer with the smaller end to search closer candidates
            if le < re:
                i += 1
            else:
                j += 1

        if matched:
            result.add(did)

    return result
