"""
Unified Index Package Builder - Task 1
Creates a single on-disk package containing all three sub-indexes.
"""
from typing import List, Dict, Union, Tuple, Any, Optional
from collections import defaultdict

from utils.ngram import make_ngrams_tokens, make_ngrams_chars
from utils.positions import make_positions
from .io import dump


Key = Union[str, Tuple[str, ...]]
PositionsMap = Dict[Key, Dict[int, List[int]]] 

def _dedup_sorted(lst):
    """Return a deduplicated, ascending-sorted list preserving numeric/lex order."""
    if not lst:
        return []
    # Fast path: sort first, then unique
    lst = sorted(lst)
    out = [lst[0]]
    for x in lst[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def create_all_indexes(
    tokenized_docs: List[List[str]],
    index_path: str,
    doc_ids: Optional[List[int]] = None
) -> None:
    """
    Build a unified index package containing all three sub-indexes in a single pass.

    Args:
        tokenized_docs: List of tokenized documents, each document is a list of tokens
        index_path: Path where the unified index package will be saved
        doc_ids: Optional list of document IDs. If None, uses sequential IDs (0..N-1).
                 If provided, must be the same length as tokenized_docs.
    """
    if doc_ids is None:
        doc_ids = list(range(len(tokenized_docs)))
    if len(doc_ids) != len(tokenized_docs):
        raise ValueError("doc_ids and tokenized_docs must have the same length")

    # in-memory accumulators
    unified_postings: Dict[Key, List[int]] = defaultdict(list)         # term/bigram/trigram -> [doc_ids...]
    wildcard_index: Dict[str, List[str]] = defaultdict(list)           # char n-gram -> [terms...]
    proximity_positions: PositionsMap = defaultdict(lambda: defaultdict(list))


    # meta
    doc_lengths: Dict[int, int] = {}
    N = len(tokenized_docs)

    # Collect unique terms once for per-doc to avoid duplicate wildcard insertions
    for doc, did in zip(tokenized_docs, doc_ids):
        # metadata
        doc_lengths[did] = len(doc)


        # UNIFIED (postings for 1-3 grams)
        # add did once per key per doc (deduped afterwards, but avoid extra work by de-duping per doc)
        seen_keys_in_doc: set[Key] = set()

        # unigrams, bigrams, trigrams (no boundary symbols here)
        for n in (1, 2, 3):
            if n == 1:
                grams_iter = ((tok,) for tok in doc)  # temporary tuples; i normalize below
            else:
                grams_iter = (tuple(doc[i:i+n]) for i in range(0, max(0, len(doc) - n + 1)))

            for g in grams_iter:
                key: Key = g[0] if len(g) == 1 else g
                if key not in seen_keys_in_doc:
                    unified_postings[key].append(did)
                    seen_keys_in_doc.add(key)

        # PROXIMITY (positions)
        # Required: unigram positions
        uni_pos = make_positions(doc, n=1)  # {term: [pos...]}
        for term, pos_list in uni_pos.items():
            proximity_positions[term][did].extend(pos_list)

        # Optional accelerators: bigram and trigram positions
        for n in (2, 3):
            if len(doc) >= n:
                npos = make_positions(doc, n=n)  # {(t1,...,tn): [pos...]}
                for gram, pos_list in npos.items():
                    proximity_positions[gram][did].extend(pos_list)

        # WILDCARD (char n-grams with $ boundaries, up to 3)
        # Only unique tokens per doc to avoid repeated pushes into the same char-gram bucket
        unique_terms = set(doc)
        for term in unique_terms:
            # For each n in 1..3, add gram->term (exclude unigram '$')
            for n in (1, 2, 3):
                for gram in make_ngrams_chars(term, n):
                    # Exclude the boundary-only unigram "$"
                    if n == 1 and gram == "$":
                        continue
                    wildcard_index[gram].append(term)

    # finalize: sort & dedup

    # unified postings: each list must be deduped and sorted ascending numerically
    for key, plist in unified_postings.items():
        unified_postings[key] = _dedup_sorted(plist)

    # wildcard: each term list must be deduped and lexicographically sorted
    for gram, terms in wildcard_index.items():
        wildcard_index[gram] = _dedup_sorted(terms)

    # proximity: for each key -> doc_id -> positions, each positions list sorted & deduped
    for key, docmap in proximity_positions.items():
        for did, pos_list in docmap.items():
            # positions are integers: sort ascending & dedup
            docmap[did] = _dedup_sorted(pos_list)
        # ensure deterministic doc_id order when iterating (dicts are ordered by insertion;
        # reinsert by sorted order so serialization is stable across runs)
        ordered = dict(sorted(docmap.items(), key=lambda kv: kv[0]))
        proximity_positions[key] = ordered

    def _sort_key(k):
        if isinstance(k, tuple):
            return ("tuple", " ".join(k))   # prefix ensures tuples group separately
        else:
            return ("str", k)

    unified_ordered = dict(sorted(unified_postings.items(), key=lambda kv: _sort_key(kv[0])))
    wildcard_ordered = dict(sorted(wildcard_index.items(), key=lambda kv: kv[0]))
    proximity_ordered = dict(sorted(proximity_positions.items(), key=lambda kv: _sort_key(kv[0])))

    avgdl = (sum(doc_lengths.values()) / N) if N > 0 else 0.0

    package: Dict[str, Any] = {
        "__META__": {
            "N": N,
            "doc_lengths": doc_lengths,
            "avgdl": avgdl,
            "version": "1.0",
            "ngrams_max": 3,
            "char_ngrams_max": 3,
        },
        "unified": unified_ordered,
        "wildcard": wildcard_ordered,
        "proximity": proximity_ordered,
    }

    # Save the unified package to disk
    dump(package, index_path)
