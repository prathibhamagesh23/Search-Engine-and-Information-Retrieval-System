# ranking/rankers.py
"""
Task 3: Multi-Algorithm Ranking Framework (FINAL, spec-compliant)

Public API:
  rank_documents(query_toks, candidate_docs, doc_ids, inverted_index_path, method="default")
  -> (ranked_doc_ids, ranking_scores)

Spec requirements satisfied:
- Default method == BEST performer on dev with FIXED hyperparameters (here: "hybrid").
- Pure ranking (no query expansion inside this function).
- Semantic budget ≤ 20 documents per query (lexical shortlist first, deterministic).
- Deterministic, including tie-breaking (score desc, then doc_id asc).
- Read-only over Task-1 index (index build is used only in the dev runner).

Dev runner (optional):
  data/dev/queries.json
  data/dev/documents.jsonl
  data/dev/relevance_judge.json  (or relevancejudge.json)

Run from repo root:
  python ranking/rankers.py
"""

# --- make repo root importable (needed before importing index.*) ---
import pathlib, sys
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- stdlib imports ----
import os, json, math, tempfile, shutil, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

# ---- Task 1 access only (no building in ranking path) ----
from index.builders import create_all_indexes  # used only by dev runner
from index.io import load as _load_pkg

SEED = 13  # for keyed hashing (deterministic)
DEFAULT_METHOD = "hybrid"  # SPEC: default must be best performer on dev with fixed hyperparams

_STATS_CACHE: dict[str, "IndexStats"] = {}

def _get_stats(index_path: str) -> "IndexStats":
    st = _STATS_CACHE.get(index_path)
    if st is None:
        st = IndexStats(index_path)
        _STATS_CACHE[index_path] = st
    return st

# ---------------------- Utilities ----------------------

def _preprocess(text: Any) -> List[str]:
    """Very light preprocessing. Dev inputs assumed clean; keep minimal & deterministic."""
    if isinstance(text, list):
        return [str(t) for t in text]
    return str(text).strip().lower().split()

def _tie_break_sort(ids: List[int], scores: List[float]) -> Tuple[List[int], List[float]]:
    """Sort by score desc, then doc_id asc."""
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [ids[i] for i in order], [scores[i] for i in order]

def _pearson(y_true: List[float], y_pred: List[float]) -> float:
    """Pearson r with safe zero-variance handling."""
    n = len(y_true)
    if n < 2:
        return 0.0
    mt = sum(y_true)/n
    mp = sum(y_pred)/n
    num = sum((t-mt)*(p-mp) for t,p in zip(y_true,y_pred))
    dt = math.sqrt(sum((t-mt)**2 for t in y_true))
    dp = math.sqrt(sum((p-mp)**2 for p in y_pred))
    if dt == 0.0 or dp == 0.0:
        return 0.0
    return num/(dt*dp)

def _resolve_dev_path(*relative_candidates: str) -> str:
    """
    Return first existing path among candidates, resolved to repo root,
    else fall back to /mnt/data for convenience.
    """
    tried = []
    for rel in relative_candidates:
        p = (_PROJECT_ROOT / rel).resolve()
        tried.append(str(p))
        if p.exists():
            return str(p)
    for rel in relative_candidates:
        p = Path("/mnt/data") / Path(rel).name
        tried.append(str(p))
        if p.exists():
            return str(p)
    raise FileNotFoundError("Dev file not found. Tried:\n  " + "\n  ".join(tried))

# ----------------- Index stats from package -----------------

class IndexStats:
    """Reads global stats from the Task-1 package and caches them."""
    def __init__(self, index_path: str):
        self.pkg = _load_pkg(index_path)
        meta = self.pkg.get("__META__", {})
        self.N: int = int(meta.get("N", 0))
        self.avgdl: float = float(meta.get("avgdl", 0.0))
        self.doc_lengths: Dict[int,int] = {int(k): int(v) for k,v in meta.get("doc_lengths", {}).items()}
        self.unified: Dict[Any, List[int]] = self.pkg.get("unified", {})  # term->posting list

    def df(self, term: str) -> int:
        postings = self.unified.get(term)
        return len(postings) if postings is not None else 0

# ----------------- Scorers: BM25 / TF-IDF -----------------

def _bm25_idf_classic(N: int, df: int) -> float:
    # Classic Okapi BM25 IDF (may be negative for very common terms)
    return math.log((N - df + 0.5) / (df + 0.5)) if N > 0 else 0.0

def _bm25_idf_floored(N: int, df: int) -> float:
    # Non-negative IDF: classic but floored at 0 (empirically stronger on many dev sets)
    return max(0.0, _bm25_idf_classic(N, df))

def _bm25_scores(query_toks: List[str],
                 docs_tokens: List[List[str]],
                 doc_ids: List[int],
                 stats: IndexStats,
                 k1: float = 1.2,
                 b: float = 0.75,
                 idf_floor: bool = False,
                 precomputed_tf: Optional[List[Dict[str, int]]] = None,
                 precomputed_dl: Optional[List[int]] = None) -> List[float]:
    """
    Okapi BM25. Query terms deduped (common for short queries).
    idf_floor=False -> classic IDF   (spec-conformant default for "bm25")
    idf_floor=True  -> non-negative  (used by "bm25plus"/hybrid/semantic to lift correlation)

    If precomputed_tf/precomputed_dl are provided, uses them for speed.
    precomputed_tf[i] is a dict with counts ONLY for query terms in docs_tokens[i].
    precomputed_dl[i] is the document length for doc_ids[i].
    """
    if not docs_tokens:
        return []
    q_terms = list(dict.fromkeys(query_toks))
    N = stats.N
    idf = {t: (_bm25_idf_floored(N, stats.df(t)) if idf_floor else _bm25_idf_classic(N, stats.df(t)))
           for t in q_terms}
    scores: List[float] = []
    avgdl = stats.avgdl or 1.0

    use_tf = precomputed_tf is not None and len(precomputed_tf) == len(docs_tokens)
    use_dl = precomputed_dl is not None and len(precomputed_dl) == len(docs_tokens)

    for i, (did, toks) in enumerate(zip(doc_ids, docs_tokens)):
        dl = precomputed_dl[i] if use_dl else stats.doc_lengths.get(did, len(toks))
        denom_norm = k1*(1 - b + b * (dl / avgdl))
        tf = precomputed_tf[i] if use_tf else Counter(toks)
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                s += idf[t] * (f*(k1+1)) / (f + denom_norm)
        scores.append(float(s))
    return scores

def _tfidf_cosine_scores(query_toks: List[str],
                         docs_tokens: List[List[str]],
                         doc_ids: List[int],
                         stats: IndexStats,
                         precomputed_tf: Optional[List[Dict[str, int]]] = None) -> List[float]:
    """
    Cosine similarity between tf-idf vectors (idf from index).
    idf(t) = log((N + 1)/(df + 1)) + 1  (smooth, positive)

    If precomputed_tf is provided, uses those counts for speed.
    """
    if not docs_tokens:
        return []
    vocab = list(dict.fromkeys(query_toks))
    N = stats.N
    idf: Dict[str, float] = {t: (math.log((N + 1)/(stats.df(t) + 1)) + 1.0) if N > 0 else 1.0 for t in vocab}

    q_tf = Counter(query_toks)
    q_vec: Dict[str, float] = {t: q_tf[t]*idf[t] for t in vocab if t in q_tf}
    q_norm = math.sqrt(sum(v*v for v in q_vec.values())) or 1.0

    use_tf = precomputed_tf is not None and len(precomputed_tf) == len(docs_tokens)

    scores: List[float] = []
    for i, toks in enumerate(docs_tokens):
        tf_map = precomputed_tf[i] if use_tf else Counter(toks)
        dot = 0.0
        d_norm_sq = 0.0
        for t in vocab:
            f = tf_map.get(t, 0)
            if not f:
                continue
            w_d = f * idf[t]
            d_norm_sq += w_d*w_d
            if t in q_vec:
                dot += q_vec[t] * w_d
        d_norm = math.sqrt(d_norm_sq) or 1.0
        scores.append(float(dot/(q_norm*d_norm)))
    return scores

# -------- Deterministic "semantic" rerank (BM25+ shortlist) --------

def _stable_hash(s: str, seed: int = SEED) -> int:
    """Keyed, deterministic 64-bit hash (portable across runs/platforms)."""
    h = hashlib.blake2b(digest_size=8, key=str(seed).encode("utf-8"))
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), "big")

from functools import lru_cache

@lru_cache(maxsize=50000)
def _hash_pair(tok: str, seed: int = SEED) -> tuple[int, int]:
    # deterministic index & sign for a token
    import hashlib
    def h(s: str) -> int:
        hh = hashlib.blake2b(digest_size=8, key=str(seed).encode("utf-8"))
        hh.update(s.encode("utf-8"))
        return int.from_bytes(hh.digest(), "big")
    idx  = h(tok + "|proj") % 256  # keep dim consistent with your current 256
    sign = 1 if (h(tok + "|sign") & 1) == 0 else -1
    return idx, sign

def _hashing_vector(tokens: List[str], dim: int = 256, seed: int = SEED) -> List[float]:
    vec = [0.0] * dim
    for tok in tokens:
        idx, sign = _hash_pair(tok, seed)
        vec[idx] += sign
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    return float(sum(x*y for x,y in zip(a,b)))

def _semantic_rerank(query_toks: List[str],
                     docs_tokens: List[List[str]],
                     doc_ids: List[int],
                     base_scores: List[float],
                     shortlist_k: int = 20,   # SPEC: ≤ 20 docs
                     alpha: float = 0.60) -> List[float]:
    """
    Two-stage: shortlist top-K by BM25+ scores, compute hash-cosine for those,
    and blend: final = alpha*base + (1-alpha)*cosine. Others keep base.
    Deterministic due to _stable_hash.
    """
    if not docs_tokens:
        return []
    k = min(shortlist_k, len(doc_ids))
    order = sorted(range(len(doc_ids)), key=lambda i: (-base_scores[i], doc_ids[i]))[:k]
    qv = _hashing_vector(query_toks)
    cosines = {i: _cosine(qv, _hashing_vector(docs_tokens[i])) for i in order}
    out = []
    for i in range(len(doc_ids)):
        out.append(alpha*base_scores[i] + (1.0 - alpha)*cosines[i] if i in cosines else base_scores[i])
    return out

# ----------------------- Public API -----------------------

def rank_documents(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    inverted_index_path: str,
    method: str = "default"
) -> Tuple[List[int], List[float]]:
    """
    Rank ALL candidate documents and return aligned (ranked_ids, scores).

    Args:
        query_toks: tokenized, cleaned query (no expansion here).
        candidate_docs: tokenized, cleaned docs aligned 1-to-1 with doc_ids.
        doc_ids: document IDs aligned with candidate_docs.
        inverted_index_path: path to your single Task-1 index package.
        method: one of:
            - "default": maps to BEST performer (fixed hyperparams) per spec (here: "hybrid")
            - "bm25": classic Okapi BM25 (classic IDF; may be negative)
            - "bm25plus": BM25 with non-negative IDF floor, tuned k1/b
            - "tfidf": cosine tf-idf
            - "hybrid": 0.8*minmax(BM25+) + 0.2*minmax(TFIDF) with flat fallback
            - "semantic": BM25+ shortlist (20) + deterministic hash-cosine blend (alpha=0.60)

    Returns:
        (ranked_doc_ids, ranking_scores) sorted by score desc with doc_id asc tie-breaks.
    """
    assert len(candidate_docs) == len(doc_ids), "candidate_docs and doc_ids must align"
    if not candidate_docs:
        return [], []

    stats = _get_stats(inverted_index_path)


    m = (method or "default").lower()
    # during evaluation, grader passes method="default".
    # Map default to BEST performer (fixed hyperparameters)
    if m == "default":
        m = DEFAULT_METHOD

    # ------- shared fast-path precomputation (identical outputs) -------
    # Count ONLY query terms per doc in a single pass; reuse for BM25 & TF-IDF.
    q_terms = list(dict.fromkeys(query_toks))
    qset = set(q_terms)
    pre_tf: List[Dict[str, int]] = []
    pre_dl: List[int] = []
    for did, toks in zip(doc_ids, candidate_docs):
        c: Dict[str, int] = {}
        for tok in toks:
            if tok in qset:
                c[tok] = c.get(tok, 0) + 1
        pre_tf.append(c)
        pre_dl.append(stats.doc_lengths.get(did, len(toks)))

    if m == "bm25":
        base = _bm25_scores(query_toks, candidate_docs, doc_ids, stats,
                            k1=1.2, b=0.75, idf_floor=False,
                            precomputed_tf=pre_tf, precomputed_dl=pre_dl)  # classic IDF

    elif m == "bm25plus":
        base = _bm25_scores(query_toks, candidate_docs, doc_ids, stats,
                            k1=1.4, b=0.7, idf_floor=True,
                            precomputed_tf=pre_tf, precomputed_dl=pre_dl)

    elif m == "tfidf":
        base = _tfidf_cosine_scores(query_toks, candidate_docs, doc_ids, stats,
                                    precomputed_tf=pre_tf)

    elif m == "hybrid":
        bm = _bm25_scores(query_toks, candidate_docs, doc_ids, stats,
                          k1=1.4, b=0.7, idf_floor=True,
                          precomputed_tf=pre_tf, precomputed_dl=pre_dl)  # stronger BM25+
        tf = _tfidf_cosine_scores(query_toks, candidate_docs, doc_ids, stats,
                                  precomputed_tf=pre_tf)
        # min-max normalize each channel then blend; fallback if both flat
        def _minmax(x: List[float]) -> List[float]:
            if not x: return []
            mn, mx = min(x), max(x)
            if mx == mn:
                return [0.0]*len(x)
            return [(v - mn)/(mx - mn) for v in x]
        bm_n, tf_n = _minmax(bm), _minmax(tf)
        if all(v == 0 for v in bm_n) and all(v == 0 for v in tf_n):
            base = bm[:]  # fallback to BM25+ raw scores
        else:
            w_bm, w_tf = 0.8, 0.2
            base = [w_bm*b + w_tf*t for b,t in zip(bm_n, tf_n)]

    elif m == "semantic":
        bm = _bm25_scores(query_toks, candidate_docs, doc_ids, stats,
                          k1=1.4, b=0.7, idf_floor=True,
                          precomputed_tf=pre_tf, precomputed_dl=pre_dl)  # shortlist by BM25+
        base = _semantic_rerank(query_toks, candidate_docs, doc_ids, bm,
                                shortlist_k=20, alpha=0.60)  # SPEC budget
    else:
        raise ValueError(f"Unknown method '{method}'")

    ranked_ids, ranking_scores = _tie_break_sort(doc_ids, base)
    return ranked_ids, ranking_scores

# ----------------------- Dev loader & runner (optional) -----------------------

def _load_dev_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for q in data:
        qid = q.get("qid", q.get("id", q.get("query_id", q.get("qid_str", None))))
        text = q.get("query", q.get("text", ""))
        out.append({"qid": str(qid), "text": text})
    return out

def _load_dev_docs(path: str) -> Dict[int, List[str]]:
    id2tok: Dict[int, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = obj.get("doc_id", obj.get("id"))
            text = obj.get("text", obj.get("tokens", ""))
            id2tok[int(did)] = _preprocess(text)
    return id2tok

def _load_dev_gold(path: str) -> Dict[str, Dict[int, float]]:
    """
    Load dev relevance judgments. Supports multiple common JSON layouts.
    Returns: {qid: {doc_id: rel, ...}, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    gold: Dict[str, Dict[int, float]] = {}

    # dict-of-dicts {qid: {doc_id: rel}}
    if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
        for qid, m in raw.items():
            dd: Dict[int, float] = {}
            for k, v in m.items():
                try:
                    dd[int(k)] = float(v)
                except Exception:
                    continue
            gold[str(qid)] = dd
        return gold

    # unwrap dict-with-list
    items = None
    if isinstance(raw, dict):
        for key in ("qrels", "relevance", "judgments", "labels", "data"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]; break
        if items is None:
            for v in raw.values():
                if isinstance(v, list):
                    items = v; break
            if items is None:
                raise ValueError("Unrecognized relevance JSON structure (dict without list or nested dicts).")
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Unrecognized relevance JSON structure (not dict or list).")

    # parse list-forms
    for e in items:
        if not isinstance(e, dict):
            continue
        qid = e.get("qid") or e.get("query_id") or e.get("id") or e.get("qid_str")
        if qid is None:
            continue
        qid = str(qid)

        if isinstance(e.get("relevance_scores"), dict):
            for k, v in e["relevance_scores"].items():
                try:
                    did = int(k); rel = float(v)
                except Exception:
                    continue
                gold.setdefault(qid, {})[did] = rel
            continue

        if any(k in e for k in ("doc_id", "document_id", "doc")):
            did = e.get("doc_id", e.get("document_id", e.get("doc")))
            rel = e.get("rel", e.get("score", e.get("label", e.get("relevance", e.get("grade", 0)))))
            try:
                did = int(did); rel = float(rel)
            except Exception:
                continue
            gold.setdefault(qid, {})[did] = rel
            continue

        if isinstance(e.get("relevance"), dict):
            for k, v in e["relevance"].items():
                try:
                    did = int(k); rel = float(v)
                except Exception:
                    continue
                gold.setdefault(qid, {})[did] = rel
            continue

        if isinstance(e.get("docs"), list):
            for d in e["docs"]:
                if not isinstance(d, dict):
                    continue
                did = d.get("doc_id", d.get("id", d.get("document_id")))
                rel = d.get("rel", d.get("score", d.get("label", d.get("relevance", d.get("grade", 0)))))
                try:
                    did = int(did); rel = float(rel)
                except Exception:
                    continue
                gold.setdefault(qid, {})[did] = rel
            continue

    return gold

def _build_small_index(id2tok: Dict[int, List[str]], out_path: str) -> None:
    doc_ids = list(id2tok.keys())
    tokenized_docs = [id2tok[d] for d in doc_ids]
    create_all_indexes(tokenized_docs, out_path, doc_ids=doc_ids)

def _print_table(rows: List[Tuple[str, float]]) -> None:
    w = max(6, max(len(n) for n,_ in rows)) if rows else 6
    print("\nMethod".ljust(w), " | Pearson r")
    print("-"*w + "-+-----------")
    for name, r in rows:
        print(name.ljust(w), f"| {r:.4f}")

def _evaluate_methods(index_path: str,
                      methods: List[str],
                      queries: List[Dict[str,Any]],
                      id2tok: Dict[int, List[str]],
                      gold: Dict[str, Dict[int, float]]) -> List[Tuple[str, float]]:
    rows = []
    for method in methods:
        all_true: List[float] = []
        all_pred: List[float] = []
        for q in queries:
            qid = q["qid"]
            q_tokens = _preprocess(q["text"])
            # Candidate set: use gold keys for this query (common in dev). Fallback to all docs.
            candidates = sorted(gold.get(qid, {}).keys()) or sorted(id2tok.keys())
            cand_docs = [id2tok[d] for d in candidates]
            ranked_ids, scores = rank_documents(q_tokens, cand_docs, candidates, index_path, method=method)
            y_true = [gold.get(qid, {}).get(d, 0.0) for d in ranked_ids]
            all_true.extend(y_true)
            all_pred.extend(scores)
        r = _pearson(all_true, all_pred)
        rows.append((method, r))
    return rows

def main():
    # Resolve dev paths; accept both relevance judge filenames
    qpath = _resolve_dev_path("data/dev/queries.json")
    dpath = _resolve_dev_path("data/dev/documents.jsonl")
    gpath = _resolve_dev_path("data/dev/relevance_judge.json", "data/dev/relevancejudge.json")

    # Load dev data
    queries = _load_dev_queries(qpath)
    id2tok = _load_dev_docs(dpath)
    gold = _load_dev_gold(gpath)

    # Build an isolated index for these dev docs
    tmpdir = tempfile.mkdtemp(prefix="rank_dev_")
    try:
        index_path = os.path.join(tmpdir, "dev_index.pkl.gz")
        _build_small_index(id2tok, index_path)

        # Evaluate: default is BEST performer
        methods = os.getenv(
            "TASK3_METHODS",
            "bm25,bm25plus,tfidf,hybrid,semantic,default"
        ).split(",")
        rows = _evaluate_methods(index_path, methods, queries, id2tok, gold)

        best_name, best_r = max(rows, key=lambda x: x[1]) if rows else ("bm25", 0.0)
        print(f"\nBest dev performer: {best_name} (r={best_r:.4f})")
        #print("Note: method='default' == 'hybrid' (best performer). Use other methods explicitly to compare.\n")

        _print_table(rows)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
