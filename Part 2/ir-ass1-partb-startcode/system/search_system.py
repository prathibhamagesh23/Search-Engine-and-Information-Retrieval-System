#!/usr/bin/env python3
"""
Command-line Information Retrieval System - Task 4

Usage (from repo root):
  python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_default.json

Inputs:
  queries.json    -> [{"qid": "...", "query": "..."}, ...]
  documents.jsonl -> one JSON per line, at least {"id": <int>, "text": <str>} or {"id": <int>, "tokens": [..]}
Output (run JSON):
  [
    {"qid": "<qid>", "results": [{"doc_id": <int>, "score": <float>}, ...]},  # top-10
    ...
  ]
"""

from __future__ import annotations
import json, sys, os, pathlib, gzip
from typing import Dict, List, Any, Tuple, Iterable

# --- make repo root importable ---
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Task 1 / 2 / 3
from index.builders import create_all_indexes
from query_processing.query_process import process_query
from ranking.rankers import rank_documents

TOPK = 10  

# ----------------- helpers -----------------

def _preproc(val: Any) -> List[str]:
    """Very light, deterministic tokenization (dev data are already clean)."""
    if isinstance(val, list):
        return [str(t) for t in val if t is not None]
    return str(val if val is not None else "").strip().lower().split()

def _open_text_flex(path: str):
    """Open text with Windows-safe fallbacks."""
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return open(path, "r", encoding=enc, newline="")
        except UnicodeDecodeError:
            continue
    return open(path, "r", encoding="utf-8", errors="replace", newline="")

def _iter_lines_flex(path: str) -> Iterable[str]:
    """Iterate lines from a (maybe gzipped) text file with encoding fallbacks."""
    if path.endswith(".gz"):
        for enc in ("utf-8", "utf-8-sig", "cp1252"):
            try:
                with gzip.open(path, "rt", encoding=enc, newline="") as f:
                    for line in f:
                        yield line
                return
            except UnicodeDecodeError:
                continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                yield line
    else:
        with _open_text_flex(path) as f:
            for line in f:
                yield line

def _load_docs(path: str) -> Dict[int, List[str]]:
    """documents.jsonl -> {doc_id: [tokens...]} (supports 'text' or 'tokens')."""
    print(f"[load] documents: {path}")
    id2tok: Dict[int, List[str]] = {}
    seen = set()
    for line_num, line in enumerate(_iter_lines_flex(path), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: invalid JSON at line {line_num}: {e}")
            continue

        if "id" not in obj:
            print(f"Warning: line {line_num} missing 'id'; skipped")
            continue
        did = int(obj["id"])
        if did in seen:
            print(f"Warning: duplicate doc_id {did} at line {line_num}; keeping first")
            continue
        seen.add(did)

        # Prefer explicit tokens; else tokenize text.
        if "tokens" in obj:
            toks = obj["tokens"]
        elif "text" in obj:
            toks = obj["text"]
        else:
            print(f"Warning: line {line_num} missing 'text'/'tokens'; skipped")
            continue

        # Coerce to List[str], guard against None / wrong types
        if toks is None:
            id2tok[did] = []
        elif isinstance(toks, list):
            id2tok[did] = [str(t) for t in toks if t is not None]
        else:
            id2tok[did] = _preproc(toks)

    if not id2tok:
        raise ValueError("No documents loaded.")
    print(f"[load] documents: {len(id2tok)} loaded")
    return id2tok

def _load_queries(path: str) -> List[Dict[str, str]]:
    print(f"[load] queries:   {path}")
    with _open_text_flex(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Queries file must be a JSON array.")
    out: List[Dict[str, str]] = []
    for i, q in enumerate(data):
        if not isinstance(q, dict) or "qid" not in q or "query" not in q:
            raise ValueError(f"Query {i} missing required fields (qid, query).")
        out.append({"qid": str(q["qid"]), "query": str(q["query"])})
    print(f"[load] queries:   {len(out)} loaded")
    return out

# ----------------- main -----------------

def main():
    if len(sys.argv) != 4:
        print("Usage: python system/search_system.py <queries_json> <documents_jsonl> <run_output_json>")
        sys.exit(1)

    queries_path, docs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # 1) Load data
    try:
        queries = _load_queries(queries_path)
    except Exception as e:
        print(f"Error loading queries: {e}"); sys.exit(1)

    try:
        id2tok = _load_docs(docs_path)
    except Exception as e:
        print(f"Error loading documents: {e}"); sys.exit(1)

    # 2) Build unified Task-1 index (fresh, isolated for this run)
    cache_dir = _REPO / ".cache"
    cache_dir.mkdir(exist_ok=True)
    index_path = cache_dir / "task4_index.pkl.gz"

    print(f"[index] building unified index at: {index_path}")
    try:
        doc_ids_list = list(id2tok.keys())
        tokenized_docs_list = [id2tok[d] for d in doc_ids_list]
        create_all_indexes(tokenized_docs_list, str(index_path), doc_ids=doc_ids_list)
    except Exception as e:
        print(f"Error building index: {e}"); sys.exit(1)
    print("[index] done.")

    # 3) For each query: Task-2 candidates -> Task-3 ranking (method='default')
    results: List[Dict[str, Any]] = []
    for q in queries:
        qid = q["qid"]
        qtext = q["query"]

        # Task-2 (defensive: malformed structured queries -> empty)
        try:
            candidates = process_query(qtext, str(index_path))  # Set[int]
        except ValueError as e:
            print(f"Warning: malformed query (qid={qid}): {e}")
            results.append({"qid": qid, "doc_ids": [], "results": []})
            continue

        if not candidates:
            results.append({"qid": qid, "doc_ids": [], "results": []})
            continue

        cand_ids  = sorted(candidates)                     # deterministic
        cand_docs = [id2tok.get(d, []) for d in cand_ids]  # aligned

        # Task-3 ranking
        ranked_ids, scores = rank_documents(
            query_toks=_preproc(qtext),
            candidate_docs=cand_docs,
            doc_ids=cand_ids,
            inverted_index_path=str(index_path),
            method="default"  # maps to your best fixed performer
        )

        # ---- Top-K packaging (per spec: 10) ----
        ids_cut    = [int(d) for d in ranked_ids[:TOPK]]
        scores_cut = [float(s) for s in scores[:TOPK]]
        pairs      = [{"doc_id": d, "score": s} for d, s in zip(ids_cut, scores_cut)]

        results.append({
            "qid": qid,
            "doc_ids": ids_cut,   # for sanity checker
            "results": pairs      # for MAP/MRR evaluator
        })

    # 4) Write run JSON
    out_dir = pathlib.Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ok] wrote {out_path}  (queries={len(queries)})")

if __name__ == "__main__":
    main()
