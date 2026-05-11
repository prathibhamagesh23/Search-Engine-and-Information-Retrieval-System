#!/usr/bin/env python3
"""
MAP Evaluation Script for Task 4
Evaluates run JSON files in ./runs against development relevance judgments.

Default usage (from repo root, per spec):
  python metrics/eval_map.py

Optional:
  python metrics/eval_map.py --runs runs/run_default.json runs/other.json --k 10
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import glob
import pathlib
from typing import Dict, List, Any, Tuple, Set

# ---- repo root (for consistent relative paths) ----
_REPO = pathlib.Path(__file__).resolve().parents[1]
os.chdir(_REPO)

# ---- hard-coded dev judge paths----
_QRELS_CANDIDATES = [
    "data/dev/relevance_judge.json",
    "data/dev/relevancejudge.json",
]

def _resolve_qrels_path() -> str:
    tried = []
    for p in _QRELS_CANDIDATES:
        if pathlib.Path(p).exists():
            return p
        tried.append(p)
    raise FileNotFoundError("Dev qrels not found. Tried:\n  " + "\n  ".join(tried))

# -------------------- QRELS LOADER --------------------

def load_qrels(path: str) -> Dict[str, Dict[int, float]]:
    """
    Returns {qid: {doc_id: graded_rel(float)}}.
    Supports:
      - dict-of-dicts: {"Q1": {"101": 2, ...}, ...}
      - list entries with:
         * {"qid": "...", "relevance_scores": {...}}
         * {"qid": "...", "relevance": {...}}
         * {"qid": "...", "docs": [{"doc_id": 101, "rel"/"score"/"label"/"grade": 2}, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    gold: Dict[str, Dict[int, float]] = {}

    # Case: dict-of-dicts
    if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
        for qid, m in raw.items():
            gold[str(qid)] = {int(k): float(v) for k, v in m.items()}
        return gold

    # Case: list of dicts with variants
    if isinstance(raw, list):
        for e in raw:
            if not isinstance(e, dict):
                continue
            qid = e.get("qid") or e.get("query_id") or e.get("id") or e.get("qid_str")
            if qid is None:
                continue
            qid = str(qid)

            if isinstance(e.get("relevance_scores"), dict):
                gold.setdefault(qid, {}).update({int(k): float(v) for k, v in e["relevance_scores"].items()})
                continue
            if isinstance(e.get("relevance"), dict):
                gold.setdefault(qid, {}).update({int(k): float(v) for k, v in e["relevance"].items()})
                continue
            if isinstance(e.get("docs"), list):
                for d in e["docs"]:
                    if not isinstance(d, dict):
                        continue
                    did = d.get("doc_id", d.get("id", d.get("document_id")))
                    rel = d.get("rel", d.get("score", d.get("label", d.get("relevance", d.get("grade", 0)))))
                    if did is None:
                        continue
                    gold.setdefault(qid, {})[int(did)] = float(rel)
                continue

    return gold

# -------------------- RUN LOADER --------------------

def load_run(path: str) -> List[dict]:
    """
    Expects a list:
      [{"qid": "...", "results": [{"doc_id": int, "score": float}, ...]}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Run file must be a JSON array: {path}")
    # light schema check
    for i, row in enumerate(data):
        if not isinstance(row, dict) or "qid" not in row or "results" not in row:
            raise ValueError(f"Invalid run row {i} in {path}")
        if not isinstance(row["results"], list):
            raise ValueError(f"Invalid 'results' type at row {i} in {path}")
    return data

# -------------------- METRICS --------------------

def average_precision_at_k(ranked_docs: List[int], relset: Set[int], k: int) -> float:
    """AP@K, treating any relevance > 0 as relevant. If no relevant docs, returns 0.0."""
    if k <= 0:
        k = len(ranked_docs)
    hit = 0
    ap = 0.0
    used: Set[int] = set()  # de-dup if a run repeats a doc_id; first occurrence counts
    i_rank = 0
    for did in ranked_docs:
        if did in used:
            continue
        used.add(did)
        i_rank += 1
        if i_rank > k:
            break
        if did in relset:
            hit += 1
            ap += hit / i_rank
    if not relset:
        return 0.0
    return ap / len(relset)

def reciprocal_rank_at_k(ranked_docs: List[int], relset: Set[int], k: int) -> float:
    """MRR@K: first relevant document's reciprocal rank; 0 if none."""
    if k <= 0:
        k = len(ranked_docs)
    used: Set[int] = set()
    i_rank = 0
    for did in ranked_docs:
        if did in used:
            continue
        used.add(did)
        i_rank += 1
        if i_rank > k:
            break
        if did in relset:
            return 1.0 / i_rank
    return 0.0

# -------------------- EVALUATION --------------------

def evaluate_run(run_path: str, qrels: Dict[str, Dict[int, float]], k: int) -> Tuple[float, float, int, int]:
    """
    Returns (MAP@K, MRR@K, num_eval_queries, num_covered_queries_in_run)
    - Evaluates over all qids present in qrels (queries without relevant docs contribute 0 AP).
    - If a qid is missing in the run, it's treated as an empty ranking (AP=0, RR=0).
    """
    data = load_run(run_path)

    # Build map qid -> ranked list
    run_map: Dict[str, List[int]] = {}
    for row in data:
        qid = str(row["qid"])
        ranked = [int(r.get("doc_id")) for r in row.get("results", []) if "doc_id" in r]
        run_map[qid] = ranked

    aps, rrs = [], []
    covered = 0
    for qid, judg in qrels.items():
        ranked = run_map.get(qid, [])
        if ranked:
            covered += 1
        relset = {int(d) for d, g in judg.items() if float(g) > 0.0}
        aps.append(average_precision_at_k(ranked, relset, k))
        rrs.append(reciprocal_rank_at_k(ranked, relset, k))

    n_q = len(qrels)
    MAP = sum(aps) / n_q if n_q else 0.0
    MRR = sum(rrs) / n_q if n_q else 0.0
    return MAP, MRR, n_q, covered

# -------------------- MAIN --------------------

def _scan_runs_default() -> List[str]:
    files = sorted(glob.glob("runs/*.json"))
    return files

def main():
    parser = argparse.ArgumentParser(description="Evaluate runs/*.json with MAP@K (and MRR@K).")
    parser.add_argument("--runs", nargs="*", help="Specific run files (default: scan runs/*.json)")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K (default: 10)")
    args = parser.parse_args()

    qrels_path = _resolve_qrels_path()
    qrels = load_qrels(qrels_path)

    runs = args.runs if args.runs else _scan_runs_default()
    if not runs:
        print("No run files found under ./runs. Generate one with system/search_system.py.", file=sys.stderr)
        sys.exit(2)

    print(f"Judgments: {len(qrels)} queries   (K={args.k})")
    rows: List[Tuple[str, float, float, int, int]] = []
    for rp in runs:
        try:
            MAP, MRR, n_q, covered = evaluate_run(rp, qrels, args.k)
            rows.append((os.path.basename(rp), MAP, MRR, n_q, covered))
        except Exception as e:
            print(f"[skip] {rp}: {e}", file=sys.stderr)

    # Sort by MAP desc, then filename asc
    rows.sort(key=lambda r: (-r[1], r[0]))

    # Print table
    w = max(12, max(len(name) for name, *_ in rows)) if rows else 12
    print("\n" + "Run".ljust(w) + "  MAP@K    MRR@K    #Q  covQ")
    print("-" * w + "  -------  -------  ---  ----")
    for name, MAP, MRR, n_q, cov in rows:
        print(f"{name.ljust(w)}  {MAP:7.4f}  {MRR:7.4f}  {n_q:3d}  {cov:4d}")

if __name__ == "__main__":
    main()
