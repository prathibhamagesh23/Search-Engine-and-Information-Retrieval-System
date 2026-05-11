#!/usr/bin/env python3
# test_sanity/check_submission.py
# Minimal smoke tests to ensure the submission imports and runs.
# This does NOT check correctness—only that functions execute without crashing.

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = REPO_ROOT / "test_sanity" / "_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---- tiny demo corpus for Task 1/2/3 smoke only ----
DOC_IDS: List[int] = [10, 20, 30]
TOKENIZED_DOCS: List[List[str]] = [
    ["climate", "change"],       # doc 10
    ["machine", "learning"],     # doc 20
    ["climate", "policy"],       # doc 30
]
QUERY_SIMPLE = "climate AND change"
QUERY_PHRASE = '"machine learning"'
QUERY_WILDCARD = "climat*"
QUERY_NEAR = "climate NEAR/1 change"

RESULTS = []
INDEX_PATH = TMP_DIR / "index_pkg.pkl"

def record(name: str, ok: bool, msg: str = ""):
    RESULTS.append((name, ok, msg))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {msg}" if msg else ""))

def import_or_fail(module_path: str):
    try:
        return __import__(module_path, fromlist=["*"])
    except Exception as e:
        raise ImportError(f"Import failed for '{module_path}': {e}") from e

def step_task1_build_index():
    name = "Task1: create_all_indexes() builds one package"
    try:
        builders = import_or_fail("index.builders")
        if not hasattr(builders, "create_all_indexes"):
            raise AttributeError("Missing function: index.builders.create_all_indexes(...)")
        builders.create_all_indexes(TOKENIZED_DOCS, str(INDEX_PATH), doc_ids=DOC_IDS)
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Index package not created at {INDEX_PATH}")
        record(name, True)
    except Exception as e:
        record(name, False, str(e))

def step_task1_access():
    try:
        access = import_or_fail("index.access")
    except Exception as e:
        record("Task1: import index.access", False, str(e))
        return

    # get_posting_list
    name = "Task1: get_posting_list('climate', index_path)"
    try:
        if not hasattr(access, "get_posting_list"):
            raise AttributeError("Missing function: index.access.get_posting_list(...)")
        pl = access.get_posting_list("climate", str(INDEX_PATH))
        if not isinstance(pl, list) or any(not isinstance(x, int) for x in pl):
            raise TypeError("Expected List[int] from get_posting_list")
        # Basic correctness check: 'climate' appears in docs 10 and 30
        expected_docs = {10, 30}
        if set(pl) != expected_docs:
            record(name, True, f"returned {len(pl)} doc id(s) but expected docs {expected_docs}, got {set(pl)}")
        else:
            record(name, True, f"returned {len(pl)} doc id(s) - correct!")
    except Exception as e:
        record(name, False, str(e))

    # find_wildcard_matches
    name = r"Task1: find_wildcard_matches('$cl', index_path)"
    try:
        if not hasattr(access, "find_wildcard_matches"):
            raise AttributeError("Missing function: index.access.find_wildcard_matches(...)")
        matches = access.find_wildcard_matches("$cl", str(INDEX_PATH))
        if not isinstance(matches, list) or any(not isinstance(t, str) for t in matches):
            raise TypeError("Expected List[str] from find_wildcard_matches")
        # Basic correctness check: '$cl' should match 'climate'
        if "climate" not in matches:
            record(name, True, f"returned {len(matches)} term(s) but expected 'climate' in results, got {matches}")
        else:
            record(name, True, f"returned {len(matches)} term(s) - correct!")
    except Exception as e:
        record(name, False, str(e))

    # get_term_positions
    name = "Task1: get_term_positions('climate', doc_id=10, index_path)"
    try:
        if not hasattr(access, "get_term_positions"):
            raise AttributeError("Missing function: index.access.get_term_positions(...)")
        pos = access.get_term_positions("climate", 10, str(INDEX_PATH))
        if not isinstance(pos, list) or any(not isinstance(p, int) for p in pos):
            raise TypeError("Expected List[int] from get_term_positions")
        # Basic correctness check: 'climate' is at position 0 in doc 10
        if pos != [0]:
            record(name, True, f"returned {len(pos)} position(s) but expected [0], got {pos}")
        else:
            record(name, True, f"returned {len(pos)} position(s) - correct!")
    except Exception as e:
        record(name, False, str(e))

def step_task2_processors():
    name_base = "Task2"
    try:
        det = import_or_fail("query_processing.detection")
        qp  = import_or_fail("query_processing.query_process")
        bo  = import_or_fail("query_processing.boolean")
        wc  = import_or_fail("query_processing.wildcard")
        prx = import_or_fail("query_processing.proximity")
    except Exception as e:
        record(f"{name_base}: import query_processing modules", False, str(e))
        return

    # detect_query_type
    name = f"{name_base}: detect_query_type('{QUERY_SIMPLE}')"
    try:
        if not hasattr(det, "detect_query_type"):
            raise AttributeError("Missing function: detection.detect_query_type")
        qt = det.detect_query_type(QUERY_SIMPLE)
        if not isinstance(qt, str):
            raise TypeError("detect_query_type should return str")
        record(name, True, f"type='{qt}'")
    except Exception as e:
        record(name, False, str(e))

    # convert_natural_language
    name = f"{name_base}: convert_natural_language('climate change')"
    try:
        if not hasattr(qp, "convert_natural_language"):
            raise AttributeError("Missing function: query_process.convert_natural_language")
        s = qp.convert_natural_language("climate change")
        if not isinstance(s, str):
            raise TypeError("convert_natural_language should return str")
        record(name, True, f"converted='{s}'")
    except Exception as e:
        record(name, False, str(e))

    # process_boolean_query
    name = f"{name_base}: process_boolean_query('{QUERY_SIMPLE}', index_path)"
    try:
        if not hasattr(bo, "process_boolean_query"):
            raise AttributeError("Missing function: boolean.process_boolean_query")
        res = bo.process_boolean_query(QUERY_SIMPLE, str(INDEX_PATH))
        if not isinstance(res, set) or any(not isinstance(x, int) for x in res):
            raise TypeError("process_boolean_query should return Set[int]")
        record(name, True, f"returned {len(res)} doc id(s)")
    except Exception as e:
        record(name, False, str(e))

    # process_wildcard_query
    name = f"{name_base}: process_wildcard_query('{QUERY_WILDCARD}', index_path)"
    try:
        if not hasattr(wc, "process_wildcard_query"):
            raise AttributeError("Missing function: wildcard.process_wildcard_query")
        res = wc.process_wildcard_query(QUERY_WILDCARD, str(INDEX_PATH))
        if not isinstance(res, set) or any(not isinstance(x, int) for x in res):
            raise TypeError("process_wildcard_query should return Set[int]")
        record(name, True, f"returned {len(res)} doc id(s)")
    except Exception as e:
        record(name, False, str(e))

    # process_proximity_query
    name = f"{name_base}: process_proximity_query('{QUERY_NEAR}', index_path)"
    try:
        if not hasattr(prx, "process_proximity_query"):
            raise AttributeError("Missing function: proximity.process_proximity_query")
        res = prx.process_proximity_query(QUERY_NEAR, str(INDEX_PATH))
        if not isinstance(res, set) or any(not isinstance(x, int) for x in res):
            raise TypeError("process_proximity_query should return Set[int]")
        record(name, True, f"returned {len(res)} doc id(s)")
    except Exception as e:
        record(name, False, str(e))

    # process_query (router)
    name = f"{name_base}: process_query('{QUERY_PHRASE}', index_path)"
    try:
        if not hasattr(qp, "process_query"):
            raise AttributeError("Missing function: query_process.process_query")
        res = qp.process_query(QUERY_PHRASE, str(INDEX_PATH))
        if not isinstance(res, set) or any(not isinstance(x, int) for x in res):
            raise TypeError("process_query should return Set[int]")
        record(name, True, f"returned {len(res)} doc id(s)")
    except Exception as e:
        record(name, False, str(e))

def step_task3_ranker():
    name = "Task3: rank_documents(query_toks, candidate_docs, doc_ids, index_path, method='default')"
    try:
        rankers = import_or_fail("ranking.rankers")
        if not hasattr(rankers, "rank_documents"):
            raise AttributeError("Missing function: ranking.rankers.rank_documents(...)")

        query_toks = ["climate", "change"]
        candidate_docs = TOKENIZED_DOCS[:]  # aligned with DOC_IDS
        ranked_ids, scores = rankers.rank_documents(
            query_toks, candidate_docs, DOC_IDS, str(INDEX_PATH), method="default"
        )
        if not (isinstance(ranked_ids, list) and isinstance(scores, list)):
            raise TypeError("rank_documents should return Tuple[List[int], List[float]]")
        if len(ranked_ids) != len(scores):
            raise ValueError("ranked_doc_ids and scores must have same length")
        if sorted(ranked_ids) != sorted(DOC_IDS):
            raise ValueError("ranked_doc_ids must be a permutation of input doc_ids")
        record(name, True, f"ranked {len(ranked_ids)} docs")
    except Exception as e:
        record(name, False, str(e))

def step_task4_cli():
    name = "Task4: CLI system/search_system.py data/dev -> runs/run_sanity.json"
    try:
        # Ensure required dirs exist
        (REPO_ROOT / "runs").mkdir(exist_ok=True)
        (REPO_ROOT / "cache").mkdir(exist_ok=True)

        # Check dev data presence
        dev_dir = REPO_ROOT / "data" / "dev"
        q_path = dev_dir / "queries.json"
        d_path = dev_dir / "documents.jsonl"
        if not q_path.exists() or not d_path.exists():
            raise FileNotFoundError(
                "Expected dev data at data/dev/{queries.json, documents.jsonl}. "
                "Please include the dev files."
            )

        out_path = REPO_ROOT / "runs" / "run_sanity.json"
        if out_path.exists():
            out_path.unlink()

        script_path = REPO_ROOT / "system" / "search_system.py"
        if not script_path.exists():
            raise FileNotFoundError("Missing system/search_system.py")

        cmd = [sys.executable, "-u", str(script_path), str(q_path), str(d_path), str(out_path)]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(f"CLI exited {proc.returncode}\nSTDERR:\n{proc.stderr}")

        if not out_path.exists():
            raise FileNotFoundError("CLI did not produce the output JSON")

        data = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("Output JSON must be a non-empty list")
        item = data[0]
        if "qid" not in item or "doc_ids" not in item or not isinstance(item["doc_ids"], list):
            raise ValueError("Each result must have fields: 'qid' and 'doc_ids' (list)")
        record(name, True, f"produced {len(data)} result object(s)")
    except Exception as e:
        record(name, False, str(e))

def main():
    print("=== Sanity Check: starting ===")
    sys.path.insert(0, str(REPO_ROOT))

    step_task1_build_index()
    step_task1_access()
    step_task2_processors()
    step_task3_ranker()
    step_task4_cli()

    print("\n=== Summary ===")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    for name, ok, msg in RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} - {name}")
        if not ok and msg:
            print(f"       └─ {msg}")
    print(f"\nTotal: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
