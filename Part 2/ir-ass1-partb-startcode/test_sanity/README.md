# Sanity Check (Smoke Tests)

This folder provides **minimal smoke tests** to ensure your code imports and
runs without crashing. It does **not** check correctness.

## What it checks

- **Task 1 (index)**
  - `index.builders.create_all_indexes(...)` writes a single package
  - `index.access.get_posting_list(...)` -> `List[int]`
  - `index.access.find_wildcard_matches(...)` -> `List[str]`
  - `index.access.get_term_positions(...)` -> `List[int]`

- **Task 2 (query processing)**
  - `query_processing.detection.detect_query_type`
  - `query_processing.query_process.convert_natural_language`
  - `query_processing.boolean.process_boolean_query`
  - `query_processing.wildcard.process_wildcard_query`
  - `query_processing.proximity.process_proximity_query`
  - `query_processing.query_process.process_query`

- **Task 3 (ranking)**
  - `ranking.rankers.rank_documents(...)` -> `(List[int], List[float])`

- **Task 4 (CLI)**
  - Runs `python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_sanity.json`
  - Validates output JSON schema (`qid`, `doc_ids` list)

## Run

From the repository root:

```bash
python test_sanity/check_submission.py
```

Exit code:

- 0 – all smoke tests passed

- 1 – at least one check failed (see messages)

## Notes

- Uses only the Python standard library.

- Builds a tiny temporary index in `test_sanity/_tmp/index_pkg.pkl` for Task 1/2/3 smoke.

- For Task 4, it uses your dev data at `data/dev/`.

- Creates `runs/` and `cache/` if missing.