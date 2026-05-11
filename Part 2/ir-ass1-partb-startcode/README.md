# Information Retrieval Systems — Part B (Starter Skeleton)

This repository is a clean starter for Part B of the IR assignment. It includes module stubs and entry points that match the required interfaces. Many functions are intentionally left unimplemented so students can complete Tasks 1–4.

## Setup

- Python 3.8+
- Optional: virtual environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Download minimal NLTK data used by utils/text_preprocessing.py
python - <<'PY'
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg)
```

## Repository layout

```
index/
  __init__.py
  access.py          # Task 1 access (O(1) in-memory lookups after load)
  builders.py        # Task 1 single builder: create_all_indexes(...)
  io.py              # gzip+pickle I/O helpers
metrics/
  eval_map.py        # Task 4: compute MAP over ./runs/ vs data/dev/relevance_judge.json
query_processing/
  boolean.py         # Task 2: process_boolean_query(...)
  detection.py       # Task 2: detect_query_type(...)
  proximity.py       # Task 2: process_proximity_query(...)
  query_process.py   # Task 2: convert_natural_language(...), process_query(...)
  wildcard.py        # Task 2: process_wildcard_query(...)
ranking/
  rankers.py         # Task 3: rank_documents(...); main prints Pearson table on dev set
system/
  search_system.py   # Task 4: batch CLI
utils/
  embeddings.py      # Part A: semantic_vector(...)
  ngram.py           # Part A: make_ngrams_tokens(...), make_ngrams_chars(...)
  positions.py       # Part A: make_positions(...)
  text_preprocessing.py  # Part A: preprocess(...)
  tfidf.py           # Part A: tfidf_variants(...)
data/
  dev/
    documents.jsonl
    queries.json
    relevance_judge.json
runs/                # Your output runs (*.json)
cache/               # Transient index package file written by the CLI
```

## What is provided vs to-do

- Provided: function signatures, I/O scaffolding, CLI shells, and data files under `data/dev/`.
- To implement (placeholders present):
  - Task 1: `index/builders.py:create_all_indexes`; `index/access.py` accessors.
  - Task 2: query detection, boolean/wildcard/proximity processors, OR-converter.
  - Task 3: `ranking/rankers.py:rank_documents` and a small main that prints Pearson correlation for your methods on the dev set.
  - Task 4: complete the pipeline inside `system/search_system.py` and `metrics/eval_map.py`.

No completed solutions are included; all core algorithmic functions are left as TODOs.

## Required CLIs (after you implement TODOs)

- Batch search run (Task 4):
```bash
python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_default.json

mine:  python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_default.json
```

- MAP evaluation over all files in `./runs/` (Task 4):
```bash
python metrics/eval_map.py
```

- Ranking development evaluation (Task 3):
```bash
python ranking/rankers.py
```

## Query types (detection is case-sensitive)

- Boolean: `AND` / `OR` / `NOT`, parentheses, and quoted phrases.
- Wildcard: patterns with `*`, expanded using char n‑grams with `$` boundaries.
- Proximity: `NEAR/k` between terms or quoted phrases (edge-to-edge distance).
- Natural language: whitespace tokens converted to `a OR b OR c`.

## Notes and constraints

- Single unified index package file; load once and cache for O(1) dictionary lookups in memory (disk load excluded from complexity).
- Deterministic outputs and tie-breaking (e.g., by ascending `doc_id`).
- Enforce the 20-document semantic budget per query inside ranking if you use semantic vectors.
- Do not check in persistent caches; the CLI writes to `cache/` per run and may overwrite.

## Academic integrity

Do not use Generative AI to produce code or algorithms. You may use it to edit your one-page write‑up (include the required declaration).
