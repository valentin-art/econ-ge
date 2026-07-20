#!/bin/bash
# Runs the BEA job chain end-to-end: extract -> parse -> transform.
#   extract_bea.py       BEA API -> data/external/bea/ (raw JSON, needs BEA_API_KEY)
#   parse_bea_bronze.py  data/external/bea/ -> data/bronze/bea/ (tidy parquet)
#   transform_bea_silver.py  data/bronze/bea/ -> data/silver/bea/ (silver panel)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python -m src.jobs.extract_bea
uv run python -m src.jobs.parse_bea_bronze
uv run python -m src.jobs.transform_bea_silver
