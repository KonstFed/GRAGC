#!/bin/bash
# Step 1: Parse SWE-QA-Bench repos into graph dataset
# Usage: bash run/swe_qa_create_ds.sh <path-to-SWE-QA-Bench-repos-dir>
#
# Example:
#   bash run/swe_qa_create_ds.sh ../SWE-QA-Bench/SWE-QA-Bench/datasets/repos

REPOS_DIR="${1:?Usage: $0 <path-to-SWE-QA-Bench/datasets/repos>}"

uv run python -m ragc.datasets.create_swe_qa_dataset \
    configs/swe_qa_bench/create_ds.yml \
    --repos-dir "$REPOS_DIR"
