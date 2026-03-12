#!/bin/bash
# Step 2: Run GNN inference on SWE-QA-Bench
# Usage: bash run/swe_qa_inference.sh
#
# Prerequisites:
#   1. Dataset created via run/swe_qa_create_ds.sh
#   2. GNN model trained (BEST_CHECKPOINT.pt path set in config)
#   3. API_KEY env var set for OpenAI/OpenRouter
#
# Output: per-repo JSONL files in output/swe_qa_bench/gnn/

uv run python -m ragc.test.swe_qa_inference \
    -c configs/swe_qa_bench/gnn/greedy.yml \
    -o output/swe_qa_bench/gnn
