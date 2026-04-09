#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

files=()
while IFS= read -r -d '' f; do
    files+=("$f")
done < <(find outputs -type f -name '*.jsonl' ! -name '*__raw.jsonl' -print0)

if [ ${#files[@]} -eq 0 ]; then
    echo "No .jsonl files to clean."
    exit 0
fi

uv run python -m ragc.test.clean "${files[@]}"
