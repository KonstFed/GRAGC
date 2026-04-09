import argparse
import json
import re
from pathlib import Path

import pandas as pd

def extract_python_code_blocks(text):
    """Extract only Python code blocks from text, ignoring all other content."""
    pattern = r"```python(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)

    code_blocks = []
    for match in matches:
        code_block = match.group(1).strip()
        if code_block:
            code_blocks.append(code_block)

    return code_blocks


def align_body(code: str) -> str:
    """Mirror of `CompletionGenerator.__align` for chat-model outputs.

    Drops everything before the function body (decorators, `def ...:`, preamble),
    keeps the contiguous indented block as the body, and cuts at the first
    zero-indent line after the body starts. If no indented lines exist at all,
    the input is assumed to already be a bare body and is indented by 4 spaces.
    """
    lines = code.split("\n")

    # Find the first line that looks like an indented body line.
    start_ix = 0
    while start_ix < len(lines) and not lines[start_ix].startswith("    "):
        start_ix += 1

    # No indented line found — assume the whole thing is already a bare body.
    if start_ix >= len(lines):
        body = "\n".join("    " + line if line else line for line in lines)
        return body.strip("\n")

    # Extend through the contiguous body (blank lines or further indented lines).
    end_ix = start_ix + 1
    while end_ix < len(lines) and (lines[end_ix] == "" or lines[end_ix].startswith(" ")):
        end_ix += 1

    return "\n".join(lines[start_ix:end_ix]).strip("\n")


def clean_single(code: str) -> str:
    """Clean single code instance: strip ```python``` fences and align to body-only."""
    blocks = extract_python_code_blocks(code)
    if blocks:
        code = blocks[0]
    return align_body(code)


def clean(completions: pd.DataFrame) -> pd.DataFrame:
    """Standartize LLM output for insertion into tests."""
    completions["completion"] = completions["completion"].apply(clean_single)
    return completions


def clean_jsonl_inplace(path: Path) -> None:
    """Apply `clean_single` to the `completion` field of every line in a .jsonl file, in-place."""
    path = Path(path)
    with open(path, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if "completion" in record:
            record["completion"] = clean_single(record["completion"])
        cleaned.append(json.dumps(record))

    with open(path, "w") as f:
        f.write("\n".join(cleaned) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean LLM completions in .jsonl files in-place (strip ```python``` fences)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more .jsonl files to clean in-place.",
    )
    args = parser.parse_args()

    for path in args.paths:
        print(f"Cleaning {path}")
        clean_jsonl_inplace(path)


if __name__ == "__main__":
    main()
