"""Entry point for SWE-QA-Bench evaluation with GRAGC."""

import argparse
from pathlib import Path
from pprint import pprint

from ragc.test.swe_qa_bench import SWEQABenchConfig
from ragc.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRAGC on SWE-QA-Bench")

    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to .yaml config for SWE-QA-Bench inference",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for per-repo answer JSONL files",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("args:")
    pprint(args.__dict__)
    print("-" * 80)

    cfg: SWEQABenchConfig = load_config(SWEQABenchConfig, args.config)
    bench = cfg.create()

    print(f"Starting SWE-QA-Bench evaluation...")
    bench.generate_answers_per_repo(output_dir=args.output)
    print(f"Done. Answers written to {args.output}")


if __name__ == "__main__":
    main()
