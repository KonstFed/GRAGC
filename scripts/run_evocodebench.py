#!/usr/bin/env python3
"""Run evocodebench inference across multiple configs in parallel.

Usage:
    python scripts/run_evocodebench.py                    # all configs, 3 workers
    python scripts/run_evocodebench.py -w 2               # limit to 2 workers
    python scripts/run_evocodebench.py -m gnn             # only gnn mode
    python scripts/run_evocodebench.py -m gnn local_context  # gnn + local_context
    python scripts/run_evocodebench.py --dry-run           # preview without running
    python scripts/run_evocodebench.py --models gpt-oss-120b deepseekv3  # only these models
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "evocodebench"
OUTPUTS_DIR = ROOT / "outputs" / "evocodebench"
LOGS_DIR = ROOT / "logs" / "evocodebench"

MODES = ["gnn", "local_context", "gnn_local_context", "without_context"]


def discover_configs(modes: list[str], models: list[str] | None = None) -> list[tuple[str, Path]]:
    """Return (mode, config_path) pairs for all greedy_*.yml in given modes.
    If models is provided, only include configs whose stem contains one of the model keywords.
    """
    configs = []
    for mode in modes:
        mode_dir = CONFIGS_DIR / mode
        if not mode_dir.exists():
            continue
        for cfg in sorted(mode_dir.glob("greedy_*.yml")):
            if models and not any(m in cfg.stem for m in models):
                continue
            configs.append((mode, cfg))
    return configs


def run_single(mode: str, config_path: Path) -> dict:
    """Run a single inference config. Returns a result dict."""
    stem = config_path.stem  # e.g. greedy_deepseekv3
    output_path = OUTPUTS_DIR / mode / f"{stem}.jsonl"
    log_path = LOGS_DIR / mode / f"{stem}.log"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tag = f"[{mode}/{stem}]"
    started = datetime.now()

    cmd = [
        sys.executable, "-m", "ragc.test.inference",
        "-t", "completion",
        "-c", str(config_path),
        "-o", str(output_path),
    ]

    with open(log_path, "w") as log_file:
        log_file.write(f"{'='*80}\n")
        log_file.write(f"{tag} started at {started:%Y-%m-%d %H:%M:%S}\n")
        log_file.write(f"cmd: {' '.join(cmd)}\n")
        log_file.write(f"{'='*80}\n\n")
        log_file.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = datetime.now() - started

    return {
        "tag": tag,
        "mode": mode,
        "model": stem,
        "returncode": proc.returncode,
        "elapsed": str(elapsed).split(".")[0],
        "log": str(log_path),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run evocodebench inference in parallel")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Max parallel workers (default: 3)")
    parser.add_argument("-m", "--modes", nargs="+", choices=MODES, default=MODES, help="Modes to run")
    parser.add_argument("--models", nargs="+", default=None, help="Filter by model name substring (e.g. deepseekv3 qwen2.5-72b)")
    parser.add_argument("--dry-run", action="store_true", help="List configs without running")
    args = parser.parse_args()

    configs = discover_configs(args.modes, args.models)
    if not configs:
        print("No configs found.")
        return

    print(f"\n{'='*60}")
    print(f"  Evocodebench batch runner")
    print(f"  {len(configs)} configs | {args.workers} workers | modes: {', '.join(args.modes)}")
    print(f"{'='*60}\n")

    for mode, cfg in configs:
        status = "  " if not args.dry_run else "  [dry-run] "
        print(f"{status}{mode:20s} {cfg.stem}")

    if args.dry_run:
        return

    print(f"\nStarting at {datetime.now():%Y-%m-%d %H:%M:%S}\n{'-'*60}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_single, mode, cfg): (mode, cfg)
            for mode, cfg in configs
        }

        for future in as_completed(futures):
            result = future.result()
            status = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"  [{status}] {result['tag']:40s} {result['elapsed']}  ->  {result['log']}")
            results.append(result)

    # Summary
    ok = sum(1 for r in results if r["returncode"] == 0)
    fail = len(results) - ok
    print(f"\n{'='*60}")
    print(f"  Done: {ok} passed, {fail} failed out of {len(results)}")
    if fail:
        print(f"\n  Failed runs:")
        for r in results:
            if r["returncode"] != 0:
                print(f"    {r['tag']}  ->  check {r['log']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
