import argparse
import json
import os
from pathlib import Path
from pprint import pprint

from ragc.test.test_config import TestInferenceConfig, EvoCodeBenchInferenceConfig
from ragc.utils import load_config
from ragc.test.clean import clean_single


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        help='Either "retrieval" or "completion"',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to an output file with completions",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to .yaml config for inference",
    )

    return parser.parse_args()


def generate_completions(
    output_path: str | Path,
    config_path: str | Path,
):
    output_path = Path(output_path).absolute().resolve()
    raw_output_path = output_path.parent / f"{output_path.stem}__raw.jsonl"
    output_dir = os.path.dirname(output_path)
    raw_output_dir = os.path.dirname(raw_output_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_output_dir, exist_ok=True)

    test_inference_cfg: EvoCodeBenchInferenceConfig = load_config(EvoCodeBenchInferenceConfig, config_path)
    test_inference = test_inference_cfg.create()
    print(f"Loaded {len(test_inference.tasks)} tasks")
    print("-" * 256)

    f = open(output_path, "w")
    f.close()

    f = open(raw_output_path, "w")
    f.close()

    for result in test_inference.generate_completion():
        # change if we have multiple pass@k > 1
        result["idx"] = 0

        with open(raw_output_path, "a") as f:
            json_line = json.dumps(result)
            f.write(json_line + "\n")

        # result["completion"] = clean_single(result["completion"])

        with open(output_path, "a") as f:
            json_line = json.dumps(result)
            f.write(json_line + "\n")


def retrieval_metrics(
    output_path: str | Path,
    config_path: str | Path,
):
    output_path = Path(output_path).absolute().resolve()
    test_inference_cfg: TestInferenceConfig = load_config(TestInferenceConfig, config_path)
    test_inference = test_inference_cfg.create()
    print(f"Loaded {len(test_inference.tasks)} tasks")
    print("-" * 256)

    recalls = []
    precision = []
    for gold, gotten in test_inference.generate_retrieval_pairs():
        gold_names = set(g.name for g in gold)
        real = set(r.name for r in gotten)
        hit = gold_names.intersection(real)
        recalls.append(len(hit) / len(gold))
        precision.append(len(hit) / len(gotten))

    result = {
        "avg. recall": sum(recalls) / len(recalls),
        "avg. precision": sum(precision) / len(precision),
    }

    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    # parse args
    args = parse_args()
    print("args:")
    pprint(args.__dict__)
    print("-" * 256)


    # run generation

    match args.task:
        case "completion":
            generate_completions(
                args.output,
                config_path=args.config,
            )
        case "retrieval":
            retrieval_metrics(
                args.output,
                config_path=args.config,
            )

