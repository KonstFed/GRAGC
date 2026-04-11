"""Create Torch Geometric dataset from SWE-QA-Bench cloned repos."""

import argparse
import logging
import sys
from pathlib import Path

from ragc.datasets.train_dataset import TorchGraphDatasetConfig
from ragc.utils import load_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Create Torch Geometric dataset from SWE-QA-Bench repos.",
    )
    parser.add_argument("config", type=Path, help="Path to dataset config YAML")
    parser.add_argument(
        "--repos-dir",
        type=Path,
        required=True,
        help="Path to SWE-QA-Bench/datasets/repos/ directory containing cloned repos",
    )

    args = parser.parse_args()
    config_path = args.config.resolve()
    config: TorchGraphDatasetConfig = load_config(TorchGraphDatasetConfig, config_path)

    repos_dir = args.repos_dir.resolve()
    if not repos_dir.is_dir():
        logger.error("--repos-dir is not a directory: %s", repos_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("create_swe_qa_dataset")
    logger.info("=" * 60)
    logger.info("config:     %s", config_path)
    logger.info("root_path:  %s", config.root_path.resolve())
    logger.info("repos_dir:  %s", repos_dir)
    logger.info("")

    # Add each repo subdirectory
    total = 0
    for repo_p in sorted(repos_dir.iterdir()):
        if not repo_p.is_dir():
            continue
        config.add_repo(repo_p.absolute())
        total += 1
        logger.info("  added: %s", repo_p.name)

    logger.info("")
    logger.info("Total repos: %s", total)

    if total == 0:
        logger.error("No repo directories found.")
        sys.exit(1)

    logger.info("")
    logger.info("Building dataset at %s ...", config.root_path.resolve())
    config.create()
    logger.info("Done. Torch Geometric dataset created.")
