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
        description=(
            "Create Torch Geometric dataset from repos or graphs (e.g. EvoCodeBench repos)."
        ),
    )
    parser.add_argument(
        "--evocodebench",
        type=Path,
        required=False,
        help="Path to EvoCodeBench repos root (category subdirs: Communications/, Internet/, ...)",
    )
    parser.add_argument("config", type=Path, help="Path to dataset config YAML")

    args = parser.parse_args()
    config_path = args.config.resolve()
    config: TorchGraphDatasetConfig = load_config(TorchGraphDatasetConfig, config_path)

    logger.info("=" * 60)
    logger.info("create_dataset")
    logger.info("=" * 60)
    logger.info("config:        %s", config_path)
    logger.info("root_path:     %s", config.root_path.resolve())
    evocodebench_src = (
        args.evocodebench.resolve() if args.evocodebench else "not set (config repos_path/graphs_path)"
    )
    logger.info("evocodebench: %s", evocodebench_src)
    logger.info("")

    if args.evocodebench is not None:
        global_p = Path(args.evocodebench).resolve()
        if not global_p.is_dir():
            logger.error("--evocodebench is not a directory: %s", global_p)
            sys.exit(1)
        total = 0
        for domain in sorted(global_p.iterdir()):
            if not domain.is_dir():
                continue
            repos = [p for p in domain.iterdir() if p.is_dir()]
            for repo_p in repos:
                config.add_repo(repo_p.absolute())
            n = len(repos)
            total += n
            logger.info("  %s: %s repos", domain.name, n)
        logger.info("  total repos added: %s", total)
        logger.info("")
        if total == 0:
            logger.error("No repo directories found under --evocodebench.")
            sys.exit(1)

    n_repos = len(config._repos)  # noqa: SLF001
    if n_repos:
        logger.info("Building dataset at %s from %s repos...", config.root_path.resolve(), n_repos)
    else:
        logger.info(
            "Building dataset at %s from config (repos_path/graphs_path)...",
            config.root_path.resolve(),
        )
    logger.info("")
    config.create()
    logger.info("Done. Torch Geometric dataset created.")
