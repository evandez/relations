"""Utilities for managing experiment runtimes and results."""
import argparse
import json
import logging
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from src.utils import env_utils
from src.utils.typing import PathLike

import numpy
import torch
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)

DEFAULT_SEED = 123456


@dataclass(frozen=True)
class Experiment:
    """A configured experiment."""

    name: str
    results_dir: Path
    seed: int


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    logger.info("setting all seeds to %d", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_results_dir(
    experiment_name: str,
    root: PathLike | None = None,
    args: argparse.Namespace | None = None,
    args_file_name: str | None = None,
    clear_if_exists: bool = False,
) -> Path:
    """Create a directory for storing experiment results.

    Args:
        name: Experiment name.
        root: Root directory to store results in. Consults env if not set.
        args: If set, save the full argparse namespace as JSON.
        args_file: Save args file here.
        clear_if_exists: Clear the results dir if it already exists.

    Returns:
        The initialized results directory.

    """
    if root is None:
        root = env_utils.determine_results_dir()
    root = Path(root)

    results_dir = root / experiment_name
    results_dir = results_dir.resolve()

    if results_dir.exists():
        logger.info(f"rerunning experiment {experiment_name}")
        if clear_if_exists:
            logger.info(f"clearing previous results from {results_dir}")
            shutil.rmtree(results_dir)

    results_dir.mkdir(exist_ok=True, parents=True)
    if args is not None:
        if args_file_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args_file_name = f"args-{timestamp}.json"
        args_file = results_dir / args_file_name
        logger.info(f"saving args to {args_file}")
        with args_file.open("w") as handle:
            json.dump({key: str(value) for key, value in vars(args).items()}, handle)

    return results_dir


ResultsT = TypeVar("ResultsT", bound=DataClassJsonMixin)


def load_results_file(
    *,
    results_dir: PathLike | None,
    results_type: type[ResultsT],
    name: str,
    resume: bool,
) -> ResultsT | None:
    """Read an intermediate result, if present."""
    if results_dir is None or not resume:
        logger.debug("results_dir not set, so not reading intermediate results")
        return None

    relation_results_file = name_results_file(
        results_dir=results_dir,
        name=name,
    )
    if not relation_results_file.exists():
        logger.debug(f'no intermediate results for "{name}"')
        return None

    logger.debug(f"reading intermediate results from {relation_results_file}")
    with relation_results_file.open("r") as handle:
        return results_type.from_json(handle.read())


def save_results_file(
    *,
    results_dir: PathLike | None,
    name: str,
    results: ResultsT,
) -> None:
    """Save an intermediate result."""
    if results_dir is None:
        logger.debug(
            "results_dir not set, so not saving intermediate results for " f'"{name}"'
        )
        return None
    relation_results_file = name_results_file(results_dir=results_dir, name=name)
    logger.debug(f"saving intermediate results to {relation_results_file}")
    relation_results_file.parent.mkdir(exist_ok=True, parents=True)
    with relation_results_file.open("w") as handle:
        handle.write(results.to_json())


def name_results_file(
    *,
    results_dir: PathLike,
    name: str,
) -> Path:
    """Create file name for an intermediate result."""
    name_slug = name.replace(" ", "_").replace("'", "")
    return Path(results_dir) / f"{name_slug}.json"


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Add args common to all experiments.

    The args include:
        --experiment-name (-n): Unique identifier for this experiment.
            Defaults to script name.
        --results-dir: Root directory containing all experiment folders.
        --clear-results-dir: If set, experiment-specific results directory is cleared.
        --args-file-name: Dump all args to this file; defaults to generated name.
        --seed: Random seed.

    """
    parser.add_argument(
        "--experiment-name",
        "-n",
        help="unique name for the experiment",
    )
    parser.add_argument(
        "--results-dir", type=Path, help="root directory containing experiment results"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="resume previous run"
    )
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        default=False,
        help="clear any old results and start anew",
    )
    parser.add_argument("--args-file-name", help="file name for args dump")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")


def setup_experiment(args: argparse.Namespace) -> Experiment:
    """Configure experiment from the args."""
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = Path(sys.argv[0]).stem
    seed = args.seed

    logger.info(f"setting up experiment {experiment_name}")

    set_seed(seed)

    results_dir = create_results_dir(
        experiment_name,
        root=args.results_dir,
        args=args,
        args_file_name=args.args_file_name,
        clear_if_exists=args.clear_results_dir,
    )

    return Experiment(
        name=experiment_name,
        results_dir=results_dir,
        seed=seed,
    )
