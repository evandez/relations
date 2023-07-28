"""Tools for reading and controlling the runtime environment."""
import os
import pathlib

from src.utils.typing import PathLike

ENV_DATA_DIR = "RELATIONS_DATA_DIR"
ENV_MODELS_DIR = "RELATIONS_MODELS_DIR"
ENV_RESULTS_DIR = "RELATIONS_RESULTS_DIR"
ENV_HPARAMS_DIR = "RELATIONS_HPARAMS_DIR"

DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "/data/arnab_sen_sharma/weights"  # "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_HPARAMS_DIR = "hparams"


def maybe_relative_to_repo(path: PathLike) -> pathlib.Path:
    """Resolve the (potentially relative) path.

    Args:
        path: The path to resolve. If the path is relative, it is assumed to be
            relative to the repository root. If this is already an absolute path,
            it is returned unchanged.

    Returns:
        The resolved path.

    """
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return pathlib.Path(__file__).parents[2] / path


def read_path(name: str, default: PathLike) -> pathlib.Path:
    """Try to read a path from the env.

    Args:
        name: Name of the env variable to read.
        default: The default path in case the environment variable
            does not exist. If relative, assumed to be relative from the repo root.

    Returns:
        The path, if the variable could be read and/or if default was provided.

    """
    read = os.environ.get(name)
    path: PathLike = maybe_relative_to_repo(default) if read is None else read
    return pathlib.Path(path)


def determine_data_dir(default: PathLike = DEFAULT_DATA_DIR) -> pathlib.Path:
    """Return directory containing project datasets.

    Args:
        default: Default to use if RELATIONS_DATA_DIR env variable is not set.
            Defaults to './data'.

    Returns:
        pathlib.Path: Directory data is stored in.

    """
    return read_path(ENV_DATA_DIR, default)


def determine_models_dir(default: PathLike = DEFAULT_MODELS_DIR) -> pathlib.Path:
    """Return directory containing project models.

    Args:
        default: Default to use if RELATIONS_MODELS_DIR env variable is not set.
            Defaults to './models'.

    Returns:
        Directory data is stored in.

    """
    return read_path(ENV_MODELS_DIR, default)


def determine_results_dir(default: PathLike = DEFAULT_RESULTS_DIR) -> pathlib.Path:
    """Return directory containing results from any scripts.

    Args:
        default: Default to use if RELATIONS_RESULTS_DIR env
            variable is not set. Defaults to './results'.

    Returns:
        Directory results are stored in.

    """
    return read_path(ENV_RESULTS_DIR, default)


def determine_hparams_dir(default: PathLike = DEFAULT_HPARAMS_DIR) -> pathlib.Path:
    """Return directory containing hyperparameters.

    Args:
        default: Default to use if RELATIONS_HPARAMS_DIR env
            variable is not set. Defaults to './hparams'.

    Returns:
        Directory hyparams are stored in.

    """
    return read_path(ENV_HPARAMS_DIR, default)
