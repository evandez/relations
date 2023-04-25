import argparse
from pathlib import Path

from src import benchmarks, data, models, operators
from src.utils import logging_utils

import torch


def main(args: argparse.Namespace) -> None:
    """Run one of the benchmarks to show everything works end to end."""
    logging_utils.configure(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        mt = models.load_model(args.model, fp16=args.fp16, device=device)
        dataset = data.load_dataset()
        estimator = operators.JacobianEstimator(mt=mt, h_layer=args.h_layer)
        results = benchmarks.faithfulness(dataset=dataset, estimator=estimator)

    results_file = Path("results.json")
    with results_file.open("w") as handle:
        handle.write(results.to_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-layer", type=int, default=5, help="layer to get h from")
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
