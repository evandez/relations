import argparse
import json
import logging
from typing import Any

from src import benchmarks, data, models, operators
from src.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)

BENCHMARKS = ("reconstruction", "faithfulness")
ESTIMATORS = {
    "j": operators.JacobianEstimator,
    "j-icl": operators.JacobianIclEstimator,
    "corner-gd": operators.CornerGdEstimator,
}


def main(args: argparse.Namespace) -> None:
    """Run one of the benchmarks to show everything works end to end."""
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        mt = models.load_model(args.model, fp16=args.fp16, device=device)
        dataset = data.load_dataset()

        estimator = ESTIMATORS[args.estimator](
            mt=mt,
            h_layer=args.h_layer,
            z_layer=args.z_layer,
        )

        for bench in args.benchmarks:
            results: Any
            if bench == "reconstruction":
                results = benchmarks.reconstruction(
                    dataset=dataset, estimator=estimator
                )
            elif bench == "faithfulness":
                results = benchmarks.faithfulness(dataset=dataset, estimator=estimator)

            metrics_json = json.dumps(results.metrics.to_dict(), indent=4)

            results_file = experiment.results_dir / f"{bench}_metrics.json"
            with results_file.open("w") as handle:
                handle.write(metrics_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-layer", type=int, default=5, help="layer to get h from")
    parser.add_argument("--z-layer", type=int, help="layer to get z from")
    parser.add_argument(
        "--estimator",
        "-e",
        default="j-icl",
        choices=ESTIMATORS,
        help="lre estimator to use",
    )
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        help="benchmarks to run",
    )
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
