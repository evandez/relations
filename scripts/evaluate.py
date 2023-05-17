import argparse
import logging
from typing import Any

from src import benchmarks, data, editors, models, operators
from src.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)

BENCHMARKS = ("reconstruction", "faithfulness", "causality")
ESTIMATORS = {
    "j": operators.JacobianEstimator,
    "j-icl-max": operators.JacobianIclMaxEstimator,
    "j-icl-mean": operators.JacobianIclMeanEstimator,
    "corner-gd": operators.CornerGdEstimator,
}
EDITORS = {
    "bl-h": editors.HiddenBaselineEditor,
    "bl-e": editors.EmbedBaselineEditor,
    "lr": editors.LowRankPInvEditor,
    "lr-e": editors.LowRankPInvEmbedEditor,
}


def main(args: argparse.Namespace) -> None:
    """Run one of the benchmarks to show everything works end to end."""
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        mt = models.load_model(args.model, fp16=args.fp16, device=device)
        dataset = data.load_dataset_from_args(args)

        estimator = ESTIMATORS[args.estimator](
            mt=mt,
            h_layer=args.h_layer,
            z_layer=args.z_layer,
        )

        for bench in args.benchmarks:
            logger.info(f"begin benchmark: {bench}")
            bench_results_dir = experiment.results_dir / bench

            results: Any
            if bench == "reconstruction":
                results = benchmarks.reconstruction(
                    dataset=dataset,
                    estimator=estimator,
                    results_dir=bench_results_dir,
                    resume=args.resume,
                )
            elif bench == "faithfulness":
                results = benchmarks.faithfulness(
                    dataset=dataset,
                    estimator=estimator,
                    results_dir=bench_results_dir,
                    resume=args.resume,
                )
            elif bench == "causality":
                bench_results_dir /= args.editor
                editor_type: type[editors.Editor] = EDITORS[args.editor]
                logger.info(f"using editing algorithm: {editor_type.__name__}")
                results = benchmarks.causality(
                    dataset=dataset,
                    estimator=estimator,
                    editor_type=editor_type,
                    # NB(evan): Results dir also needs to index on the editor type.
                    results_dir=bench_results_dir,
                    resume=args.resume,
                )
            else:
                raise ValueError(f"unknown benchmark: {bench}")

            results_file = bench_results_dir / "all.json"
            results_json = results.to_json(indent=4)
            with results_file.open("w") as handle:
                handle.write(results_json)

            metrics_json = results.metrics.to_json(indent=4)
            logger.info(metrics_json)

            metrics_file = bench_results_dir / "metrics.json"
            with metrics_file.open("w") as handle:
                handle.write(metrics_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-layer", type=int, default=13, help="layer to get h from")
    parser.add_argument("--z-layer", type=int, help="layer to get z from")
    parser.add_argument(
        "--estimator",
        "-e",
        default="j-icl-mean",
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
    parser.add_argument(
        "--editor",
        choices=EDITORS,
        default="lr-e",
        help="editor to use",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="do not recompute results already in the results dir",
    )
    data.add_data_args(parser)
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
