"""Run sweeps over different hyperparameters by relation."""
import argparse
import logging

from src import data, functional, hparams, models, sweeps
from src.utils import experiment_utils, logging_utils
from src.utils.sweep_utils import read_sweep_results, relation_from_dict

import torch

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"

    dataset = data.load_dataset().filter(
        relation_names=[args.relation_name]
    )  # full rank sweep is supposed to be run on a single relation at a time

    sweep_results_dir = f"{args.sweep_results_dir}/{args.model}"
    sweep_results = read_sweep_results(
        sweep_results_dir, relation_names=[args.relation_name]
    )
    if args.relation_name not in sweep_results:
        logger.warning(
            f"Could not find sweep results for relation {args.relation_name} -- can't calculate hyperparameters"
        )
        return

    relation_from_sweep = relation_from_dict(sweep_results[args.relation_name])
    #######################################################
    beta = 2.25  # best beta for GPT-J and GPT2-XL
    #######################################################
    hparams = relation_from_sweep.best_by_efficacy(beta=beta)
    logger.info(
        f"{args.relation_name} | h_layer: {hparams.layer} | rank: {hparams.rank.mean} +/- {hparams.rank.stderr} |>> expected lre recall: {hparams.recall.mean} +/- {hparams.recall.stderr}"
    )

    mt = models.load_model(args.model, fp16=args.fp16, device=device)
    ranks = (
        list(range(0, 320, 2))
        + list(range(320, 1024, 8))
        + list(range(1024, models.determine_hidden_size(mt), 64))
    )

    with torch.device(device):
        results = sweeps.sweep(
            mt=mt,
            dataset=dataset,
            h_layers=[hparams.layer],  # only use the best performing layer
            betas=[beta],
            ranks=ranks,
            n_trials=args.n_trials,
            n_train_samples=args.n_train_samples,
            recall_k=args.recall_k,
            batch_size=args.batch_size,
            results_dir=experiment.results_dir,
            resume=args.resume,
            subj_token_filter=args.subj_token_filter,
            consider_rank_for_recall=True,
            limit_test_samples=100,  # limit number of test samples to evalated upon
        )
        for relation in results.relations:
            log_msg = f"{relation.relation_name}"
            if len(relation.trials) < sweeps.DEFAULT_N_TRIALS:
                log_msg += f" -- not enough number of trials ({len(relation.trials)} < {sweeps.DEFAULT_N_TRIALS}) --> skipping"
                logger.info(log_msg)
                continue
            log_msg += f" (n_trials={len(relation.trials)})"
            logger.info(log_msg)

    results_file = experiment.results_dir / "results_all.json"
    results_file.parent.mkdir(exist_ok=True, parents=True)
    with results_file.open("w") as handle:
        handle.write(results.to_json(indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sweep over hyperparameters")
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    models.add_model_args(parser)
    parser.add_argument("--relation-name", type=str, help="name of the relation")
    parser.add_argument(
        "--recall-k",
        type=int,
        default=sweeps.DEFAULT_RECALL_K,
        help="compute up to recall@k",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=sweeps.DEFAULT_BATCH_SIZE,
        help="max batch size for lm",
    )
    parser.add_argument(
        "--subj-token-filter",
        type=str,
        default="all",
        choices=["all", "multi", "single"],
        help="allows filtering out samples with multiple or single subj tokens. defaults to all",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=sweeps.DEFAULT_N_TRIALS,
        help="number of trials per relation",
    )
    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=sweeps.DEFAULT_N_TRAIN_SAMPLES,
        help="number of train samples to use per trial",
    )
    parser.add_argument(
        "--sweep-results-dir",
        type=str,
        default="results/sweep-24-trials",
        help="directory to find sweep results",
    )
    args = parser.parse_args()
    logger.info(args)
    main(args)
