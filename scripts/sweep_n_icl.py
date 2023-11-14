"""Run sweeps over different hyperparameters by relation."""
import argparse
import logging
import os

from src import data, models, sweeps
from src.hparams import RelationHParams
from src.utils import experiment_utils, logging_utils

import torch

# from src.utils.sweep_utils import read_sweep_results, relation_from_dict


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"

    dataset = data.load_dataset().filter(
        relation_names=[args.relation_name]
    )  # full rank sweep is supposed to be run on a single relation at a time

    # sweep_results_dir = f"{args.sweep_results_dir}/{args.model}"
    # sweep_results = read_sweep_results(
    #     sweep_results_dir, relation_names=[args.relation_name]
    # )
    # if args.relation_name not in sweep_results:
    #     logger.warning(
    #         f"Could not find sweep results for relation {args.relation_name} -- can't calculate hyperparameters"
    #     )
    #     return

    # relation_from_sweep = relation_from_dict(sweep_results[args.relation_name])
    # #######################################################
    # beta = 2.25  # best beta for GPT-J and GPT2-XL
    # #######################################################
    # hparams = relation_from_sweep.best_by_efficacy(beta=beta)

    hparams = RelationHParams.from_relation(
        model=args.model,
        relation=args.relation_name,
    )

    logger.info(
        f"{args.relation_name} | h_layer={hparams.h_layer} | rank={hparams.rank} | beta={hparams.beta}"
    )

    mt = models.load_model(args.model, fp16=args.fp16, device=device)

    for n_train in range(args.n_train_from, args.n_train_to + 1):
        logger.info("###############################################")
        logger.info(f"n_train={n_train}")
        logger.info("###############################################")
        with torch.device(device):
            results = sweeps.sweep(
                mt=mt,
                dataset=dataset,
                h_layers=[hparams.h_layer],  # only use the best performing layer
                betas=[hparams.beta],
                ranks=[hparams.rank],
                n_trials=args.n_trials,
                n_train_samples=n_train,
                recall_k=args.recall_k,
                batch_size=args.batch_size,
                results_dir=os.path.join(experiment.results_dir, str(n_train)),
                resume=args.resume,
                subj_token_filter=args.subj_token_filter,
                consider_rank_for_recall=False,
            )
            for relation in results.relations:
                log_msg = f"{relation.relation_name}"
                if len(relation.trials) < sweeps.DEFAULT_N_TRIALS:
                    log_msg += f" -- not enough number of trials ({len(relation.trials)} < {sweeps.DEFAULT_N_TRIALS}) --> skipping"
                    logger.info(log_msg)
                    continue
                log_msg += f" (n_trials={len(relation.trials)})"
                logger.info(log_msg)


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
        "--sweep-results-dir",
        type=str,
        default="results/num_train",
        help="directory to find sweep results",
    )
    parser.add_argument(
        "--n-train-from",
        type=int,
        default=3,
        help="number of train samples to start from",
    )
    parser.add_argument(
        "--n-train-to",
        type=int,
        default=12,
        help="number of train samples to end at",
    )
    args = parser.parse_args()
    logger.info(args)
    main(args)
