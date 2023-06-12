import argparse
import json
import logging
import os
from dataclasses import dataclass

from src import data, editors, functional, metrics, models, operators
from src.data import RelationSample
from src.utils import experiment_utils, logging_utils
from src.utils.sweep_utils import (
    SweepLayerSummary,
    read_sweep_results,
    relation_from_dict,
)

import torch
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialResult(DataClassJsonMixin):
    train_samples: list[RelationSample]
    test_samples: list[RelationSample]
    W_norm: float
    bias_norm: float
    recall: metrics.AggregateMetric
    efficacy: metrics.AggregateMetric


@dataclass(frozen=True)
class AllTrialResults(DataClassJsonMixin):
    relation_name: str
    trials: list[TrialResult]


def perform_trial(
    mt: models.ModelAndTokenizer,
    relation: data.Relation,
    hparams: SweepLayerSummary,
    n_train_samples: int = 5,
    recall_k: int = 3,
) -> TrialResult:
    prompt_template = " {} :"  # bare prompt with colon
    relation = relation.set(prompt_templates=[prompt_template])
    train_relation, test_relation = relation.split(n_train_samples)
    train_samples = train_relation.samples

    logger.info(f"will train using: {[str(x) for x in train_samples]}")
    logger.info(f"{prompt_template=}")

    icl_prompt = functional.make_prompt(
        mt=mt,
        prompt_template=prompt_template,
        subject="{}",
        examples=train_samples,
    )

    test_relation = functional.filter_relation_samples_based_on_provided_fewshots(
        mt=mt,
        test_relation=test_relation,
        prompt_template=icl_prompt,
    )

    logger.info(f"filtered test relation to {len(test_relation.samples)} samples")
    test_samples = test_relation.samples
    test_subjects = [x.subject for x in test_samples]
    test_objects = [x.object for x in test_samples]
    test_targets = functional.random_edit_targets(
        test_samples
    )  # for causal tests (editing)

    estimator = operators.JacobianIclMeanEstimator(
        mt=mt, h_layer=hparams.layer, beta=None  # equivalent to beta = 1
    )
    operator = estimator(train_relation)
    assert operator.weight is not None
    assert operator.bias is not None

    # calculate faithfulness
    pred_objects = []
    test_objects = []
    for sample in test_samples:
        test_objects.append(sample.object)
        preds = operator(sample.subject, k=recall_k)
        pred = str(preds.predictions[0])
        logger.debug(f"{sample.subject=} -> {sample.object=} | {pred=}")
        pred_objects.append([p.token for p in preds.predictions])

    recall = metrics.recall(pred_objects, test_objects)
    logger.info(f"reading finished {recall[0]=:.2f}")

    # calculate efficacy
    rank = int(hparams.rank.mean)
    svd = torch.svd(operator.weight.float())
    editor = editors.LowRankPInvEditor(
        lre=operator,
        rank=rank,
        n_samples=1,
        n_new_tokens=1,
        n_top_tokens=recall_k,
        svd=svd,
    )

    pred_objects = []
    targ_objects = []
    for sample in test_samples:
        target = test_targets.get(sample)
        assert target is not None
        if target is None:
            logger.debug(f"skipping {sample.subject} -> {sample.object}")
            continue
        result = editor(
            sample.subject,
            target.subject,
        )
        pred = str(result.predicted_tokens[0])
        logger.debug(
            f"editing: {sample.subject=} | {target.subject=} -> {target.object=} |>> {pred=}"
        )
        pred_objects.append([p.token for p in result.predicted_tokens])
        targ_objects.append(target.object)
    efficacy = metrics.recall(pred_objects, targ_objects)
    logger.info(f"editing finished: {efficacy=}")

    return TrialResult(
        train_samples=train_samples,
        test_samples=test_samples,
        W_norm=operator.weight.norm().item(),
        bias_norm=operator.bias.norm().item(),
        recall=metrics.AggregateMetric.aggregate(recall),
        efficacy=metrics.AggregateMetric.aggregate(efficacy),
    )


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)
    logger.info(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, fp16=args.fp16, device=device)
    logger.info(
        f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}"
    )
    relation_name = args.rel_name
    save_dir = f'{args.save_dir}/{args.model}/{relation_name.replace(" ", "_")}/{args.n_training}'
    os.makedirs(save_dir, exist_ok=True)

    sweep_results_dir = f"{args.sweep_results_dir}/{args.model}"
    sweep_results = read_sweep_results(
        sweep_dir=sweep_results_dir, filter_relations=[relation_name]
    )

    relation_results = relation_from_dict(sweep_results[relation_name])
    efficacy_hparams = relation_results.best_by_efficacy()
    logger.info(
        f"""Best efficacy hparams: 
            layer={efficacy_hparams.layer}
            beta={efficacy_hparams.beta.mean}, {efficacy_hparams.beta.values}
            rank={efficacy_hparams.rank.mean}, {efficacy_hparams.rank.values}
            -------------------------------------------------------------------
            Efficacy={efficacy_hparams.efficacy.mean}, {efficacy_hparams.efficacy.values}
            Recall={efficacy_hparams.recall.mean}, {efficacy_hparams.recall.values}
        """
    )

    dataset = data.load_dataset()
    relation = dataset.filter(
        relation_names=[args.rel_name],
    )[0]
    results_file_name = f"seed_{args.seed}.json"
    all_trial_results = AllTrialResults(relation_name=relation.name, trials=[])
    for trial in range(args.n_trials):
        trial_result = perform_trial(
            mt=mt,
            relation=relation,
            hparams=efficacy_hparams,
            n_train_samples=args.n_training,
            recall_k=args.recall_k,
        )
        all_trial_results.trials.append(trial_result)
        # save results after each trial
        with open(f"{save_dir}/{results_file_name}", "w") as f:
            json.dump(all_trial_results.to_dict(), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run faithfulness baselines on optimum hparams"
    )

    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--sweep-results-dir",
        type=str,
        default="results/sweep-24-trials",
        help="directory to find sweep results",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/explain_beta",
        help="path to save results",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="number of trials",
    )

    parser.add_argument(
        "--n-training",
        type=int,
        default=functional.DEFAULT_N_ICL_LM,
        help="number of training samples",
    )

    parser.add_argument(
        "--recall-k",
        type=int,
        default=3,
        help="Store results upto recall@k",
    )

    parser.add_argument("--rel-name", type=str, help="filter by relation name")

    args = parser.parse_args()
    main(args)
