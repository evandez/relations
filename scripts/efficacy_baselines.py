import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

from src import data, editors, functional, metrics, models, operators, sweeps
from src.data import RelationSample
from src.sweeps import (
    EfficacyTestPair,
    SweepBetaResults,
    SweepLayerResults,
    SweepRankResults,
    SweepRelationResults,
    SweepTrainResults,
    SweepTrialResults,
)
from src.utils import dataclasses_utils, experiment_utils, logging_utils
from src.utils.typing import Layer

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def parse_results(sweep_result: dict) -> SweepRelationResults:
    relation_results = SweepRelationResults(
        relation_name=sweep_result["relation_name"], trials=[]
    )

    for trial in sweep_result["trials"]:
        trial_results = SweepTrialResults(
            prompt_template=trial["prompt_template"],
            train_samples=[RelationSample.from_dict(s) for s in trial["train_samples"]],
            layers=[],
            n_test_samples=trial["n_test_samples"],
        )
        for layer in trial["layers"]:
            train_results = SweepTrainResults(
                samples=[
                    RelationSample.from_dict(s) for s in layer["result"]["samples"]
                ],
                betas=[],
                ranks=[],
                jh_norm=layer["result"]["jh_norm"],
            )
            for beta in layer["result"]["betas"]:
                beta_results = SweepBetaResults(
                    beta=beta["beta"],
                    recall=beta["recall"],
                    faithfulness_successes=[
                        RelationSample.from_dict(s)
                        for s in beta["faithfulness_successes"]
                    ],
                )
                train_results.betas.append(beta_results)

            for rank in layer["result"]["ranks"]:
                rank_results = SweepRankResults(
                    rank=rank["rank"],
                    efficacy=rank["efficacy"],
                    efficacy_successes=[
                        EfficacyTestPair(
                            source=RelationSample.from_dict(s["source"]),
                            target=RelationSample.from_dict(s["target"]),
                        )
                        for s in rank["efficacy_successes"]
                    ],
                )
                train_results.ranks.append(rank_results)

            layer_results = SweepLayerResults(
                layer=layer["layer"], result=train_results
            )

            trial_results.layers.append(layer_results)
        relation_results.trials.append(trial_results)
    return relation_results


def read_sweep_results(sweep_path: str) -> dict:
    sweep_results = {}

    for relation_folder in os.listdir(sweep_path):
        cur_sweep = f"{sweep_path}/{relation_folder}"
        if "results_all.json" not in os.listdir(cur_sweep):
            continue
        with open(f"{cur_sweep}/results_all.json") as f:
            res = json.load(f)["relations"]
            if len(res) == 0:
                continue
            res = res[0]
            sweep_results[res["relation_name"]] = res
    return sweep_results


def filter_not_in_train_samples(
    sample: RelationSample, train_samples: list[RelationSample]
) -> bool:
    for train_sample in train_samples:
        if (
            sample.subject == train_sample.subject
            and sample.object == train_sample.object
        ):
            return False
    return True


BASELINE_EDITOR_TYPES = {
    "hidden_baseline": editors.HiddenBaselineEditor,
    "embed_baseline": editors.EmbedBaselineEditor,
    "low_rank_pinv": editors.LowRankPInvEditor,
    "hidden_baseline_z": editors.HiddenBaselineEditor_Obj,
}


@dataclass(frozen=True)
class EfficacyBaselineLayerResult(DataClassJsonMixin):
    layer: Layer
    efficacy: float
    rank: int
    results: dict[str, float]


@dataclass(frozen=True)
class EfficacyBaselineTrialResult(DataClassJsonMixin):
    train_samples: list[RelationSample]
    prompt_template: str
    efficacy_test_pairs: list[EfficacyTestPair]
    layerwise_baseline_results: list[EfficacyBaselineLayerResult]


@dataclass(frozen=True)
class EfficacyBaselineRelationResult(DataClassJsonMixin):
    relation_name: str
    trials: list[EfficacyBaselineTrialResult]


@dataclass(frozen=True)
class EfficacyBaselineResults(DataClassJsonMixin):
    relations: list[EfficacyBaselineRelationResult]


def run_edit_baselines(
    model_name: Literal["gptj", "gpt2-xl", "llama-13b"] = "gptj",
    sweep_results_dir: str = "results/sweep/",
    save_dir: str = "results/efficacy_baselines/",
    device: str | None = None,
    batch_size: int = sweeps.DEFAULT_BATCH_SIZE,
    experiment_name: str | None = None,
    rel_names: list[str] | None = None,
) -> None:
    save_dir = f"{save_dir}/{model_name}"
    if experiment_name is not None:
        save_dir = f"{save_dir}/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    device = device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(name=model_name, device=device)
    sweep_results_dir = f"{sweep_results_dir}/{model_name}"
    sweep_results = read_sweep_results(sweep_results_dir)

    logger.info("found %d relations", len(sweep_results))
    logger.info(json.dumps(list(sweep_results.keys()), indent=4))

    dataset = data.load_dataset()

    all_results: list[EfficacyBaselineRelationResult] = []
    for relation_name, sweep_result in tqdm(sweep_results.items()):
        if rel_names is not None and relation_name not in rel_names:
            logger.info("skipping %s", relation_name)
            continue
        logger.info("relation: %s", relation_name)
        relation_results = parse_results(sweep_result)
        relation = dataset.filter(relation_names=[relation_results.relation_name])[0]
        by_layer = relation_results.by_layer()

        baseline_trial_results: list[EfficacyBaselineTrialResult] = []
        for n_trial in range(len(relation_results.trials)):
            logger.info(f"trial: {n_trial+1}/{len(relation_results.trials)}")
            trial_results = relation_results.trials[n_trial]
            prompt_template = trial_results.prompt_template
            train_samples = trial_results.train_samples
            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=train_samples,
            )
            test_samples = [
                sample
                for sample in relation.samples
                if filter_not_in_train_samples(sample, train_samples)
            ]
            test_relation = relation.set(samples=test_samples)

            test_relation = (
                functional.filter_relation_samples_based_on_provided_fewshots(
                    mt=mt,
                    test_relation=test_relation,
                    prompt_template=icl_prompt,
                    subj_token_filter="all",
                )
            )
            logger.info(
                f"filtered test relation to {len(test_relation.samples)} samples"
            )

            if len(test_relation.samples) <= len(train_samples):
                logger.warning(
                    f"Not enough samples ( < {len(train_samples)}) to test for faithfulness and efficacy."
                )
                break  # only consider relations that have enough number of known test samples

            h_layers = [layer.layer for layer in trial_results.layers]

            logger.info("precomputing test hs and zs...")
            hs_by_subj, zs_by_subj = functional.compute_hs_and_zs(
                mt=mt,
                prompt_template=prompt_template,
                subjects=[x.subject for x in test_samples],
                h_layer=h_layers,
                z_layer=-1,
                batch_size=batch_size,
                examples=train_samples,
            )

            layerwise_baseline_results: list[EfficacyBaselineLayerResult] = []
            for layer in trial_results.layers:
                layer_no = layer.layer
                if layer_no not in h_layers:
                    continue
                efficacy = by_layer[layer.layer].efficacy
                rank = int(np.floor(by_layer[layer.layer].rank.mean))
                logger.info(
                    f"layer: {layer_no}, efficacy = {efficacy.mean} +/- {efficacy.stderr}, {rank=}"
                )

                estimator = operators.JacobianIclMeanEstimator(
                    mt=mt,
                    h_layer=layer_no,
                )
                operator = estimator(
                    relation.set(
                        samples=train_samples,
                        prompt_templates=[prompt_template],
                    )
                )
                assert operator.weight is not None
                svd = torch.svd(operator.weight.float())
                test_targets = functional.random_edit_targets(test_samples)

                baseline_results: dict[str, float] = {}

                for editor_type in BASELINE_EDITOR_TYPES:
                    editor_class = BASELINE_EDITOR_TYPES[editor_type]
                    editor: editors.Editor = (
                        dataclasses_utils.create_with_optional_kwargs(
                            editor_class,
                            h_layer=layer_no,
                            rank=rank,
                            lre=operator,
                            svd=svd,
                            prompt_template=operator.prompt_template,
                            mt=mt,
                            n_samples=1,
                            n_new_tokens=1,
                        )
                    )

                    pred_objects = []
                    target_objects = []
                    for original in test_samples:
                        target = test_targets.get(original)
                        if target is None:
                            logger.debug(f"cannot edit {target}, skipping")
                            continue
                        target_objects.append(target.object)

                        z_original = zs_by_subj[original.subject]
                        z_target = zs_by_subj[target.subject]
                        h_original = hs_by_subj[original.subject][layer_no]
                        h_target = hs_by_subj[target.subject][layer_no]

                        if editor.expects() == "object":
                            result = dataclasses_utils.call_with_optional_kwargs(
                                editor.__call__,
                                subject=original.subject,
                                target=target.object,
                                z_original=z_original,
                            )
                        else:
                            assert editor.expects() == "subject"
                            result = dataclasses_utils.call_with_optional_kwargs(
                                editor.__call__,
                                subject=original.subject,
                                target=target.subject,
                                z_original=z_original,
                                z_target=z_target,
                            )
                        pred_objects.append([result.predicted_tokens[0].token])

                        pred = str(result.predicted_tokens[0])
                        logger.debug(
                            f"editing: {original.subject=} {target.subject=} {target.object=} {pred=}"
                        )

                    [baseline_efficacy] = metrics.recall(pred_objects, target_objects)
                    logger.info(
                        f"editing finished: {layer_no=} {rank=} {editor_type}={baseline_efficacy:.2f}"
                    )
                    logger.debug("--------------------------------------------------")
                    baseline_results[editor_type] = baseline_efficacy

                layerwise_baseline_results.append(
                    EfficacyBaselineLayerResult(
                        layer=layer_no,
                        efficacy=efficacy.mean,
                        rank=rank,
                        results=baseline_results,
                    )
                )
            baseline_trial_results.append(
                EfficacyBaselineTrialResult(
                    train_samples=train_samples,
                    prompt_template=prompt_template,
                    efficacy_test_pairs=[
                        EfficacyTestPair(source=key, target=value)
                        for key, value in test_targets.items()
                    ],
                    layerwise_baseline_results=layerwise_baseline_results,
                )
            )

        efficacy_baseline_relation_result = EfficacyBaselineRelationResult(
            relation_name=relation_name,
            trials=baseline_trial_results,
        )
        experiment_utils.save_results_file(
            results_dir=save_dir,
            name=relation_results.relation_name,
            results=efficacy_baseline_relation_result,
        )

        all_results.append(efficacy_baseline_relation_result)

    efficacy_baseline_results = EfficacyBaselineResults(relations=all_results)
    results_file = f"{save_dir}/results_all.json"
    logger.info(f"saving all results to {results_file}")
    with open(results_file, "w") as handle:
        handle.write(efficacy_baseline_results.to_json(indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="calculate layerwise efficacy baselines"
    )
    logging_utils.add_logging_args(parser)
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=sweeps.DEFAULT_BATCH_SIZE,
        help="max batch size for lm",
    )

    parser.add_argument(
        "--sweep-results-dir",
        type=str,
        default="results/sweep/",
        help="directory to find sweep results",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/efficacy_baselines/",
        help="directory to find sweep results",
    )
    parser.add_argument(
        "--rel-names", "-r", nargs="+", type=str, help="filter by relation name"
    )

    args = parser.parse_args()

    logging_utils.configure(args)

    logger.info(args)

    run_edit_baselines(
        model_name=args.model,
        sweep_results_dir=args.sweep_results_dir,
        save_dir=args.save_dir,
        device=args.device,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        rel_names=args.rel_names,
    )
