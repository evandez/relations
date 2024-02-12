"""Tools for running sweeps over hyperparameters."""

import logging
import os
import random
from typing import Any, Literal, Optional, Sequence

from src import data, editors, functional, metrics, models, operators
from src.functional import low_rank_approx
from src.operators import LinearRelationOperator
from src.utils import experiment_utils
from src.utils.typing import Layer, PathLike

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_RECALL_K = 3
DEFAULT_N_TRIALS = 3
DEFAULT_N_TRAIN_SAMPLES = 5
DEFAULT_BATCH_SIZE = 64

from src.utils.sweep_utils import (
    EfficacyTestPair,
    SweepBetaResults,
    SweepLayerResults,
    SweepRankResults,
    SweepRelationResults,
    SweepResuts,
    SweepTrainResults,
    SweepTrialResults,
)


def add_npz_extension(path: str):
    return path if path.endswith(".npz") else path + ".npz"


def load_o1_approxes(path: str, sample_subjects: Optional[list[str]] = None):
    approxes = []
    to_load = sample_subjects if sample_subjects is not None else os.listdir(path)
    for cached_file in to_load:
        file_path = add_npz_extension(os.path.join(path, cached_file))
        approx = functional.load_cached_linear_operator(file_path=file_path)
        approxes.append(approx)
    return approxes


def sweep(
    *,
    mt: models.ModelAndTokenizer,
    dataset: data.RelationDataset,
    h_layers: Sequence[Layer] | None = None,
    betas: Sequence[float] | None = None,
    ranks: Sequence[int] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    n_train_samples: int = DEFAULT_N_TRAIN_SAMPLES,
    recall_k: int = DEFAULT_RECALL_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    results_dir: PathLike | None = None,
    resume: bool = False,
    subj_token_filter: Literal["all", "single", "multi"] = "all",
    consider_rank_for_recall: bool = False,
    limit_test_samples: int | None = None,
    use_bare_prompt: bool = False,
    prompt_template: str | None = None,
    o1_approxes_path: str | None = None,
    **kwargs: Any,
) -> SweepResuts:
    """Sweep over hyperparameters for faithfulness."""
    if h_layers is None:
        emb_layer: Layer = "emb"
        h_layers = [emb_layer] + list(models.determine_layers(mt))
        # Currently hardcoded for mamba.
        # TODO: make this more general depending on o1_approxes_path
        if mt.is_mamba:
            h_layers = list(np.arange(0, 64, 2))
    if betas is None:
        if mt.is_mamba or mt.name == "llama":
            beta_upper_limit = 12.0
        else:
            beta_upper_limit = 5.0
        betas = torch.linspace(0, beta_upper_limit, steps=25).tolist()
    if ranks is None:
        if mt.is_mamba or mt.name == "llama":
            low_rank_upper_limit = 512
        else:
            low_rank_upper_limit = 320
        ranks = range(0, low_rank_upper_limit, 8)
        # ranks = range(0, models.determine_hidden_size(mt), 64)
    limit_test_samples = (
        200
        if (mt.name == "llama" and limit_test_samples is None)
        else limit_test_samples
    )

    logger.info("begin sweeping faithfulness")

    relation_results = []
    for ri, relation in enumerate(dataset.relations):
        logger.info(
            f'begin relation "{relation.name}" ({ri + 1}/{len(dataset.relations)})'
        )

        relation_result = experiment_utils.load_results_file(
            results_dir=results_dir,
            results_type=SweepRelationResults,
            name=relation.name,
            resume=resume,
        )
        if relation_result is not None:
            logger.info(f"loaded previous results for {relation.name}")
            relation_results.append(relation_result)
            continue

        if use_bare_prompt:
            prompt_template = " {}"  # bare prompt
        else:
            prompt_template = (
                relation.prompt_templates[0]
                if prompt_template is None
                else prompt_template
            )
            # prompt_template = " {} :"  # bare prompt with colon

        relation_result = SweepRelationResults(
            relation_name=relation.name,
            trials=[],
        )
        for trial in range(n_trials):
            logger.info(f"begin trial {trial + 1}/{n_trials}")
            exit_trial = False

            if len(relation.samples) <= n_train_samples:
                logger.warning(
                    f"Not enough samples ({len(relation.samples)}) to "
                    f'test for "{relation.name} with n_train_samples={n_train_samples}.'
                    f"You should fix this by adding more known samples for the relation."
                )
                continue

            # Decide which will be the train samples we will try, and which will be the
            # ICL prompt examples.

            if o1_approxes_path is not None:
                logger.info(
                    f"attempting to load cached approxes from {o1_approxes_path}"
                )
                relation_approxes_path = os.path.join(
                    o1_approxes_path,
                    mt.name,
                    relation.name.lower().replace(" ", "_"),
                )
                layer_approxes_path = os.path.join(
                    relation_approxes_path, str(h_layers[0])
                )
                all_cached_files = list(os.listdir(layer_approxes_path))
                train_subj_files = random.sample(all_cached_files, n_train_samples)
                train_approxes = load_o1_approxes(
                    path=layer_approxes_path, sample_subjects=train_subj_files
                )
                train_samples = [
                    data.RelationSample.from_dict(approx.metadata["sample"])
                    for approx in train_approxes
                ]
                train_relation = relation.set(samples=train_samples)
                test_relation = relation.set(
                    samples=list(set(relation.samples) - set(train_relation.samples))
                )

            else:
                train_relation, test_relation = relation.split(n_train_samples)
                train_samples = train_relation.samples

            logger.info(f"training samples: {[str(x) for x in train_samples]}")

            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=train_samples,
            )

            try:
                test_relation = (
                    functional.filter_relation_samples_based_on_provided_fewshots(
                        mt=mt,
                        test_relation=test_relation,
                        prompt_template=icl_prompt,
                        subj_token_filter=subj_token_filter,
                    )
                )

                # Precompute all the hs to speed things up.
                hs_by_subj, zs_by_subj = functional.compute_hs_and_zs(
                    mt=mt,
                    prompt_template=prompt_template,
                    subjects=[x.subject for x in test_relation.samples],
                    h_layer=h_layers,
                    z_layer=-1,
                    batch_size=batch_size,
                    examples=train_samples,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(
                        f"OOM while filtering on trial {trial + 1}/{n_trials} for {relation.name}, skipping"
                    )
                    exit_trial = True
                    continue
                else:
                    raise e

            logger.info(
                f"filtered test relation to {len(test_relation.samples)} samples"
            )

            if len(test_relation.samples) <= n_train_samples:
                logger.warning(
                    f"Not enough samples ({len(test_relation.samples)} < {n_train_samples}) to test for faithfulness and efficacy."
                )
                # break  # only write results for the relations that have enough test samples for all the trials.
                continue  # only skip the trial that doesn't have enough test samples. continue otherwise

            if limit_test_samples is not None:
                logger.info(f"limiting test samples to {limit_test_samples}")
                test_relation = test_relation.set(
                    samples=test_relation.samples[:limit_test_samples]
                )

            test_samples = test_relation.samples
            test_subjects = [x.subject for x in test_samples]
            test_objects = [x.object for x in test_samples]
            test_targets = functional.random_edit_targets(test_samples)

            layer_results = []
            for h_layer in h_layers:
                logger.info(f"begin layer: {h_layer}")

                # precompute the hs for the test samples
                test_hs = [hs_by_subj[x.subject][h_layer][None] for x in test_samples]

                if o1_approxes_path is not None:
                    layer_approxes_path = os.path.join(
                        relation_approxes_path, str(h_layer)
                    )
                    layer_approxes_path = os.path.join(
                        relation_approxes_path, str(h_layer)
                    )
                    train_approxes = load_o1_approxes(
                        path=layer_approxes_path, sample_subjects=train_subj_files
                    )

                    weight = torch.stack(
                        [approx.weight for approx in train_approxes]
                    ).mean(dim=0)
                    bias = torch.stack([approx.bias for approx in train_approxes]).mean(
                        dim=0
                    )
                    prompt_template_icl = functional.make_prompt(
                        mt=mt,
                        prompt_template=prompt_template,
                        subject="{}",
                        examples=train_samples,
                    )

                    operator = LinearRelationOperator(
                        mt=mt,
                        weight=weight,
                        bias=bias,
                        h_layer=train_approxes[0].h_layer,
                        z_layer=train_approxes[0].z_layer,
                        prompt_template=prompt_template_icl,
                        metadata={
                            "Jh": [
                                (approx.weight @ approx.h).detach().cpu()
                                for approx in train_approxes
                            ],
                            "|w|": [
                                approx.weight.norm().item() for approx in train_approxes
                            ],
                            "|b|": [
                                approx.bias.norm().item() for approx in train_approxes
                            ],
                        },
                    )

                else:
                    estimator = operators.JacobianIclMeanEstimator(
                        mt=mt, h_layer=h_layer, **kwargs
                    )

                    # Estimate for imaginary/mythical subjects.
                    # estimator = operators.JacobianIclMeanEstimator_Imaginary(
                    #     mt=mt,
                    #     h_layer=h_layer,
                    #     ##############################################################
                    #     interpolate_on=4,  # interpolate on 5 real subjects
                    #     magnitude_h=65.0,  # magnitude of h)
                    #     ##############################################################
                    #     n_trials=len(train_samples),
                    #     **kwargs,
                    # )
                    try:
                        operator = estimator(
                            relation.set(
                                samples=train_samples,
                                prompt_templates=[prompt_template],
                            )
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(
                                f"OOM while LRE calculation on trial {trial + 1}/{n_trials} for {relation.name}, skipping"
                            )
                            exit_trial = True
                            break
                        else:
                            raise e

                assert operator.weight is not None
                weight = operator.weight.clone()
                svd = torch.svd(weight.float())

                # Try all betas and record recall.
                results_by_beta = []
                recall_ranks = [models.determine_hidden_size(mt)] if consider_rank_for_recall is False else ranks  # type: ignore

                for rank in recall_ranks:
                    for beta in betas:
                        operator.weight[:] = (
                            low_rank_approx(matrix=weight, rank=rank, svd=svd) * beta
                            if consider_rank_for_recall is True
                            else weight * beta
                        )

                        pred_objects = []
                        for subj, h in zip(test_subjects, test_hs):
                            preds = operator(subj, h=h, k=recall_k)

                            pred = str(preds.predictions[0])
                            logger.debug(f"reading {h_layer=} {beta=} {subj=} {pred=}")

                            pred_objects.append([p.token for p in preds.predictions])

                        recall = metrics.recall(pred_objects, test_objects)
                        cur_hparams_info = f"{h_layer=} {beta=}"
                        cur_hparams_info += f" {rank=}" if rank is not None else ""
                        logger.info("-" * 80)
                        logger.info(
                            f"reading finished, {cur_hparams_info} <> {recall=}"
                        )
                        logger.info("-" * 80)

                        faithfulness_successes = []
                        for prediction, sample in zip(pred_objects, test_samples):
                            if functional.is_nontrivial_prefix(
                                prediction=prediction[0], target=sample.object
                            ):
                                faithfulness_successes.append(sample)

                        results_by_beta.append(
                            SweepBetaResults(
                                beta=beta,
                                recall=recall,
                                faithfulness_successes=faithfulness_successes,
                                rank=rank,
                            )
                        )

                # Try all ranks and record efficacy.
                assert operator.weight is not None
                operator.weight[:] = weight  # reset to original weight

                results_by_rank = []
                for rank in ranks:
                    editor = editors.LowRankPInvEditor(
                        lre=operator,
                        rank=rank,
                        n_samples=1,
                        n_new_tokens=1,
                        svd=svd,
                    )

                    pred_objects = []
                    targ_objects = []
                    efficacy_successes = []
                    for sample in test_samples:
                        target = test_targets.get(sample)
                        assert target is not None
                        if target is None:
                            logger.debug(f"cannot edit {target}, skipping")
                            continue

                        z_original = zs_by_subj[sample.subject]
                        z_target = zs_by_subj[target.subject]
                        result = editor(
                            sample.subject,
                            target.subject,
                            z_original=z_original,
                            z_target=z_target,
                        )

                        pred = str(result.predicted_tokens[0])
                        logger.debug(
                            f"editing: {h_layer=} {rank=} {sample.subject=} | {target.subject=} -> {target.object=} |>> {pred=}"
                        )

                        pred_objects.append([p.token for p in result.predicted_tokens])
                        targ_objects.append(target.object)
                        if functional.is_nontrivial_prefix(
                            prediction=result.predicted_tokens[0].token,
                            target=target.object,
                        ):
                            efficacy_successes.append(
                                EfficacyTestPair(
                                    source=sample,
                                    target=target,
                                )
                            )

                    efficacy = metrics.recall(pred_objects, targ_objects)

                    logger.info("-" * 80)
                    logger.info(f"editing finished: {h_layer=} {rank=} {efficacy=}")
                    logger.info("-" * 80)

                    results_by_rank.append(
                        SweepRankResults(
                            rank=rank,
                            efficacy=efficacy,
                            efficacy_successes=efficacy_successes,
                        )
                    )

                train_result = SweepTrainResults(
                    samples=train_samples,
                    betas=results_by_beta,
                    ranks=results_by_rank,
                    lre_stats={
                        "|jh|": torch.stack(operator.metadata["Jh"])
                        .float()
                        .view(len(train_samples), models.determine_hidden_size(mt))
                        .norm(dim=-1)
                        .mean(dim=0)
                        .item(),
                        "|weight|": operator.metadata["|w|"],
                        "|bias|": operator.metadata["|b|"],
                    },
                )
                train_result.summarize()
                layer_results.append(
                    SweepLayerResults(layer=h_layer, result=train_result)
                )

            if exit_trial:
                continue

            relation_result.trials.append(
                SweepTrialResults(
                    prompt_template=prompt_template,
                    train_samples=train_samples,
                    layers=layer_results,
                    n_test_samples=len(test_samples),
                    efficacy_trials=[
                        EfficacyTestPair(
                            source=sample,
                            target=target,
                        )
                        for sample, target in test_targets.items()
                    ],
                )
            )
            # Save results after each of the trials.
            experiment_utils.save_results_file(
                results_dir=results_dir,
                results=relation_result,
                name=relation.name,
            )
        relation_result.summarize()
        relation_results.append(relation_result)
    return SweepResuts(relation_results)
