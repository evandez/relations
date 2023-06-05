"""Tools for running sweeps over hyperparameters."""
import logging
from typing import Any, Literal, Sequence

from src import data, editors, functional, metrics, models, operators
from src.utils import experiment_utils
from src.utils.typing import Layer, PathLike

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
    **kwargs: Any,
) -> SweepResuts:
    """Sweep over hyperparameters for faithfulness."""
    if h_layers is None:
        emb_layer: Layer = "emb"
        h_layers = [emb_layer] + list(models.determine_layers(mt))
    if betas is None:
        betas = torch.linspace(0, 1, steps=21).tolist()
    if ranks is None:
        ranks = range(0, 250, 10)
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

        # prompt_template = relation.prompt_templates[0]
        prompt_template = " {} :"  # bare prompt with colon

        relation_result = SweepRelationResults(
            relation_name=relation.name,
            trials=[],
        )
        for trial in range(n_trials):
            logger.info(f"begin trial {trial + 1}/{n_trials}")

            if len(relation.samples) <= n_train_samples:
                logger.warning(
                    f"Not enough samples ({len(relation.samples)}) to "
                    f'test for "{relation.name} with n_train_samples={n_train_samples}.'
                    f"You should fix this by adding more known samples for the relation."
                )
                continue

            # Decide which will be the train samples we will try, and which will be the
            # ICL prompt examples.
            train_relation, test_relation = relation.split(n_train_samples)
            train_samples = train_relation.samples

            logger.info(f"will train using: {[str(x) for x in train_samples]}")

            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=train_samples,
            )

            test_relation = (
                functional.filter_relation_samples_based_on_provided_fewshots(
                    mt=mt,
                    test_relation=test_relation,
                    prompt_template=icl_prompt,
                    subj_token_filter=subj_token_filter,
                )
            )

            logger.info(
                f"filtered test relation to {len(test_relation.samples)} samples"
            )

            if len(test_relation.samples) <= n_train_samples:
                logger.warning(
                    f"Not enough samples ({len(test_relation.samples)} < {n_train_samples}) to test for faithfulness and efficacy."
                )
                break  # only write results for the relations that have enough test samples for all the trials.

            test_samples = test_relation.samples
            test_subjects = [x.subject for x in test_samples]
            test_objects = [x.object for x in test_samples]
            test_targets = functional.random_edit_targets(
                test_samples
            )  # for causal tests (editing)

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

            layer_results = []

            for h_layer in h_layers:
                logger.info(f"begin layer: {h_layer}")

                estimator = operators.JacobianIclMeanEstimator(
                    mt=mt, h_layer=h_layer, **kwargs
                )

                operator = estimator(
                    relation.set(
                        samples=train_samples,
                        prompt_templates=[prompt_template],
                    )
                )
                assert operator.bias is not None
                bias = operator.bias.clone()

                test_hs = [hs_by_subj[x.subject][h_layer][None] for x in test_samples]

                # Try all betas and record recall.
                results_by_beta = []
                for beta in betas:
                    operator.bias[:] = bias * beta

                    pred_objects = []
                    for subj, h in zip(test_subjects, test_hs):
                        preds = operator(subj, h=h, k=recall_k)

                        pred = str(preds.predictions[0])
                        logger.debug(f"reading {h_layer=} {beta=} {subj=} {pred=}")

                        pred_objects.append([p.token for p in preds.predictions])

                    recall = metrics.recall(pred_objects, test_objects)
                    logger.info(f"reading finished {h_layer=} {beta=} {recall[0]=:.2f}")

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
                        )
                    )

                # Try all ranks and record efficacy.
                assert operator.weight is not None
                svd = torch.svd(operator.weight.float())
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

                        pred_objects.append([result.predicted_tokens[0].token])
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

                    [efficacy] = metrics.recall(pred_objects, targ_objects)
                    logger.info(f"editing finished: {h_layer=} {rank=} {efficacy=:.2f}")

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
                    jh_norm=torch.stack(operator.metadata["Jh"])
                    .float()
                    .view(len(train_samples), models.determine_hidden_size(mt))
                    .norm(dim=-1)
                    .mean(dim=0)
                    .item(),
                )
                train_result.summarize()
                layer_results.append(
                    SweepLayerResults(layer=h_layer, result=train_result)
                )

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
