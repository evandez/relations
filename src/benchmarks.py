import logging
import random
from collections import defaultdict
from dataclasses import dataclass

from src import data, functional, metrics, operators

import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkMetrics(DataClassJsonMixin):
    frac_correct: float
    frac_dist_subj: float
    frac_dist_rel: float


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkResults(DataClassJsonMixin):
    metrics: ReconstructionBenchmarkMetrics


def reconstruction(
    estimator: operators.LinearRelationEstimator,
    dataset: data.RelationDataset,
    n_trials: int = 3,
    n_train: int = 3,
    n_random_distractors: int = 3,
    desc: str | None = None,
) -> ReconstructionBenchmarkResults:
    if desc is None:
        desc = "reconstruction"
    mt = estimator.mt

    everything = sorted(
        {
            (relation.name, prompt_template, subject)
            for relation in dataset.relations
            for prompt_template in relation.prompt_templates
            for subject in relation.domain
        }
    )

    counts: dict[int, int] = defaultdict(int)
    for relation in dataset.relations:
        for _ in range(n_trials):
            prompt_template = random.choice(relation.prompt_templates)
            train, test = relation.set(prompt_templates=[prompt_template]).split(
                n_train
            )
            operator = estimator(train)

            for sample in test.samples:
                subject = sample.subject

                z_true = functional.compute_hidden_states(
                    mt=estimator.mt,
                    layers=[operator.z_layer],
                    prompt=prompt_template.format(subject),
                ).hiddens[0][0, -1]
                z_pred = operator(subject).z

                # Hard distractor 1: same subject, different relation
                matches = [
                    (r, p, s)
                    for r, p, s in everything
                    if r != relation.name and s == subject
                ]
                if not matches:
                    logger.debug(
                        f'skipped "{relation.name}"/{subject} '
                        "because no other relations have this subject"
                    )
                    continue
                (_, other_prompt_template, _) = random.choice(matches)
                z_hard_subj_prompt = functional.make_prompt(
                    prompt_template=other_prompt_template,
                    subject=subject,
                    mt=mt,
                )
                z_hard_subj = functional.compute_hidden_states(
                    mt=estimator.mt,
                    layers=[operator.z_layer],
                    prompt=z_hard_subj_prompt,
                ).hiddens[0][0, -1]

                # Distractor 2: same relation, different subject
                matches = [
                    (r, p, s)
                    for r, p, s in everything
                    if r == relation.name and p == prompt_template and s != subject
                ]
                if not matches:
                    logger.debug(
                        f'skipped "{relation.name}"/{subject} '
                        "because no other subjects have this relation"
                    )
                    continue
                (_, _, other_subject) = random.choice(matches)
                z_hard_rel_prompt = functional.make_prompt(
                    prompt_template=prompt_template,
                    subject=other_subject,
                    mt=mt,
                )
                z_hard_rel = functional.compute_hidden_states(
                    mt=estimator.mt,
                    layers=[operator.z_layer],
                    prompt=z_hard_rel_prompt,
                ).hiddens[0][0, -1]

                # Distractor 3 and 4: chosen at random!
                matches = [
                    (r, p, s)
                    for r, p, s in everything
                    if r != relation.name and s != subject
                ]
                if not matches:
                    logger.debug(
                        f'skipped "{relation.name}"/{subject} '
                        "because no other relations or subjects"
                    )
                    continue

                z_rands = []
                for _, other_prompt_template, other_subject in random.sample(matches, k=n_random_distractors):
                    z_rand_prompt = functional.make_prompt(
                        prompt_template=other_prompt_template,
                        subject=other_subject,
                        mt=mt,
                    )
                    z_rand = functional.compute_hidden_states(
                        mt=estimator.mt,
                        layers=[operator.z_layer],
                        prompt=z_rand_prompt,
                    ).hiddens[0][0, -1]
                    z_rands.append(z_rand)

                zs = torch.stack([z_true, z_hard_subj, z_hard_rel, *z_rands], dim=0).float()
                z_pred = z_pred.float()
                sims = z_pred.mul(zs).sum(dim=-1) / (
                    z_pred.norm(dim=-1) * zs.norm(dim=-1)
                )
                chosen = sims.argmax().item()
                counts[chosen] += 1

    total = sum(counts.values())
    return ReconstructionBenchmarkResults(
        metrics=ReconstructionBenchmarkMetrics(
            frac_correct=counts[0] / total,
            frac_dist_subj=counts[1] / total,
            frac_dist_rel=counts[2] / total,
        )
    )


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkOutputs(DataClassJsonMixin):

    subject: str
    target: str
    lre: list[functional.PredictedToken]
    lm: list[functional.PredictedToken]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkRelationTrial(DataClassJsonMixin):
    train: data.Relation
    test: data.Relation
    outputs: list[FaithfulnessBenchmarkOutputs]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkRelationResults(DataClassJsonMixin):
    relation_name: str
    trials: list[FaithfulnessBenchmarkRelationTrial]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkMetrics(DataClassJsonMixin):
    recall_lm: list[float]
    recall_lre: list[float]
    recall_lre_if_lm_correct: list[float]
    recall_lre_if_lm_wrong: list[float]
    count_lm_correct: int
    count_lm_wrong: int


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkResults(DataClassJsonMixin):
    relations: list[FaithfulnessBenchmarkRelationResults]
    metrics: FaithfulnessBenchmarkMetrics


def faithfulness(
    *,
    estimator: operators.LinearRelationEstimator,
    dataset: data.RelationDataset,
    n_train: int = 3,
    n_trials: int = 3,
    k: int = 3,
    desc: str | None = None,
) -> FaithfulnessBenchmarkResults:
    """Measure how faithful the LREs are to the true relation.

    Put simply, evaluate how often LRE(subject) returns the true object in its
    top-k predictions. Additionally record how often LM(prompt % subject)
    produces correct answer.

    Args:
        estimator: LRE estimator.
        dataset: Dataset of relations.
        n_train: Number of samples in each relation to use for training.
        n_trials: Number of times to repeat the experiment for each relation.
        k: Number of top predictions to take from LRE.
        desc: Progress bar description.

    Returns:
        Benchmark results.

    """
    if desc is None:
        desc = "faithfulness"

    results_by_relation = []
    recalls_lm = []
    recalls_lre = []
    recalls_lre_if_lm_correct = []
    recalls_lre_if_lm_wrong = []
    count_lm_correct = 0
    count_lm_wrong = 0
    for relation in tqdm(dataset.relations, desc=desc):
        trials = []
        for _ in range(n_trials):
            prompt_template = random.choice(relation.prompt_templates)
            train, test = relation.set(prompt_templates=[prompt_template]).split(
                n_train
            )
            targets = [x.object for x in test.samples]

            operator = estimator(train)
            mt = operator.mt

            # Compute LM predictions.
            prompt_template_icl = functional.make_prompt(
                prompt_template=prompt_template,
                subject="{}",
                examples=train.samples,
            )
            prompts_lm = [prompt_template_icl.format(x.subject) for x in test.samples]
            outputs_lm = functional.predict_next_token(mt=mt, prompt=prompts_lm, k=k)
            preds_lm = [[x.token for x in xs] for xs in outputs_lm]
            recall_lm = metrics.recall(preds_lm, targets)
            recalls_lm.append(recall_lm)

            # Compute LRE predictions.
            outputs_lre = []
            for sample in test.samples:
                output_lre = operator(sample.subject, k=k)
                outputs_lre.append(output_lre.predictions)

            preds_lre = [[x.token for x in xs] for xs in outputs_lre]
            recall_lre = metrics.recall(preds_lre, targets)
            recalls_lre.append(recall_lre)

            # Compute LRE predictions if LM is correct.
            preds_lre_if_lm_correct = []
            targets_if_lm_correct = []
            preds_lre_if_lm_wrong = []
            targets_if_lm_wrong = []
            for pred_lm, pred_lre, target in zip(preds_lm, preds_lm, targets):
                if functional.any_is_nontrivial_prefix(pred_lm, target):
                    preds_lre_if_lm_correct.append(pred_lre)
                    targets_if_lm_correct.append(target)
                    count_lm_correct += 1
                else:
                    preds_lre_if_lm_wrong.append(pred_lre)
                    targets_if_lm_wrong.append(target)
                    count_lm_wrong += 1

            if preds_lre_if_lm_correct:
                assert targets_if_lm_correct
                recall_lre_if_lm_correct = metrics.recall(
                    preds_lre_if_lm_correct, targets_if_lm_correct
                )
                recalls_lre_if_lm_correct.append(recall_lre_if_lm_correct)

            if preds_lre_if_lm_wrong:
                assert targets_if_lm_wrong
                recall_lre_if_lm_wrong = metrics.recall(
                    preds_lre_if_lm_wrong, targets_if_lm_wrong
                )
                recalls_lre_if_lm_wrong.append(recall_lre_if_lm_wrong)

            trials.append(
                FaithfulnessBenchmarkRelationTrial(
                    train=train,
                    test=test,
                    outputs=[
                        FaithfulnessBenchmarkOutputs(
                            lre=lre, lm=lm, subject=sample.subject, target=sample.object
                        )
                        for lre, lm, sample in zip(
                            outputs_lre, outputs_lm, test.samples
                        )
                    ],
                )
            )
        results_by_relation.append(
            FaithfulnessBenchmarkRelationResults(
                relation_name=relation.name, trials=trials
            )
        )

    faithfulness_metrics = FaithfulnessBenchmarkMetrics(
        **{
            key: torch.tensor(values).mean(dim=0).tolist()
            for key, values in (
                ("recall_lm", recalls_lm),
                ("recall_lre", recalls_lre),
                ("recall_lre_if_lm_correct", recalls_lre_if_lm_correct),
                ("recall_lre_if_lm_wrong", recalls_lre_if_lm_wrong),
            )
        },
        count_lm_correct=count_lm_correct,
        count_lm_wrong=count_lm_wrong,
    )

    return FaithfulnessBenchmarkResults(
        relations=results_by_relation, metrics=faithfulness_metrics
    )
