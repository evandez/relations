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


@torch.inference_mode()
def reconstruction(
    estimator: operators.LinearRelationEstimator,
    dataset: data.RelationDataset,
    desc: str | None = None,
) -> ReconstructionBenchmarkResults:
    if desc is None:
        desc = "reconstruction"

    operators = {}
    for relation in dataset.relations:
        train_settings = [
            (relation.name, prompt_template, sample)
            for prompt_template in relation.prompt_templates
            for sample in relation.samples
        ]

        for relation_name, prompt_template, sample in tqdm(
            train_settings, desc=f"{desc} [compute operators]"
        ):
            train_relation = data.Relation(
                name=relation.name,
                prompt_templates=[prompt_template],
                samples=[sample],
                _domain=list(relation.domain),
                _range=list(relation.range),
            )
            operator = estimator(train_relation)
            operators[relation_name, prompt_template, sample.subject] = operator

    counts: dict[int, int] = defaultdict(int)
    for (relation_name, prompt_template, subject), operator in tqdm(
        operators.items(), desc=f"{desc} [compute scores]"
    ):
        z_true = functional.compute_hidden_states(
            mt=estimator.mt,
            layers=[operator.z_layer],
            prompt=prompt_template.format(subject),
        ).hiddens[0][0, -1]

        key = random.choice(
            [
                (r, p, s)
                for r, p, s in operators
                if r == relation_name and (p != prompt_template or s != subject)
            ]
        )
        operator = operators[key]
        z_pred = operator(subject).z

        # Distractor 1: same subject, different relation
        matches = [
            (r, p, s) for r, p, s in operators if r == relation_name and s != subject
        ]
        if not matches:
            logger.debug(
                f"skipped {relation_name}/{prompt_template}/{subject} "
                "because no other relations have this subject"
            )
            continue
        (_, other_prompt_template, other_subject) = random.choice(matches)
        z_dist_subj = functional.compute_hidden_states(
            mt=estimator.mt,
            layers=[operator.z_layer],
            prompt=other_prompt_template.format(other_subject),
        ).hiddens[0][0, -1]

        # Distractor 2: same relation, different subject
        matches = [
            (r, p, s) for r, p, s in operators if r == relation_name and s != subject
        ]
        if not matches:
            logger.debug(
                f"skipped {relation_name}/{prompt_template}/{subject} "
                "because no other subjects have this relation"
            )
            continue
        (_, other_prompt_template, other_subject) = random.choice(matches)
        z_dist_rel = functional.compute_hidden_states(
            mt=estimator.mt,
            layers=[operator.z_layer],
            prompt=other_prompt_template.format(other_subject),
        ).hiddens[0][0, -1]

        zs = torch.stack([z_true, z_dist_subj, z_dist_rel], dim=0).float()
        z_pred = z_pred.float()
        sims = z_pred.mul(zs).sum(dim=-1) / (z_pred.norm(dim=-1) * zs.norm(dim=-1))
        chosen = sims.argmax().item()
        counts[chosen] += 1

    return ReconstructionBenchmarkResults(
        metrics=ReconstructionBenchmarkMetrics(
            frac_correct=counts[0] / sum(counts.values()),
            frac_dist_subj=counts[1] / sum(counts.values()),
            frac_dist_rel=counts[2] / sum(counts.values()),
        )
    )


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkRelationTrial(DataClassJsonMixin):
    train: data.Relation
    test: data.Relation
    outputs: list[operators.RelationOutput]
    recall: list[float]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkRelationResults(DataClassJsonMixin):
    relation: data.Relation
    trials: list[FaithfulnessBenchmarkRelationTrial]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkMetrics(DataClassJsonMixin):
    recall: list[float]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkResults(DataClassJsonMixin):
    relations: list[FaithfulnessBenchmarkRelationResults]
    metrics: FaithfulnessBenchmarkMetrics


# TODO(evandez): Record predictions, save models and hiddens, etc.
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
    top-k predictions.

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

    results = []
    for relation in tqdm(dataset.relations, desc=desc):
        trials = []
        for _ in range(n_trials):
            prompt_template = random.choice(relation.prompt_templates)
            train, test = relation.set(prompt_templates=[prompt_template]).split(
                n_train
            )
            operator = estimator(train)

            outputs = []
            predictions = []
            for sample in test.samples:
                output = operator(sample.subject, k=k)
                outputs.append(output.as_relation_output())
                predictions.append([p.token for p in output.predictions])

            targets = [sample.object for sample in test.samples]

            recall = metrics.recall(predictions, targets)
            trials.append(
                FaithfulnessBenchmarkRelationTrial(
                    train=train, test=test, outputs=outputs, recall=recall
                )
            )
        results.append(
            FaithfulnessBenchmarkRelationResults(relation=relation, trials=trials)
        )

    recalls = torch.tensor([[trial.recall for trial in r.trials] for r in results])
    faithfulness_metrics = FaithfulnessBenchmarkMetrics(
        recall=recalls.mean(dim=(0, 1)).tolist()
    )

    return FaithfulnessBenchmarkResults(relations=results, metrics=faithfulness_metrics)
