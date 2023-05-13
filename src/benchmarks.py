import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src import data, editors, functional, metrics, models, operators

import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkRelationTrialSample(DataClassJsonMixin):
    subject: str

    other_subj: str | None = None

    other_rel_name: str | None = None
    other_rel_prompt_template: str | None = None

    sim_z_true: float | None = None
    sim_z_hard_subj: float | None = None
    sim_z_hard_rel: float | None = None
    sim_z_random: list[float] | None = None

    skipped: bool = False
    skipped_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkRelationTrial(DataClassJsonMixin):
    train: data.Relation
    test: data.Relation
    samples: list[ReconstructionBenchmarkRelationTrialSample]


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkRelationResults(DataClassJsonMixin):
    relation_name: str
    trials: list[ReconstructionBenchmarkRelationTrial]


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkMetrics(DataClassJsonMixin):
    frac_correct: float
    frac_dist_subj: float
    frac_dist_rel: float


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkResults(DataClassJsonMixin):
    relations: list[ReconstructionBenchmarkRelationResults]
    metrics: ReconstructionBenchmarkMetrics


def reconstruction(
    estimator: operators.LinearRelationEstimator,
    dataset: data.RelationDataset,
    n_trials: int = 3,
    n_train: int = 3,
    n_random_distractors: int = 3,
    n_top_lm: int = 3,
    n_icl_lm: int = 2,
    desc: str | None = None,
) -> ReconstructionBenchmarkResults:
    """Evaluate how much LRE looks like model's own representations.

    Args:
        estimator: LRE estimator.
        dataset: Dataset of relations.
        n_trials: Number of train/test splits to try.
        n_train: Number of samples to train on per relation.
        n_random_distractors: Number of random distractors to use in addition to the
            two hard distractors.
        n_top_lm: Consider this many top next token predictions when deciding whether
            model knows the answer or not.
        n_icl_lm: Number of ICL examples to use when prompting LM to see if it knows
            a subject.
        desc: Tqdm description.

    Returns:
        Benchmark results.

    """
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
    relation_results = []
    for relation in tqdm(dataset.relations, desc=desc):
        relation_trials = []
        for _ in range(n_trials):
            prompt_template = random.choice(relation.prompt_templates)

            known_samples = _determine_known_samples(
                mt=mt,
                relation=relation,
                prompt_template=prompt_template,
                n_icl_lm=n_icl_lm,
                n_top_lm=n_top_lm,
            )
            if len(known_samples) <= n_train:
                logger.debug(
                    f"lm does not know > n_train={n_train} samples for "
                    f'relation {relation.name}, prompt "{prompt_template}" will skip'
                )
                continue

            train, test = relation.set(
                samples=known_samples, prompt_templates=[prompt_template]
            ).split(n_train)

            # Estimate operator and evaluate it.
            operator = estimator(train)

            relation_samples = []
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
                    relation_samples.append(
                        ReconstructionBenchmarkRelationTrialSample(
                            subject=subject,
                            skipped=True,
                            skipped_reason="no other relations with this subject",
                        )
                    )
                    continue
                (other_rel_name, other_rel_prompt_template, _) = random.choice(matches)
                z_hard_subj_prompt = functional.make_prompt(
                    prompt_template=other_rel_prompt_template,
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
                    relation_samples.append(
                        ReconstructionBenchmarkRelationTrialSample(
                            subject=subject,
                            skipped=True,
                            skipped_reason="no other subjects have this relation",
                        )
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

                # Distractor 3+: chosen at random!
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
                    relation_samples.append(
                        ReconstructionBenchmarkRelationTrialSample(
                            subject=subject,
                            skipped=True,
                            skipped_reason="no other relations or subjects",
                        )
                    )
                    continue

                z_rands = []
                for _, other_prompt_template, other_subject in random.sample(
                    matches, k=n_random_distractors
                ):
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

                zs = torch.stack(
                    [z_true, z_hard_subj, z_hard_rel, *z_rands], dim=0
                ).float()
                z_pred = z_pred.float()
                sims = z_pred.mul(zs).sum(dim=-1) / (
                    z_pred.norm(dim=-1) * zs.norm(dim=-1)
                )
                chosen = sims.argmax().item()
                counts[chosen] += 1

                relation_samples.append(
                    ReconstructionBenchmarkRelationTrialSample(
                        subject=subject,
                        other_subj=other_subject,
                        other_rel_name=other_rel_name,
                        other_rel_prompt_template=other_rel_prompt_template,
                        sim_z_true=sims[0].item(),
                        sim_z_hard_subj=sims[1].item(),
                        sim_z_hard_rel=sims[2].item(),
                        sim_z_random=sims[3:].tolist(),
                    )
                )

            relation_trials.append(
                ReconstructionBenchmarkRelationTrial(
                    train=train,
                    test=test,
                    samples=relation_samples,
                )
            )
        relation_results.append(
            ReconstructionBenchmarkRelationResults(
                relation_name=relation.name,
                trials=relation_trials,
            )
        )

    if not counts:
        raise ValueError(
            "no trials were run, probably because "
            "none of the provided relations share a domain!"
        )

    total = sum(counts.values())
    return ReconstructionBenchmarkResults(
        relations=relation_results,
        metrics=ReconstructionBenchmarkMetrics(
            frac_correct=counts[0] / total,
            frac_dist_subj=counts[1] / total,
            frac_dist_rel=counts[2] / total,
        ),
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
    recalls_by_lm_correct = defaultdict(list)
    counts_by_lm_correct: dict[bool, int] = defaultdict(int)
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
            preds_by_lm_correct = defaultdict(list)
            targets_by_lm_correct = defaultdict(list)
            for pred_lm, pred_lre, target in zip(preds_lm, preds_lre, targets):
                lm_correct = functional.any_is_nontrivial_prefix(pred_lm, target)
                preds_by_lm_correct[lm_correct].append(pred_lre)
                targets_by_lm_correct[lm_correct].append(target)
                counts_by_lm_correct[lm_correct] += 1

            for correct in (True, False):
                preds = preds_by_lm_correct[correct]
                targets = targets_by_lm_correct[correct]
                if not preds:
                    assert not targets
                    continue
                recall = metrics.recall(
                    preds_by_lm_correct[correct], targets_by_lm_correct[correct]
                )
                recalls_by_lm_correct[correct].append(recall)

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
                ("recall_lre_if_lm_correct", recalls_by_lm_correct[True]),
                ("recall_lre_if_lm_wrong", recalls_by_lm_correct[False]),
            )
        },
        count_lm_correct=counts_by_lm_correct[True],
        count_lm_wrong=counts_by_lm_correct[False],
    )

    return FaithfulnessBenchmarkResults(
        relations=results_by_relation, metrics=faithfulness_metrics
    )


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkRelationTrialSample(DataClassJsonMixin):

    subject_original: str
    subject_target: str
    prompt_template: str

    prob_original: float
    prob_target: float

    predicted_tokens: list[functional.PredictedToken]


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkRelationTrial(DataClassJsonMixin):
    train: data.RelationDataset
    test: data.RelationDataset
    samples: list[CausalityBenchmarkRelationTrialSample]


@dataclass(frozen=True, kw_only=True)
class CausalityRelationResults(DataClassJsonMixin):
    relation_name: str
    trials: list[CausalityBenchmarkRelationTrial]


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkMetrics(DataClassJsonMixin):
    efficacy_mean: float
    efficacy_std: float


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkResults(DataClassJsonMixin):
    relations: list[CausalityRelationResults]
    metrics: CausalityBenchmarkMetrics


def causality(
    *,
    estimator: operators.LinearRelationEstimator,
    editor_type: type[editors.Editor],
    dataset: data.RelationDataset,
    n_train: int = 3,
    n_trials: int = 3,
    n_top_lm: int = 3,
    n_icl_lm: int = 3,
    desc: str | None = None,
    **kwargs: Any,
) -> CausalityBenchmarkResults:
    if desc is None:
        desc = "causality"

    mt = estimator.mt

    relation_results = []
    for relation in tqdm(dataset.relations, desc=desc):
        relation_trials = []
        for _ in range(n_trials):
            prompt_template = random.choice(relation.prompt_templates)

            known_samples = _determine_known_samples(
                mt=mt,
                relation=relation,
                prompt_template=prompt_template,
                n_icl_lm=n_icl_lm,
                n_top_lm=n_top_lm,
            )
            if len(known_samples) <= n_train:
                logger.debug(
                    f"lm does not know > n_train={n_train} samples for "
                    f'relation {relation.name}, prompt "{prompt_template}" will skip'
                )
                continue

            train, test = relation.set(
                samples=list(known_samples), prompt_templates=[prompt_template]
            ).split(n_train)

            editor_kwargs = dict(kwargs)
            if issubclass(editor_type, editors.LinearRelationEditor):
                operator = estimator(train)
                editor_kwargs["lre"] = operator
            editor = editor_type(mt=mt, **editor_kwargs)

            relation_samples = []
            for sample in test.samples:
                others = list(set(test.samples) - {sample})
                target = random.choice(others)

                subject_original = sample.subject
                subject_target = target.subject

                object_original = sample.object
                object_target = target.object

                result = editor(subject_original, subject_target)

                [token_id_original, token_id_target] = (
                    models.tokenize_words(mt, [object_original, object_target])
                    .input_ids[:, 0]
                    .tolist()
                )
                probs = result.model_logits[0, -1].float().softmax(dim=-1)
                prob_original = probs[token_id_original].item()
                prob_target = probs[token_id_target].item()

                relation_samples.append(
                    CausalityBenchmarkRelationTrialSample(
                        subject_original=subject_original,
                        subject_target=subject_target,
                        prompt_template=prompt_template,
                        prob_original=prob_original,
                        prob_target=prob_target,
                        predicted_tokens=result.predicted_tokens,
                    )
                )
            relation_trials.append(
                CausalityBenchmarkRelationTrial(
                    train=train, test=test, samples=relation_samples
                )
            )
        relation_results.append(
            CausalityRelationResults(
                relation_name=relation.name, trials=relation_trials
            )
        )

    efficacies = torch.tensor(
        [
            sample.prob_target > sample.prob_original
            for relation_result in relation_results
            for trial in relation_result.trials
            for sample in trial.samples
        ]
    )
    efficacy_mean = efficacies.float().mean().item()
    efficacy_std = efficacies.float().std().item()

    return CausalityBenchmarkResults(
        relations=relation_results,
        metrics=CausalityBenchmarkMetrics(
            efficacy_mean=efficacy_mean, efficacy_std=efficacy_std
        ),
    )


def _determine_known_samples(
    *,
    mt: models.ModelAndTokenizer,
    relation: data.Relation,
    prompt_template: str,
    n_icl_lm: int,
    n_top_lm: int,
) -> set[data.RelationSample]:
    """Filter samples down to only those that model knows.

    Most benchmarks rely on model knowing the relation at all.
    """
    prompts = [
        functional.make_prompt(
            prompt_template=prompt_template,
            mt=mt,
            subject=sample.subject,
            examples=random.sample(set(relation.samples) - {sample}, k=n_icl_lm),
        )
        for sample in relation.samples
    ]
    predictions = functional.predict_next_token(mt=mt, prompt=prompts, k=n_top_lm)
    known_samples = {
        sample
        for sample, topk in zip(relation.samples, predictions)
        if functional.any_is_nontrivial_prefix([x.token for x in topk], sample.object)
    }
    return known_samples
