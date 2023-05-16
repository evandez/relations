import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from src import data, editors, functional, metrics, models, operators
from src.functional import make_prompt
from src.utils.typing import PathLike

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
    results_dir: PathLike | None = None,
    resume: bool = False,
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
        results_dir: If provided, save intermediate results to this directory.

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
        relation_result = _load_relation_results(
            results_dir=results_dir,
            relation_name=relation.name,
            results_type=ReconstructionBenchmarkRelationResults,
            resume=resume,
        )
        if relation_result is not None:
            relation_results.append(relation_result)
            continue

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
                    prompt=make_prompt(
                        prompt_template=prompt_template, subject=subject, mt=mt
                    ),
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

        relation_result = ReconstructionBenchmarkRelationResults(
            relation_name=relation.name,
            trials=relation_trials,
        )
        relation_results.append(relation_result)
        _save_relation_results(
            results_dir=results_dir,
            relation_name=relation.name,
            results=relation_result,
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
    wrong: str
    lre: list[functional.PredictedToken]
    lm: list[functional.PredictedToken]
    zs: list[functional.PredictedToken]
    pd: list[functional.PredictedToken]
    lens: list[functional.PredictedToken]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkRelationTrial(DataClassJsonMixin):
    train: data.Relation
    test: data.Relation
    template: str
    outputs: list[FaithfulnessBenchmarkOutputs]
    recall_lm: list[float]
    recall_lre: list[float]
    recall_zs: list[float]
    recall_pd: list[float]
    recall_lens: list[float]
    recall_lre_if_lm_correct: list[float]
    recall_lre_if_lm_wrong: list[float]
    recall_pd_if_zs_correct: list[float]
    recall_pd_if_zs_wrong: list[float]
    recall_lens_if_zs_correct: list[float]
    recall_lens_if_zs_wrong: list[float]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkRelationResults(DataClassJsonMixin):
    relation_name: str
    trials: list[FaithfulnessBenchmarkRelationTrial]


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkMetrics(DataClassJsonMixin):
    recall_lm: list[float]
    recall_lre: list[float]
    recall_zs: list[float]
    recall_pd: list[float]
    recall_lens: list[float]
    recall_lre_if_lm_correct: list[float]
    recall_lre_if_lm_wrong: list[float]
    recall_pd_if_zs_correct: list[float]
    recall_pd_if_zs_wrong: list[float]
    recall_lens_if_zs_correct: list[float]
    recall_lens_if_zs_wrong: list[float]
    count_lm_correct: int
    count_lm_wrong: int
    count_zs_correct: int
    count_zs_wrong: int


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
    results_dir: PathLike | None = None,
    resume: bool = False,
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
        results_dir: Save and read intermediate results from this directory.

    Returns:
        Benchmark results.

    """
    if desc is None:
        desc = "faithfulness"

    results_by_relation = []
    recalls_lm = []
    recalls_lre = []
    recalls_zs = []
    recalls_pd = []
    recalls_lens = []
    recalls_by_lm_correct = defaultdict(list)
    counts_by_lm_correct: dict[bool, int] = defaultdict(int)
    recalls_pd_by_zs_correct = defaultdict(list)
    recalls_lens_by_zs_correct = defaultdict(list)
    counts_by_zs_correct: dict[bool, int] = defaultdict(int)
    progress = tqdm(dataset.relations, desc=desc)
    for relation in progress:
        progress.set_description(relation.name)

        relation_results = _load_relation_results(
            results_dir=results_dir,
            relation_name=relation.name,
            results_type=FaithfulnessBenchmarkRelationResults,
            resume=resume,
        )
        if relation_results is not None:
            results_by_relation.append(relation_results)
            continue

        trials = []
        for _ in range(n_trials):
            prompt_template = random.choice(relation.prompt_templates)
            train, test = relation.set(prompt_templates=[prompt_template]).split(
                n_train
            )
            targets = [x.object for x in test.samples]
            wrong_targets = functional.random_incorrect_targets(targets)

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

            # Begin attribute-lens tests on the distracted case

            # Compute zero-shot predictions.
            prompts_zs = [
                make_prompt(prompt_template=prompt_template, subject=x.subject, mt=mt)
                for x in test.samples
            ]
            outputs_zs = functional.predict_next_token(mt=mt, prompt=prompts_zs, k=k)
            preds_zs = [[x.token for x in xs] for xs in outputs_zs]
            recall_zs = metrics.recall(preds_zs, targets)
            recalls_zs.append(recall_zs)
            # for p, o in zip(prompts_zs, outputs_zs):
            #    print(p, o)
            # print('ZS', recall_zs)

            # Compute poetry-distracted predictions.
            def poetry_prefix(subject, wrong):
                return "".join(
                    [prompt_template.format(subject) + " " + wrong + ". "] * 2
                )

            prompts_pd = [
                make_prompt(
                    prompt_template=poetry_prefix(x.subject, wrong) + prompt_template,
                    subject=x.subject,
                    mt=mt,
                )
                for x, wrong in zip(test.samples, wrong_targets)
            ]
            outputs_pd = functional.predict_next_token(
                mt=mt, prompt=prompts_pd, k=k, batch_size=1
            )  # Shrink the batch size to fit long prompts.
            preds_pd = [[x.token for x in xs] for xs in outputs_pd]
            recall_pd = metrics.recall(preds_pd, targets)
            recalls_pd.append(recall_pd)
            # print('PD', recall_pd)

            # Compute attribute lens: LRE predictions on the PD samples.
            outputs_lens = []
            for x, p in zip(test.samples, prompts_pd):
                h = functional.get_hidden_state_at_subject(
                    mt, p, x.subject, operator.h_layer
                )
                output_lens = operator("", k=k, h=h)
                outputs_lens.append(output_lens.predictions)
            preds_lens = [[x.token for x in xs] for xs in outputs_lens]
            recall_lens = metrics.recall(preds_lens, targets)
            recalls_lens.append(recall_lens)
            # print('LENS', recall_lens)

            # Compute PD, LRE predictions if ZS is correct.
            preds_pd_by_zs_correct = defaultdict(list)
            preds_lens_by_zs_correct = defaultdict(list)
            targets_by_zs_correct = defaultdict(list)
            for pred_zs, pred_pd, pred_lens, target in zip(
                preds_zs, preds_pd, preds_lens, targets
            ):
                zs_correct = functional.any_is_nontrivial_prefix(pred_zs, target)
                preds_pd_by_zs_correct[zs_correct].append(pred_pd)
                preds_lens_by_zs_correct[zs_correct].append(pred_lens)
                targets_by_zs_correct[zs_correct].append(target)
                counts_by_zs_correct[zs_correct] += 1

            ## end attribute-lens tests

            recall_by_lm_correct = {}
            recall_pd_by_zs_correct = {}
            recall_lens_by_zs_correct = {}
            for correct in (True, False):
                recall_by_lm_correct[correct] = metrics.recall(
                    preds_by_lm_correct[correct], targets_by_lm_correct[correct]
                )
                if recall_by_lm_correct[correct] is not None:
                    recalls_by_lm_correct[correct].append(recall_by_lm_correct[correct])
                recall_pd_by_zs_correct[correct] = metrics.recall(
                    preds_pd_by_zs_correct[correct], targets_by_zs_correct[correct]
                )
                if recall_pd_by_zs_correct[correct] is not None:
                    recalls_pd_by_zs_correct[correct].append(
                        recall_pd_by_zs_correct[correct]
                    )
                recall_lens_by_zs_correct[correct] = metrics.recall(
                    preds_lens_by_zs_correct[correct], targets_by_zs_correct[correct]
                )
                if recall_lens_by_zs_correct[correct] is not None:
                    recalls_lens_by_zs_correct[correct].append(
                        recall_lens_by_zs_correct[correct]
                    )

            trials.append(
                FaithfulnessBenchmarkRelationTrial(
                    train=train,
                    test=test,
                    template=prompt_template,
                    outputs=[
                        FaithfulnessBenchmarkOutputs(
                            lre=lre,
                            lm=lm,
                            zs=zs,
                            pd=pd,
                            lens=lens,
                            subject=sample.subject,
                            target=sample.object,
                            wrong=wrong,
                        )
                        for lre, lm, zs, pd, lens, sample, wrong in zip(
                            outputs_lre,
                            outputs_lm,
                            outputs_zs,
                            outputs_pd,
                            outputs_lens,
                            test.samples,
                            wrong_targets,
                        )
                    ],
                    # Record recall of individual trials for debugging
                    recall_lm=recall_lm,
                    recall_lre=recall_lre,
                    recall_zs=recall_zs,
                    recall_pd=recall_pd,
                    recall_lens=recall_lens,
                    recall_lre_if_lm_correct=recall_by_lm_correct[True],
                    recall_lre_if_lm_wrong=recall_by_lm_correct[False],
                    recall_pd_if_zs_correct=recall_pd_by_zs_correct[True],
                    recall_pd_if_zs_wrong=recall_pd_by_zs_correct[False],
                    recall_lens_if_zs_correct=recall_lens_by_zs_correct[True],
                    recall_lens_if_zs_wrong=recall_lens_by_zs_correct[False],
                )
            )

        relation_results = FaithfulnessBenchmarkRelationResults(
            relation_name=relation.name, trials=trials
        )
        _save_relation_results(
            results_dir=results_dir,
            relation_name=relation.name,
            results=relation_results,
        )
        results_by_relation.append(relation_results)

    faithfulness_metrics = FaithfulnessBenchmarkMetrics(
        **{
            key: torch.tensor(values).mean(dim=0).tolist()
            for key, values in (
                ("recall_lm", recalls_lm),
                ("recall_lre", recalls_lre),
                ("recall_zs", recalls_zs),
                ("recall_pd", recalls_pd),
                ("recall_lens", recalls_lens),
                ("recall_lre_if_lm_correct", recalls_by_lm_correct[True]),
                ("recall_lre_if_lm_wrong", recalls_by_lm_correct[False]),
                ("recall_pd_if_zs_correct", recalls_pd_by_zs_correct[True]),
                ("recall_pd_if_zs_wrong", recalls_pd_by_zs_correct[False]),
                ("recall_lens_if_zs_correct", recalls_lens_by_zs_correct[True]),
                ("recall_lens_if_zs_wrong", recalls_lens_by_zs_correct[False]),
            )
        },
        count_lm_correct=counts_by_lm_correct[True],
        count_lm_wrong=counts_by_lm_correct[False],
        count_zs_correct=counts_by_zs_correct[True],
        count_zs_wrong=counts_by_zs_correct[False],
    )

    return FaithfulnessBenchmarkResults(
        relations=results_by_relation, metrics=faithfulness_metrics
    )


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkRelationTrialSample(DataClassJsonMixin):
    subject_original: str
    subject_target: str
    object_target: str
    prompt_template: str

    prob_original: float
    prob_target: float

    predicted_tokens: list[functional.PredictedToken]
    model_generations: list[str]


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
    results_dir: PathLike | None = None,
    resume: bool = False,
    **kwargs: Any,
) -> CausalityBenchmarkResults:
    if desc is None:
        desc = "causality"

    mt = estimator.mt

    results_by_relation = []
    for relation in tqdm(dataset.relations, desc=desc):
        relation_results = _load_relation_results(
            results_dir=results_dir,
            relation_name=relation.name,
            results_type=CausalityRelationResults,
            resume=resume,
        )
        if relation_results is not None:
            results_by_relation.append(relation_results)
            continue

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
            editor = editor_type(**editor_kwargs)

            relation_samples = []
            for sample in test.samples:
                others = list(
                    {
                        x
                        for x in test.samples
                        if x.subject != sample.subject and x.object != sample.object
                    }
                )
                if not others:
                    logger.debug(
                        "no sample with different subject and different object "
                        f"than {sample}, skipping"
                    )
                    continue
                target = random.choice(others)

                subject_original = sample.subject
                subject_target = target.subject

                object_original = sample.object
                object_target = target.object

                if issubclass(editor_type, editors.LowRankPInvEmbedEditor):
                    result = editor(subject_original, object_target)
                else:
                    result = editor(subject_original, subject_target)

                [token_id_original, token_id_target] = (
                    models.tokenize_words(mt, [object_original, object_target])
                    .input_ids[:, 0]
                    .tolist()
                )
                probs = result.model_logits.float().softmax(dim=-1)
                prob_original = probs[token_id_original].item()
                prob_target = probs[token_id_target].item()

                relation_samples.append(
                    CausalityBenchmarkRelationTrialSample(
                        subject_original=subject_original,
                        subject_target=subject_target,
                        object_target=object_target,
                        prompt_template=prompt_template,
                        prob_original=prob_original,
                        prob_target=prob_target,
                        predicted_tokens=result.predicted_tokens,
                        model_generations=result.model_generations,
                    )
                )
            relation_trials.append(
                CausalityBenchmarkRelationTrial(
                    train=train, test=test, samples=relation_samples
                )
            )

        relation_results = CausalityRelationResults(
            relation_name=relation.name, trials=relation_trials
        )
        _save_relation_results(
            results_dir=results_dir,
            relation_name=relation.name,
            results=relation_results,
        )
        results_by_relation.append(relation_results)

    efficacies = torch.tensor(
        [
            sample.prob_target > sample.prob_original
            for relation_result in results_by_relation
            for trial in relation_result.trials
            for sample in trial.samples
        ]
    )
    efficacy_mean = efficacies.float().mean().item()
    efficacy_std = efficacies.float().std().item()

    return CausalityBenchmarkResults(
        relations=results_by_relation,
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


T = TypeVar("T", bound=DataClassJsonMixin)


def _load_relation_results(
    *,
    results_dir: PathLike | None,
    relation_name: str,
    results_type: type[T],
    resume: bool,
) -> T | None:
    """Read a relation result, if present."""
    if results_dir is None or not resume:
        logger.debug("results_dir not set, so not reading intermediate results")
        return None

    relation_results_file = _relation_results_file(
        results_dir=results_dir,
        relation_name=relation_name,
    )
    if not relation_results_file.exists():
        logger.debug(f'no intermediate results for "{relation_name}"')
        return None

    logger.debug(f"reading intermediate results from {relation_results_file}")
    with relation_results_file.open("r") as handle:
        return results_type.from_json(handle.read())


def _save_relation_results(
    *,
    results_dir: PathLike | None,
    relation_name: str,
    results: T,
) -> None:
    """Save relation result."""
    if results_dir is None:
        logger.debug(
            "results_dir not set, so not saving intermediate results for "
            f'"{relation_name}"'
        )
        return None
    relation_results_file = _relation_results_file(
        results_dir=results_dir,
        relation_name=relation_name,
    )
    logger.debug(f"saving intermediate results to {relation_results_file}")
    relation_results_file.parent.mkdir(exist_ok=True, parents=True)
    with relation_results_file.open("w") as handle:
        handle.write(results.to_json())


def _relation_results_file(
    *,
    results_dir: PathLike,
    relation_name: str,
) -> Path:
    relation_name_slug = relation_name.replace(" ", "_").replace("'", "")
    return Path(results_dir) / f"{relation_name_slug}.json"
