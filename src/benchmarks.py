import logging
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, Sequence

from src import data, editors, functional, hparams, metrics, models, operators
from src.functional import make_prompt
from src.utils import dataclasses_utils, experiment_utils, tokenizer_utils
from src.utils.typing import Layer, PathLike, StrSequence

import torch
from baukit import nethook
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
        relation_result = experiment_utils.load_results_file(
            results_dir=results_dir,
            name=relation.name,
            results_type=ReconstructionBenchmarkRelationResults,
            resume=resume,
        )
        if relation_result is not None:
            relation_results.append(relation_result)
            continue

        relation_trials = []
        for _ in range(n_trials):
            prompt_template = relation.prompt_templates[0]
            train, test = relation.set(prompt_templates=[prompt_template]).split(
                n_train
            )

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
        experiment_utils.save_results_file(
            results_dir=results_dir,
            name=relation.name,
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
    pdlens: list[functional.PredictedToken]
    rd: list[functional.PredictedToken]
    rdlens: list[functional.PredictedToken]


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
    recall_pdlens: list[float]
    recall_rd: list[float]
    recall_rdlens: list[float]
    recall_lre_if_lm_correct: list[float]
    recall_lre_if_lm_wrong: list[float]
    recall_pd_if_zs_correct: list[float]
    recall_pd_if_zs_wrong: list[float]
    recall_pdlens_if_zs_correct: list[float]
    recall_pdlens_if_zs_wrong: list[float]
    recall_rd_if_zs_correct: list[float]
    recall_rd_if_zs_wrong: list[float]
    recall_rdlens_if_zs_correct: list[float]
    recall_rdlens_if_zs_wrong: list[float]
    count_lm_correct: int
    count_lm_wrong: int
    count_zs_correct: int
    count_zs_wrong: int


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
    recall_pdlens: list[float]
    recall_rd: list[float]
    recall_rdlens: list[float]
    recall_lre_if_lm_correct: list[float]
    recall_lre_if_lm_wrong: list[float]
    recall_pd_if_zs_correct: list[float]
    recall_pd_if_zs_wrong: list[float]
    recall_pdlens_if_zs_correct: list[float]
    recall_pdlens_if_zs_wrong: list[float]
    recall_rd_if_zs_correct: list[float]
    recall_rd_if_zs_wrong: list[float]
    recall_rdlens_if_zs_correct: list[float]
    recall_rdlens_if_zs_wrong: list[float]
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
    mt: models.ModelAndTokenizer,
    dataset: data.RelationDataset,
    estimator_type: type[operators.LinearRelationEstimator],
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
    recalls_pdlens = []
    recalls_rd = []
    recalls_rdlens = []
    recalls_by_lm_correct = defaultdict(list)
    counts_by_lm_correct: dict[bool, int] = defaultdict(int)
    recalls_pd_by_zs_correct = defaultdict(list)
    recalls_pdlens_by_zs_correct = defaultdict(list)
    recalls_rd_by_zs_correct = defaultdict(list)
    recalls_rdlens_by_zs_correct = defaultdict(list)
    counts_by_zs_correct: dict[bool, int] = defaultdict(int)
    progress = tqdm(dataset.relations, desc=desc)
    for relation in progress:
        progress.set_description(relation.name)

        relation_results = experiment_utils.load_results_file(
            results_dir=results_dir,
            name=relation.name,
            results_type=FaithfulnessBenchmarkRelationResults,
            resume=resume,
        )
        if relation_results is not None:
            results_by_relation.append(relation_results)
            continue

        # TODO(evan): Index on model once changes are merged.
        relation_hparams = hparams.get(relation.name)
        estimator = dataclasses_utils.create_with_optional_kwargs(
            estimator_type,
            mt=mt,
            h_layer=relation_hparams.h_layer,
            z_layer=relation_hparams.z_layer,
            beta=relation_hparams.beta,
        )

        trials = []
        for _ in range(n_trials):
            prompt_template = relation.prompt_templates[0]
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

            # Begin attribute-lens tests on the distracted cases

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
            def poetry_prefix(subject, wrong):  # type: ignore
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
            outputs_pdlens = []
            for x, p in zip(test.samples, prompts_pd):
                h = functional.get_hidden_state_at_subject(
                    mt, p, x.subject, operator.h_layer
                )
                output_pdlens = operator("", k=k, h=h)
                outputs_pdlens.append(output_pdlens.predictions)
            preds_pdlens = [[x.token for x in xs] for xs in outputs_pdlens]
            recall_pdlens = metrics.recall(preds_pdlens, targets)
            recalls_pdlens.append(recall_pdlens)
            # print('LENS', recall_pdlens)

            # Compute PD, LRE predictions if ZS is correct.
            preds_pd_by_zs_correct = defaultdict(list)
            preds_pdlens_by_zs_correct = defaultdict(list)
            targets_by_zs_correct = defaultdict(list)
            for pred_zs, pred_pd, pred_pdlens, target in zip(
                preds_zs, preds_pd, preds_pdlens, targets
            ):
                zs_correct = functional.any_is_nontrivial_prefix(pred_zs, target)
                preds_pd_by_zs_correct[zs_correct].append(pred_pd)
                preds_pdlens_by_zs_correct[zs_correct].append(pred_pdlens)
                targets_by_zs_correct[zs_correct].append(target)
                counts_by_zs_correct[zs_correct] += 1

            # Compute repeat-distracted predictions.
            def repeat_prefix(subject, wrong):  # type: ignore
                return (
                    prompt_template.format(subject) + " " + wrong + ". Repeat exactly: "
                )

            prompts_rd = [
                make_prompt(
                    prompt_template=repeat_prefix(x.subject, wrong) + prompt_template,
                    subject=x.subject,
                    mt=mt,
                )
                for x, wrong in zip(test.samples, wrong_targets)
            ]
            outputs_rd = functional.predict_next_token(
                mt=mt, prompt=prompts_rd, k=k, batch_size=1
            )  # Shrink the batch size to fit long prompts.
            preds_rd = [[x.token for x in xs] for xs in outputs_rd]
            recall_rd = metrics.recall(preds_rd, targets)
            recalls_rd.append(recall_rd)
            # print('RD', recall_rd)

            # Compute attribute lens: LRE predictions on the RD samples.
            outputs_rdlens = []
            for x, p in zip(test.samples, prompts_rd):
                h = functional.get_hidden_state_at_subject(
                    mt, p, x.subject, operator.h_layer
                )
                output_rdlens = operator("", k=k, h=h)
                outputs_rdlens.append(output_rdlens.predictions)
            preds_rdlens = [[x.token for x in xs] for xs in outputs_rdlens]
            recall_rdlens = metrics.recall(preds_rdlens, targets)
            recalls_rdlens.append(recall_rdlens)
            # print('LENS', recall_rdlens)

            # Compute RD, LRE predictions if ZS is correct.
            preds_rd_by_zs_correct = defaultdict(list)
            preds_rdlens_by_zs_correct = defaultdict(list)
            targets_by_zs_correct = defaultdict(list)
            for pred_zs, pred_rd, pred_rdlens, target in zip(
                preds_zs, preds_rd, preds_rdlens, targets
            ):
                zs_correct = functional.any_is_nontrivial_prefix(pred_zs, target)
                preds_rd_by_zs_correct[zs_correct].append(pred_rd)
                preds_rdlens_by_zs_correct[zs_correct].append(pred_rdlens)
                targets_by_zs_correct[zs_correct].append(target)
                counts_by_zs_correct[zs_correct] += 1

            ## end attribute-lens tests

            recall_by_lm_correct = {}
            recall_pd_by_zs_correct = {}
            recall_pdlens_by_zs_correct = {}
            recall_rd_by_zs_correct = {}
            recall_rdlens_by_zs_correct = {}
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
                recall_pdlens_by_zs_correct[correct] = metrics.recall(
                    preds_pdlens_by_zs_correct[correct], targets_by_zs_correct[correct]
                )
                if recall_pdlens_by_zs_correct[correct] is not None:
                    recalls_pdlens_by_zs_correct[correct].append(
                        recall_pdlens_by_zs_correct[correct]
                    )
                recall_rd_by_zs_correct[correct] = metrics.recall(
                    preds_rd_by_zs_correct[correct], targets_by_zs_correct[correct]
                )
                if recall_rd_by_zs_correct[correct] is not None:
                    recalls_rd_by_zs_correct[correct].append(
                        recall_rd_by_zs_correct[correct]
                    )
                recall_rdlens_by_zs_correct[correct] = metrics.recall(
                    preds_rdlens_by_zs_correct[correct], targets_by_zs_correct[correct]
                )
                if recall_rdlens_by_zs_correct[correct] is not None:
                    recalls_rdlens_by_zs_correct[correct].append(
                        recall_rdlens_by_zs_correct[correct]
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
                            rd=rd,
                            pdlens=pdlens,
                            rdlens=rdlens,
                            subject=sample.subject,
                            target=sample.object,
                            wrong=wrong,
                        )
                        for lre, lm, zs, pd, rd, pdlens, rdlens, sample, wrong in zip(
                            outputs_lre,
                            outputs_lm,
                            outputs_zs,
                            outputs_pd,
                            outputs_rd,
                            outputs_pdlens,
                            outputs_rdlens,
                            test.samples,
                            wrong_targets,
                        )
                    ],
                    # Record recall of individual trials for debugging
                    recall_lm=recall_lm,
                    recall_lre=recall_lre,
                    recall_zs=recall_zs,
                    recall_pd=recall_pd,
                    recall_pdlens=recall_pdlens,
                    recall_rd=recall_rd,
                    recall_rdlens=recall_rdlens,
                    recall_lre_if_lm_correct=recall_by_lm_correct[True],
                    recall_lre_if_lm_wrong=recall_by_lm_correct[False],
                    recall_pd_if_zs_correct=recall_pd_by_zs_correct[True],
                    recall_pd_if_zs_wrong=recall_pd_by_zs_correct[False],
                    recall_pdlens_if_zs_correct=recall_pdlens_by_zs_correct[True],
                    recall_pdlens_if_zs_wrong=recall_pdlens_by_zs_correct[False],
                    recall_rd_if_zs_correct=recall_rd_by_zs_correct[True],
                    recall_rd_if_zs_wrong=recall_rd_by_zs_correct[False],
                    recall_rdlens_if_zs_correct=recall_rdlens_by_zs_correct[True],
                    recall_rdlens_if_zs_wrong=recall_rdlens_by_zs_correct[False],
                    count_lm_correct=len(targets_by_lm_correct[True]),
                    count_lm_wrong=len(targets_by_lm_correct[False]),
                    count_zs_correct=len(targets_by_zs_correct[True]),
                    count_zs_wrong=len(targets_by_zs_correct[False]),
                )
            )

        relation_results = FaithfulnessBenchmarkRelationResults(
            relation_name=relation.name, trials=trials
        )
        experiment_utils.save_results_file(
            results_dir=results_dir,
            name=relation.name,
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
                ("recall_pdlens", recalls_pdlens),
                ("recall_rd", recalls_rd),
                ("recall_rdlens", recalls_rdlens),
                ("recall_lre_if_lm_correct", recalls_by_lm_correct[True]),
                ("recall_lre_if_lm_wrong", recalls_by_lm_correct[False]),
                ("recall_pd_if_zs_correct", recalls_pd_by_zs_correct[True]),
                ("recall_pd_if_zs_wrong", recalls_pd_by_zs_correct[False]),
                ("recall_pdlens_if_zs_correct", recalls_pdlens_by_zs_correct[True]),
                ("recall_pdlens_if_zs_wrong", recalls_pdlens_by_zs_correct[False]),
                ("recall_rd_if_zs_correct", recalls_rd_by_zs_correct[True]),
                ("recall_rd_if_zs_wrong", recalls_rd_by_zs_correct[False]),
                ("recall_rdlens_if_zs_correct", recalls_rdlens_by_zs_correct[True]),
                ("recall_rdlens_if_zs_wrong", recalls_rdlens_by_zs_correct[False]),
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

    # Post-edit LM preds at current rank.
    edited_lm_preds: list[functional.PredictedToken]
    edited_lm_generations: list[str]

    # LRE preds at current rank.
    lre_preds: list[functional.PredictedToken] | None = None


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkRelationTrialRank(DataClassJsonMixin):
    rank: int
    samples: list[CausalityBenchmarkRelationTrialSample]

    def efficacy(self) -> float:
        return sum(x.prob_target > x.prob_original for x in self.samples) / len(
            self.samples
        )


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkRelationTrial(DataClassJsonMixin):
    train: data.RelationDataset
    test: data.RelationDataset
    ranks: list[CausalityBenchmarkRelationTrialRank]

    def best(self) -> CausalityBenchmarkRelationTrialRank:
        return max(self.ranks, key=lambda x: x.efficacy())


@dataclass(frozen=True, kw_only=True)
class CausalityRelationResults(DataClassJsonMixin):
    relation_name: str
    trials: list[CausalityBenchmarkRelationTrial]


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkMetrics(DataClassJsonMixin):
    efficacy: metrics.AggregateMetric


@dataclass(frozen=True, kw_only=True)
class CausalityBenchmarkResults(DataClassJsonMixin):
    relations: list[CausalityRelationResults]
    metrics: CausalityBenchmarkMetrics


def causality(
    *,
    mt: models.ModelAndTokenizer,
    dataset: data.RelationDataset,
    estimator_type: type[operators.LinearRelationEstimator],
    editor_type: type[editors.Editor],
    n_train: int = 3,
    n_trials: int = 3,
    ranks: Sequence[int] | None = None,
    desc: str | None = None,
    results_dir: PathLike | None = None,
    resume: bool = False,
    **kwargs: Any,
) -> CausalityBenchmarkResults:
    if desc is None:
        desc = "causality"
    if ranks is None:
        ranks = [*range(0, 50, 5), *range(50, 100, 10), *range(100, 250, 25)]

    results_by_relation = []
    for relation in tqdm(dataset.relations, desc=desc):
        relation_results = experiment_utils.load_results_file(
            results_dir=results_dir,
            name=relation.name,
            results_type=CausalityRelationResults,
            resume=resume,
        )
        if relation_results is not None:
            results_by_relation.append(relation_results)
            continue

        relation_hparams = hparams.get(relation.name)
        estimator = dataclasses_utils.create_with_optional_kwargs(
            estimator_type,
            mt=mt,
            h_layer=relation_hparams.h_layer,
            z_layer=relation_hparams.z_layer,
            beta=relation_hparams.beta,
        )

        relation_trials = []
        for _ in range(n_trials):
            prompt_template = relation.prompt_templates[0]

            train, test = relation.set(prompt_templates=[prompt_template]).split(
                n_train
            )

            operator = None
            svd = None
            if issubclass(editor_type, editors.LinearRelationEditor):
                logger.debug(
                    f"estimate operator for: {[str(x) for x in train.samples]}"
                )
                operator = estimator(train)
                if operator.weight is not None:
                    svd = torch.svd(operator.weight)

            logger.debug("precompute test zs")
            [hs_by_subj, zs_by_subj] = _precompute_zs(
                mt=mt,
                prompt_template=prompt_template,
                subjects=[x.subject for x in test.samples],
                examples=train.samples,
                h_layer=operator.h_layer if operator is not None else None,
                z_layer=operator.z_layer if operator is not None else None,
            )

            relation_ranks = []
            for rank in ranks:
                relation_samples = []
                for sample in test.samples:
                    others = [
                        x
                        for x in test.samples
                        if x.subject != sample.subject and x.object != sample.object
                    ]
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

                    # Perform the edit and record LM outputs.
                    editor = dataclasses_utils.create_with_optional_kwargs(
                        editor_type,
                        rank=rank,
                        lre=operator,
                        svd=svd,
                        prompt_template=prompt_template,
                        mt=mt,
                        **kwargs,
                    )

                    result: editors.EditResult
                    if editor_type.expects() == "object":
                        result = nethook.invoke_with_optional_args(
                            editor,
                            subject_original,
                            object_target,
                            z_original=zs_by_subj.get(subject_original),
                        )
                    else:
                        assert editor_type.expects() == "subject"
                        result = nethook.invoke_with_optional_args(
                            editor,
                            subject_original,
                            subject_target,
                            z_original=zs_by_subj.get(subject_original),
                            z_target=zs_by_subj.get(subject_target),
                        )

                    [token_id_original, token_id_target] = (
                        models.tokenize_words(mt, [object_original, object_target])
                        .input_ids[:, 0]
                        .tolist()
                    )
                    probs = result.model_logits.float().softmax(dim=-1)
                    prob_original = probs[token_id_original].item()
                    prob_target = probs[token_id_target].item()

                    # Also record LRE predictions if possible.
                    lre_preds = None
                    if operator is not None and operator.weight is not None:
                        operator_low_rank = replace(
                            operator,
                            weight=functional.low_rank_approx(
                                matrix=operator.weight, rank=rank, svd=svd
                            ),
                        )
                        output_low_rank = operator_low_rank(
                            subject_original, h=hs_by_subj.get(subject_original)
                        )
                        lre_preds = output_low_rank.predictions

                    relation_samples.append(
                        CausalityBenchmarkRelationTrialSample(
                            subject_original=subject_original,
                            subject_target=subject_target,
                            object_target=object_target,
                            prompt_template=prompt_template,
                            prob_original=prob_original,
                            prob_target=prob_target,
                            lre_preds=lre_preds,
                            edited_lm_preds=result.predicted_tokens,
                            edited_lm_generations=result.model_generations,
                        )
                    )
                relation_ranks.append(
                    CausalityBenchmarkRelationTrialRank(
                        rank=rank,
                        samples=relation_samples,
                    )
                )
            relation_trials.append(
                CausalityBenchmarkRelationTrial(
                    train=train, test=test, ranks=relation_ranks
                )
            )
        relation_results = CausalityRelationResults(
            relation_name=relation.name, trials=relation_trials
        )
        experiment_utils.save_results_file(
            results_dir=results_dir,
            name=relation.name,
            results=relation_results,
        )
        results_by_relation.append(relation_results)

    # Also computed/evaluated elsewhere, but saving metrics during the run helps
    # with debugging.
    efficacies = [
        trial.best().efficacy()
        for relation_result in results_by_relation
        for trial in relation_result.trials
    ]
    efficacy = metrics.AggregateMetric.aggregate(efficacies)

    return CausalityBenchmarkResults(
        relations=results_by_relation,
        metrics=CausalityBenchmarkMetrics(efficacy=efficacy),
    )


# TODO(evan): Pretty close to the one in sweeps.py, should refactor.
class _PrecomputedHiddens(NamedTuple):
    hs_by_subj: dict[str, torch.Tensor]
    zs_by_subj: dict[str, torch.Tensor]


def _precompute_zs(
    *,
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subjects: StrSequence,
    h_layer: Layer | None = None,
    z_layer: Layer | None = None,
    batch_size: int = functional.DEFAULT_BATCH_SIZE,
    examples: Sequence[data.RelationSample] | None = None,
) -> _PrecomputedHiddens:
    """Precompute h for every subject at every layer."""
    prompts = [
        functional.make_prompt(
            mt=mt,
            prompt_template=prompt_template,
            subject=subject,
            examples=examples,
        )
        for subject in subjects
    ]
    with models.set_padding_side(mt, padding_side="left"):
        inputs = mt.tokenizer(
            prompts, return_tensors="pt", padding="longest", return_offsets_mapping=True
        )
    offset_mapping = inputs.pop("offset_mapping")

    batched_hidden_states = []
    for i in range(0, len(inputs.input_ids), batch_size):
        with torch.inference_mode():
            outputs = mt.model(
                inputs.input_ids[i : i + batch_size],
                attention_mask=inputs.attention_mask[i : i + batch_size],
                output_hidden_states=True,
                return_dict=True,
            )
        batched_hidden_states.append(torch.stack(outputs.hidden_states)[1:])
    hidden_states = torch.cat(batched_hidden_states, dim=1)

    zs_by_subj = {}
    hs_by_subj = {}
    for i, (subject, prompt) in enumerate(zip(subjects, prompts)):
        if h_layer is not None:
            _, h_index = tokenizer_utils.find_token_range(
                prompt, subject, offset_mapping=offset_mapping[i]
            )
            h_index -= 1
            hs_by_subj[subject] = hidden_states[h_layer, i, h_index]
        if z_layer is not None:
            zs_by_subj[subject] = hidden_states[z_layer, i, -1]

    return _PrecomputedHiddens(hs_by_subj, zs_by_subj)
