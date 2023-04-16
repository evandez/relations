import logging
import random
from collections import defaultdict
from dataclasses import dataclass

from src import data, functional, operators

import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkResults:
    frac_correct: float
    frac_dist_subj: float
    frac_dist_rel: float


@torch.inference_mode()
def reconstruction(
    estimator: operators.Estimator,
    dataset: data.RelationDataset,
    desc: str | None = None,
) -> ReconstructionBenchmarkResults:
    if desc is None:
        desc = "recon. score"

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
    for (relation_name, prompt_template, sample), operator in tqdm(
        operators.items(), desc=f"{desc} [compute scores]"
    ):
        z_true = functional.compute_hidden_states(
            mt=estimator.mt,
            layers=[operator.z_layer],
            prompt=prompt_template.format(sample.subject),
        ).hiddens[0][0, -1]

        key = random.choice(
            [
                (r, p, s)
                for r, p, s in operators
                if r == relation_name and (p != prompt_template or s != sample.subject)
            ]
        )
        operator = operators[key]
        z_pred = operator(sample.subject).z

        # Distractor 1: same subject, different relation
        matches = [
            (r, p, s)
            for r, p, s in operators
            if r == relation_name and s != sample.subject
        ]
        if not matches:
            logger.debug(
                f"skipped {relation_name}/{prompt_template}/{sample.subject} "
                "because no other relations have this subject"
            )
            continue
        (_, other_prompt_template, other_sample) = random.choice(matches)
        z_dist_subj = functional.compute_hidden_states(
            mt=estimator.mt,
            layers=[operator.z_layer],
            prompt=other_prompt_template.format(other_sample.subject),
        ).hiddens[0][0, -1]

        # Distractor 2: same relation, different subject
        matches = [
            (r, p, s)
            for r, p, s in operators
            if r == relation_name and s != sample.subject
        ]
        if not matches:
            logger.debug(
                f"skipped {relation_name}/{prompt_template}/{sample.subject} "
                "because no other subjects have this relation"
            )
            continue
        (_, other_prompt_template, other_sample) = random.choice(matches)
        z_dist_rel = functional.compute_hidden_states(
            mt=estimator.mt,
            layers=[operator.z_layer],
            prompt=other_prompt_template.format(other_sample.subject),
        ).hiddens[0][0, -1]

        zs = torch.cat([z_true, z_dist_subj, z_dist_rel], dim=0).float()
        z_pred = z_pred.float()
        dists = z_pred.mul(zs).sum(dim=-1) / (
            z_true.norm(dim=-1).expand(3) * zs.norm(dim=-1)
        )
        chosen = dists.argmin().item()
        counts[chosen] += 1

    return ReconstructionBenchmarkResults(
        frac_correct=counts[0] / sum(counts.values()),
        frac_dist_subj=counts[1] / sum(counts.values()),
        frac_dist_rel=counts[2] / sum(counts.values()),
    )
