"""Tools for running sweeps over hyperparameters."""
import logging
import random
from dataclasses import dataclass
from typing import Any, Sequence

from src import data, functional, metrics, models, operators
from src.utils import tokenizer_utils
from src.utils.typing import StrSequence

import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_RECALL_K = 3
DEFAULT_N_SAMPLES = 5
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class ZsPromptTemplateResults(DataClassJsonMixin):
    prompt_template: str
    recall: float


@dataclass(frozen=True)
class SweepZsPromptTemplateResults(DataClassJsonMixin):

    results: list[ZsPromptTemplateResults]

    def best(self) -> str:
        """Return the best prompt template."""
        best = max(self.results, key=lambda x: x.recall)
        return best.prompt_template


def sweep_zs_prompt_template(
    *,
    mt: models.ModelAndTokenizer,
    relation: data.Relation,
    batch_size: int = DEFAULT_BATCH_SIZE,
    desc: str | None = None,
) -> SweepZsPromptTemplateResults:
    """Choose the best prompt template according to the LM's ZS performance."""
    if desc is None:
        desc = f"sweep prompt templates ({relation.name})"

    results = []
    progress = tqdm(relation.prompt_templates)
    for prompt_template in progress:
        progress.set_description(f"{desc} ({prompt_template})")

        prompts = [
            functional.make_prompt(
                mt=mt, prompt_template=prompt_template, subject=x.subject
            )
            for x in relation.samples
        ]
        predictions = functional.predict_next_token(
            mt=mt, prompt=prompts, k=1, batch_size=batch_size
        )
        [recall] = metrics.recall(
            [[x.token] for [x] in predictions], [x.object for x in relation.samples]
        )
        results.append(
            ZsPromptTemplateResults(prompt_template=prompt_template, recall=recall)
        )

    return SweepZsPromptTemplateResults(results=results)


@dataclass(frozen=True)
class SweepHLayerBetaResuts(DataClassJsonMixin):
    pass


def sweep_h_layer_and_beta(
    *,
    mt: models.ModelAndTokenizer,
    dataset: data.RelationDataset,
    h_layers: Sequence[int] | None = None,
    betas: Sequence[float] | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    recall_k: int = DEFAULT_RECALL_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    desc: str | None = None,
    **kwargs: Any,
) -> SweepHLayerBetaResuts:
    """Sweep over h_layer and beta together, choosing best beta for each layer."""
    if desc is None:
        desc = f"sweep h_layer/beta"
    if h_layers is None:
        h_layers = models.determine_layers(mt)
    if betas is None:
        betas = torch.linspace(0, 1, steps=11).tolist()

    for relation in dataset.relations:
        logger.info(f"begin relation: {relation.name}")

        # Determine best prompt template for ZS performance.
        prompt_template = sweep_zs_prompt_template(mt=mt, relation=relation).best()
        logger.info(f"chose prompt template: {prompt_template}")

        # Precompute all the hs to speed things up.
        hs_by_subj = _precompute_hs(
            mt=mt,
            prompt_template=prompt_template,
            subjects=[x.subject for x in relation.samples],
            batch_size=batch_size,
        )

        # Decide which will be the train samples we will try.
        train_samples = random.sample(relation.samples, k=n_samples)
        logger.info(f"sweeping for train_samples={train_samples}")

        progress = tqdm(h_layers, desc=desc)
        for h_layer in progress:
            progress.set_description(f"{desc}, h_layer={h_layer}")

            estimator = operators.JacobianEstimator(mt=mt, h_layer=h_layer, **kwargs)
            for train_sample in train_samples:
                operator = estimator(relation.set(samples=[train_sample]))
                assert operator.bias is not None
                bias = operator.bias.clone()

                test_samples = [x for x in relation.samples if x != train_sample]
                test_subjects = [x.subject for x in test_samples]
                test_hs = [hs_by_subj[x.subject][h_layer] for x in test_samples]
                test_objects = [x.object for x in test_samples]

                recalls_by_beta = []
                for beta in betas:
                    operator.bias[:] = bias * beta

                    pred_objects = []
                    for subj, h in zip(test_subjects, test_hs):
                        preds = operator(subj, h=h, k=recall_k)
                        pred_objects.append([p.token for p in preds.predictions])

                    recall = metrics.recall(pred_objects, test_objects)
                    recalls_by_beta.append(recall)

                best_i = max(range(len(recalls_by_beta)), key=lambda i: recalls_by_beta[i])
                best_beta = betas[best_i]
                best_recall = recalls_by_beta[best_i]

    return SweepHLayerBetaResuts()


def _precompute_hs(
    *,
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subjects: StrSequence,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """Precompute h for every subject at every layer."""
    prompts = [
        functional.make_prompt(
            mt=mt,
            prompt_template=prompt_template,
            subject=subject,
        )
        for subject in subjects
    ]
    inputs = mt.tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")

    batched_hidden_states = []
    for i in range(batch_size):
        with torch.inference_mode():
            outputs = mt.model(
                inputs.input_ids[i : i + batch_size],
                attention_mask=inputs.attention_mask[i : i + batch_size],
            )
        batched_hidden_states.append(torch.stack(outputs.hidden_states)[1:])
    hidden_states = torch.cat(batched_hidden_states, dim=1)

    hs_by_subj = {}
    for i, (subject, prompt) in enumerate(zip(subjects, prompts)):
        _, h_index = tokenizer_utils.find_token_range(
            prompt, subject, offset_mapping=offset_mapping[i]
        )
        h_index -= 1
        hs_by_subj[subject] = hidden_states[:, i, h_index]

    return hs_by_subj
