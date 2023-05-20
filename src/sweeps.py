"""Tools for running sweeps over hyperparameters."""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from src import data, functional, metrics, models, operators
from src.utils import experiment_utils, tokenizer_utils
from src.utils.typing import PathLike, StrSequence

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)

DEFAULT_RECALL_K = 3
DEFAULT_N_TRY_SAMPLES = 3
DEFAULT_N_ICL_SAMPLES = 3
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class SweepBetaResults(DataClassJsonMixin):
    beta: float
    recall: list[float]


@dataclass(frozen=True)
class SweepTrainSampleResults(DataClassJsonMixin):
    sample: data.RelationSample
    betas: list[SweepBetaResults]

    def best(self, k: int = 1) -> SweepBetaResults:
        """Return the best beta by given recall position."""
        return max(self.betas, key=lambda x: x.recall[k - 1])

    def summarize(self) -> None:
        """Sumarize results in debug logs."""
        best = self.best()
        logger.debug(
            f"sample={self.sample} | beta={best.beta:.2f} | recall@1={best.recall[0]:.2f}"
        )


@dataclass(frozen=True)
class SweepLayerResults(DataClassJsonMixin):
    layer: int
    samples: list[SweepTrainSampleResults]


@dataclass(frozen=True)
class SweepPromptResults(DataClassJsonMixin):
    prompt_template: str
    icl_samples: list[data.RelationSample]
    train_samples: list[data.RelationSample]
    layers: list[SweepLayerResults]


@dataclass(frozen=True)
class SweepRelationResults(DataClassJsonMixin):
    relation_name: str
    prompts: list[SweepPromptResults]

    # TODO(evan): Generalize this a bit, just debugging for now.
    def summarize(self) -> None:
        """Print a summary of what happened."""
        results_by_layer = defaultdict(list)
        for prompt in self.prompts:
            for layer in prompt.layers:
                for sample in layer.samples:
                    best = sample.best()
                    results_by_layer[layer.layer].append(
                        (
                            layer.layer,
                            best.beta,
                            best.recall[0],
                        )
                    )

        scores_by_layer = {
            layer: np.mean([x[-1] for x in results])
            for layer, results in results_by_layer.items()
        }
        betas_by_layer = {
            layer: np.mean([x[1] for x in results])
            for layer, results in results_by_layer.items()
        }
        logger.debug(f'summarizing results for "{self.relation_name}"')
        for la in scores_by_layer:
            score = scores_by_layer[la]
            beta = betas_by_layer[la]
            logger.debug(f"layer={la} | beta={beta:.2f} | recall@1={score:.2f}")


@dataclass(frozen=True)
class SweepResuts(DataClassJsonMixin):
    relations: list[SweepRelationResults]


def sweep(
    *,
    mt: models.ModelAndTokenizer,
    dataset: data.RelationDataset,
    h_layers: Sequence[int] | None = None,
    betas: Sequence[float] | None = None,
    n_try_samples: int = DEFAULT_N_TRY_SAMPLES,
    n_icl_samples: int = DEFAULT_N_ICL_SAMPLES,
    recall_k: int = DEFAULT_RECALL_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    results_dir: PathLike | None = None,
    resume: bool = False,
    desc: str | None = None,
    **kwargs: Any,
) -> SweepResuts:
    """Sweep over hyperparameters for faithfulness."""
    if desc is None:
        desc = f"sweep"
    if h_layers is None:
        h_layers = models.determine_layers(mt)
    if betas is None:
        betas = torch.linspace(0, 1, steps=11).tolist()

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

        prompt_results = []
        for prompt_template in relation.prompt_templates:
            logger.info(f"begin prompt template: {prompt_template}")

            if len(relation.samples) <= n_try_samples + n_icl_samples:
                logger.warning(
                    f"Not enough samples ({len(relation.samples)}) to "
                    f'test for "{relation.name} since n_try_samples={n_try_samples} and '
                    f"n_icl_samples={n_icl_samples}. You should fix this by adding more "
                    "known samples for the relation."
                )
                continue

            # Decide which will be the train samples we will try, and which will be the
            # ICL prompt examples.
            train_relation, _ = relation.split(n_try_samples + n_icl_samples)
            train_samples = train_relation.samples
            train_icl_samples = train_samples[:n_icl_samples]
            train_try_samples = train_samples[
                n_icl_samples : n_icl_samples + n_try_samples
            ]

            logger.info(f"will do icl using: {[str(x) for x in train_icl_samples]}")
            logger.info(f"will try: {[x.subject for x in train_try_samples]}")

            # Precompute all the hs to speed things up.
            hs_by_subj = _precompute_hs(
                mt=mt,
                prompt_template=prompt_template,
                subjects=[x.subject for x in relation.samples],
                batch_size=batch_size,
                examples=train_icl_samples,
            )

            layer_results = []
            for h_layer in h_layers:
                logger.info(f"begin layer: {h_layer}")

                estimator = operators.JacobianIclEstimator(
                    mt=mt, h_layer=h_layer, **kwargs
                )

                train_sample_results = []
                for train_sample in train_try_samples:
                    operator = estimator(
                        relation.set(
                            samples=[train_sample, *train_icl_samples],
                            prompt_templates=[prompt_template],
                        )
                    )
                    assert operator.bias is not None
                    bias = operator.bias.clone()

                    test_samples = [
                        x
                        for x in relation.samples
                        if x != train_sample and x not in train_icl_samples
                    ]
                    test_subjects = [x.subject for x in test_samples]
                    test_hs = [
                        hs_by_subj[x.subject][h_layer, None] for x in test_samples
                    ]
                    test_objects = [x.object for x in test_samples]

                    results_by_beta = []
                    recalls_by_beta = []
                    for beta in betas:
                        operator.bias[:] = bias * beta

                        pred_objects = []
                        for subj, h in zip(test_subjects, test_hs):
                            preds = operator(subj, h=h, k=recall_k)
                            pred_objects.append([p.token for p in preds.predictions])

                        recall = metrics.recall(pred_objects, test_objects)
                        recalls_by_beta.append(recall)
                        results_by_beta.append(
                            SweepBetaResults(beta=beta, recall=recall)
                        )

                    train_sample_result = SweepTrainSampleResults(
                        sample=train_sample, betas=results_by_beta
                    )
                    train_sample_result.summarize()
                    train_sample_results.append(train_sample_result)
                layer_results.append(
                    SweepLayerResults(layer=h_layer, samples=train_sample_results)
                )
            prompt_results.append(
                SweepPromptResults(
                    prompt_template=prompt_template,
                    icl_samples=train_icl_samples,
                    train_samples=train_try_samples,
                    layers=layer_results,
                )
            )
        relation_result = SweepRelationResults(
            relation_name=relation.name, prompts=prompt_results
        )
        relation_result.summarize()
        experiment_utils.save_results_file(
            results_dir=results_dir,
            results=relation_result,
            name=relation.name,
        )
        relation_results.append(relation_result)
    return SweepResuts(relation_results)


def _precompute_hs(
    *,
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subjects: StrSequence,
    examples: Sequence[data.RelationSample] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, torch.Tensor]:
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
    inputs = mt.tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        return_offsets_mapping=True,
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

    hs_by_subj = {}
    for i, (subject, prompt) in enumerate(zip(subjects, prompts)):
        _, h_index = tokenizer_utils.find_token_range(
            prompt, subject, offset_mapping=offset_mapping[i]
        )
        h_index -= 1
        hs_by_subj[subject] = hidden_states[:, i, h_index]

    return hs_by_subj
