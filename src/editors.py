"""Methods for using LRE to edit representations."""
import logging
from dataclasses import dataclass
from functools import cached_property

from src import functional, models, operators
from src.utils import tokenizer_utils
from src.utils.typing import Layer, ModelInput

import baukit
import torch

logger = logging.getLogger(__name__)

DEFAULT_N_TOP_TOKENS = 10
DEFAULT_N_SAMPLES = 5
DEFAULT_N_NEW_TOKENS = 10


@dataclass(frozen=True, kw_only=True)
class EditResult:
    """Edited LM output."""

    predicted_tokens: list[functional.PredictedToken]
    model_logits: torch.Tensor
    model_generations: list[str]


@dataclass(frozen=True, kw_only=True)
class Editor:
    """Abstract editor which edits one subject to look like another."""

    def __call__(
        self,
        subject: str,
        target: str,
    ) -> EditResult:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LinearRelationEditResult(EditResult):
    """Outputs of a linear relation editor."""


@dataclass(frozen=True, kw_only=True)
class LinearRelationEditor(Editor):
    """Abstract editor that uses an linear relation operator to edit."""

    lre: operators.LinearRelationOperator
    n_top_tokens: int = DEFAULT_N_TOP_TOKENS
    n_samples: int = DEFAULT_N_SAMPLES
    n_new_tokens: int = DEFAULT_N_NEW_TOKENS

    @property
    def mt(self) -> models.ModelAndTokenizer:
        return self.lre.mt

    @property
    def prompt_template(self) -> str:
        return self.lre.prompt_template

    @property
    def h_layer(self) -> Layer:
        return self.lre.h_layer

    @property
    def z_layer(self) -> Layer:
        return self.lre.z_layer

    def __call__(self, subject: str, target: str) -> LinearRelationEditResult:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LowRankPInvEditor(LinearRelationEditor):
    """Edit h using a low-rank pseudo-inverse of the weight matrix.

    Assumes the target is a *subject* whose object is the target value.
    """

    rank: int = 100

    @cached_property
    def _low_rank_pinv(self) -> torch.Tensor:
        """Compute the pseudo-inverse of the weight matrix."""
        logger.debug(f"computing low-rank pinv (rel={self.lre.prompt_template})")
        weight = self.lre.weight
        if weight is None:
            raise AssertionError("LRE weight is None, editing does not support this")
        return functional.low_rank_pinv(matrix=weight, rank=self.rank)

    def __call__(
        self,
        subject: str,
        target: str,
    ) -> LinearRelationEditResult:
        prompt_original = functional.make_prompt(
            mt=self.mt, prompt_template=self.prompt_template, subject=subject
        )
        prompt_target = functional.make_prompt(
            mt=self.mt, prompt_template=self.prompt_template, subject=target
        )
        with models.set_padding_side(self.mt, padding_side="left"):
            inputs = self.mt.tokenizer(
                [prompt_original, prompt_target],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                return_offsets_mapping=True,
            ).to(self.mt.model.device)

        offset_mapping = inputs.pop("offset_mapping")
        _, subject_edit_index = tokenizer_utils.find_token_range(
            prompt_original,
            subject,
            offset_mapping=offset_mapping[0],
        )
        subject_edit_index -= 1

        hiddens = functional.compute_hidden_states(
            mt=self.lre.mt,
            layers=[self.z_layer],
            prompt=[prompt_original, prompt_target],
        )

        z_original = hiddens.hiddens[0][0, -1, ..., None]
        z_target = hiddens.hiddens[0][1, -1, ..., None]

        weight_pinv = self._low_rank_pinv
        delta = weight_pinv @ (z_target - z_original)

        return _apply_edit(
            mt=self.mt,
            layer=self.h_layer,
            index=subject_edit_index,
            inputs=inputs,
            delta=delta,
            n_top_tokens=self.n_top_tokens,
            n_new_tokens=self.n_new_tokens,
            n_samples=self.n_samples,
        )


@dataclass(frozen=True, kw_only=True)
class LowRankPInvEmbedEditor(LowRankPInvEditor):
    """Edit h using a low-rank pseudo-inverse of the weight matrix.

    Assumes that `target` is the object of the relation, not another subject.
    """

    def __call__(
        self,
        subject: str,
        target: str,
    ) -> LinearRelationEditResult:
        inputs, subject_edit_index = _compute_inputs(
            mt=self.mt,
            prompt_template=self.prompt_template,
            subject=subject,
        )

        hiddens = functional.compute_hidden_states(
            mt=self.mt,
            layers=[self.z_layer],
            inputs=inputs,
        )

        z_original = hiddens.hiddens[0][0, -1, ..., None]

        target_token_id = models.tokenize_words(self.mt, target).input_ids[:, 0].item()

        embed_target = self.mt.lm_head[-1].weight[target_token_id, ..., None]
        embed_target = embed_target * (z_original.norm() / embed_target.norm())

        weight_pinv = self._low_rank_pinv
        delta = weight_pinv @ (embed_target - z_original)

        return _apply_edit(
            mt=self.mt,
            layer=self.h_layer,
            index=subject_edit_index,
            inputs=inputs,
            delta=delta,
            n_top_tokens=self.n_top_tokens,
            n_new_tokens=self.n_new_tokens,
            n_samples=self.n_samples,
        )


@dataclass(frozen=True, kw_only=True)
class HiddenBaselineEditor(LinearRelationEditor):
    """Edit the model by replacing h for the subject with the h of the target."""

    def __call__(
        self,
        subject: str,
        target: str,
    ) -> LinearRelationEditResult:
        inputs, subject_edit_index = _compute_inputs(
            mt=self.mt,
            prompt_template=self.prompt_template,
            subject=subject,
        )

        target_inputs, target_subject_index = _compute_inputs(
            mt=self.mt,
            prompt_template=self.prompt_template,
            subject=target,
        )

        [[hiddens], *_] = functional.compute_hidden_states(
            mt=self.mt,
            layers=[self.h_layer],
            inputs=target_inputs,
        )
        h_target = hiddens[0, target_subject_index, ..., None]

        return _apply_edit(
            mt=self.mt,
            layer=self.h_layer,
            index=subject_edit_index,
            inputs=inputs,
            delta=h_target,
            assign=True,
            n_top_tokens=self.n_top_tokens,
            n_new_tokens=self.n_new_tokens,
            n_samples=self.n_samples,
        )


@dataclass(frozen=True, kw_only=True)
class EmbedBaselineEditor(LowRankPInvEditor):
    """Edit the model by replacing h for the object embedding."""

    def __call__(
        self,
        subject: str,
        target: str,
    ) -> LinearRelationEditResult:
        inputs, subject_edit_index = _compute_inputs(
            mt=self.mt, prompt_template=self.prompt_template, subject=subject
        )

        hiddens = functional.compute_hidden_states(
            mt=self.mt, layers=[self.h_layer], inputs=inputs
        )
        h_original = hiddens.hiddens[0][0, subject_edit_index, ..., None]

        target_token_id = models.tokenize_words(self.mt, target).inputs_ids[:, 0].item()
        embed_target = self.mt.lm_head[-1].weight[target_token_id, :]
        embed_target = embed_target * (h_original.norm() / embed_target.norm())

        return _apply_edit(
            mt=self.mt,
            layer=self.h_layer,
            index=subject_edit_index,
            inputs=inputs,
            delta=embed_target,
            assign=True,
            n_top_tokens=self.n_top_tokens,
            n_new_tokens=self.n_new_tokens,
            n_samples=self.n_samples,
        )


def _compute_inputs(
    *,
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subject: str,
) -> tuple[ModelInput, int]:
    """Compute model inputs and the subject token index."""
    prompt_subject = functional.make_prompt(
        mt=mt, prompt_template=prompt_template, subject=subject
    )
    inputs = mt.tokenizer(
        prompt_subject,
        return_tensors="pt",
        return_offsets_mapping=True,
    ).to(mt.model.device)
    assert len(inputs.input_ids) == 1, inputs.input_ids.shape

    offset_mapping = inputs.pop("offset_mapping")
    _, subject_index = tokenizer_utils.find_token_range(
        prompt_subject,
        subject,
        offset_mapping=offset_mapping[0],
    )
    subject_index -= 1

    return inputs, subject_index


def _apply_edit(
    *,
    mt: models.ModelAndTokenizer,
    layer: Layer,
    index: int,
    inputs: ModelInput,
    delta: torch.Tensor,
    assign: bool = False,
    n_top_tokens: int = DEFAULT_N_TOP_TOKENS,
    n_new_tokens: int = DEFAULT_N_NEW_TOKENS,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> LinearRelationEditResult:
    def edit_output(output):  # type: ignore
        h = output
        if isinstance(h, tuple):
            h = output[0]

        if h.shape[1] == 1:
            return output

        if assign:
            h[:, index] = delta.squeeze()
        else:
            h[:, index] += delta.squeeze()

        return output

    generate_kwargs = models.determine_generate_kwargs(mt)

    [layer_name] = models.determine_layer_paths(mt, layers=[layer])
    with baukit.Trace(mt.model, layer_name, edit_output=edit_output):
        outputs = mt.model.generate(
            input_ids=inputs.input_ids.expand(n_samples, -1),
            attention_mask=inputs.attention_mask.expand(n_samples, -1),
            max_new_tokens=n_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **generate_kwargs,
        )

    model_logits = outputs.scores[0]
    model_generations = mt.tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )

    probs = model_logits[0, -1].float().softmax(dim=-1)
    topk = probs.topk(k=n_top_tokens, dim=-1)
    predicted_tokens = [
        functional.PredictedToken(
            token=mt.tokenizer.decode(token_id),
            prob=prob,
        )
        for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist())
    ]

    return LinearRelationEditResult(
        predicted_tokens=predicted_tokens,
        model_logits=model_logits,
        model_generations=model_generations,
    )
