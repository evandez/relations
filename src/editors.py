"""Methods for using LRE to edit representations."""
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

from src import functional, models, operators
from src.utils import tokenizer_utils
from src.utils.typing import Layer, ModelInput

import baukit
import torch

logger = logging.getLogger(__name__)

DEFAULT_N_TOP_TOKENS = 10
DEFAULT_N_SAMPLES = 5
DEFAULT_N_NEW_TOKENS = 50


@dataclass(frozen=True, kw_only=True)
class EditResult:
    """Edited LM output."""

    predicted_tokens: list[functional.PredictedToken]
    model_logits: torch.Tensor
    model_generations: list[str]


@dataclass(frozen=True, kw_only=True)
class Editor:
    """Abstract editor which edits one subject to look like another."""

    n_top_tokens: int = DEFAULT_N_TOP_TOKENS
    n_samples: int = DEFAULT_N_SAMPLES
    n_new_tokens: int = DEFAULT_N_NEW_TOKENS

    def __call__(
        self,
        subject: str,
        target: str,
        **kwargs: Any,
    ) -> EditResult:
        raise NotImplementedError

    @staticmethod
    def expects() -> Literal["subject", "object"]:
        """Does this editor expect a target subject or target object as input?"""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LinearRelationEditResult(EditResult):
    """Outputs of a linear relation editor."""


@dataclass(frozen=True, kw_only=True)
class LinearRelationEditor(Editor):
    """Abstract editor that uses an linear relation operator to edit."""

    lre: operators.LinearRelationOperator

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

    def __call__(
        self, subject: str, target: str, **kwargs: Any
    ) -> LinearRelationEditResult:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LowRankPInvEditor(LinearRelationEditor):
    """Edit h using a low-rank pseudo-inverse of the weight matrix.

    Assumes the target is a *subject* whose object is the target value.
    """

    rank: int = 100
    svd: functional.Svd | None = None

    @cached_property
    def _low_rank_pinv(self) -> torch.Tensor:
        """Compute the pseudo-inverse of the weight matrix."""
        logger.debug(f"computing low-rank pinv (rel={self.lre.prompt_template})")
        weight = self.lre.weight
        if weight is None:
            raise AssertionError("LRE weight is None, editing does not support this")
        return functional.low_rank_pinv(matrix=weight, rank=self.rank, svd=self.svd)

    def __call__(
        self,
        subject: str,
        target: str,
        z_original: torch.Tensor | None = None,
        z_target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LinearRelationEditResult:
        _check_no_extra_kwargs(kwargs)
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
                return_offsets_mapping=True,
            ).to(self.mt.model.device)

        offset_mapping = inputs.pop("offset_mapping")
        _, subject_edit_index = tokenizer_utils.find_token_range(
            prompt_original,
            subject,
            offset_mapping=offset_mapping[0],
        )
        subject_edit_index -= 1

        if z_original is None or z_target is None:
            hiddens = functional.compute_hidden_states(
                mt=self.lre.mt,
                layers=[self.z_layer],
                inputs=inputs,
            )

            if z_original is None:
                z_original = hiddens.hiddens[0][0, -1, ..., None]
            if z_target is None:
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

    @staticmethod
    def expects() -> Literal["subject", "object"]:
        """Does this editor expect a target subject or target object as input?"""
        return "subject"


@dataclass(frozen=True, kw_only=True)
class LowRankPInvEmbedEditor(LowRankPInvEditor):
    """Edit h using a low-rank pseudo-inverse of the weight matrix.

    Assumes that `target` is the object of the relation, not another subject.
    """

    def __call__(
        self,
        subject: str,
        target: str,
        z_original: torch.Tensor | None = None,
        z_target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LinearRelationEditResult:
        _check_no_extra_kwargs(kwargs)
        inputs, subject_edit_index = _compute_inputs(
            mt=self.mt,
            prompt_template=self.prompt_template,
            subject=subject,
        )

        if z_original is None:
            hiddens = functional.compute_hidden_states(
                mt=self.mt,
                layers=[self.z_layer],
                inputs=inputs,
            )

            if z_original is None:
                z_original = hiddens.hiddens[0][0, -1, ..., None]

        # Target z is just an embedding vector.
        if z_target is None:
            target_token_id = (
                models.tokenize_words(self.mt, target).input_ids[:, 0].item()
            )
            z_target = self.mt.lm_head[-1].weight[target_token_id, ..., None]
            z_target = z_target * (z_original.norm() / z_target.norm())

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

    @staticmethod
    def expects() -> Literal["subject", "object"]:
        """Does this editor expect a target subject or target object as input?"""
        return "object"


@dataclass(frozen=True, kw_only=True)
class HiddenBaselineEditor(Editor):
    """Edit the model by replacing h for the subject with the h of the target."""

    mt: models.ModelAndTokenizer
    prompt_template: str
    h_layer: Layer

    def __call__(
        self,
        subject: str,
        target: str,
        **kwargs: Any,
    ) -> LinearRelationEditResult:
        _check_no_extra_kwargs(kwargs)
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

    @staticmethod
    def expects() -> Literal["subject", "object"]:
        """Does this editor expect a target subject or target object as input?"""
        return "subject"


@dataclass(frozen=True, kw_only=True)
class EmbedBaselineEditor(Editor):
    """Edit the model by replacing h for the object embedding."""

    prompt_template: str
    mt: models.ModelAndTokenizer
    h_layer: Layer

    def __call__(
        self,
        subject: str,
        target: str,
        **kwargs: Any,
    ) -> LinearRelationEditResult:
        _check_no_extra_kwargs(kwargs)
        inputs, subject_edit_index = _compute_inputs(
            mt=self.mt, prompt_template=self.prompt_template, subject=subject
        )

        hiddens = functional.compute_hidden_states(
            mt=self.mt, layers=[self.h_layer], inputs=inputs
        )
        h_original = hiddens.hiddens[0][0, subject_edit_index, ..., None]

        target_token_id = models.tokenize_words(self.mt, target).input_ids[:, 0].item()
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

    @staticmethod
    def expects() -> Literal["subject", "object"]:
        """Does this editor expect a target subject or target object as input?"""
        return "object"


def _check_no_extra_kwargs(kwargs: dict) -> None:
    """Check that no extra kwargs were passed."""
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs}")


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
            # NB(evan): Only ever apply edit to first input.
            input_ids=inputs.input_ids[:1].expand(n_samples, -1),
            attention_mask=inputs.attention_mask[:1].expand(n_samples, -1),
            max_new_tokens=n_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            **generate_kwargs,
        )

    model_logits = outputs.scores[0][0]
    model_generations = mt.tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )

    probs = model_logits.float().softmax(dim=-1)
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
