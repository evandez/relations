"""Methods for using LRE to edit representations."""
import logging
from dataclasses import dataclass
from functools import cache

from src import functional, models, operators
from src.utils import tokenizer_utils

import baukit
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class EditResult:
    """Edited LM output."""

    predictions: list[functional.PredictedToken]


@dataclass(frozen=True, kw_only=True)
class Editor:
    """Abstract editor which edits one subject to look like another."""

    def __call__(
        self,
        subject_to_edit: str,
        subject_target: str,
    ) -> EditResult:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LowRankPInvEditor(Editor):
    """Edit h using a low-rank pseudo-inverse of the weight matrix."""

    mt: models.ModelAndTokenizer
    lre: operators.LinearRelationOperator
    rank: int

    @cache
    def _low_rank_pinv(self) -> torch.Tensor:
        """Compute the pseudo-inverse of the weight matrix."""
        logger.debug(
            f"computing low-rank pseudo-inverse (rel={self.lre.prompt_template})"
        )

        weight = self.lre.weight
        assert weight is not None
        weight = weight.float()

        u, s, v = torch.svd(weight)

        weight_inv = (
            v[:, : self.rank] @ torch.diag(1 / s[: self.rank]) @ u[:, : self.rank].T
        )

        return weight_inv.to(weight.dtype)

    def _bias(self) -> torch.Tensor:
        bias = self.lre.bias
        assert bias is not None
        return bias.T

    def __call__(
        self,
        subject_to_edit: str,
        subject_target: str,
    ) -> EditResult:
        mt = self.lre.mt
        h_layer = self.lre.h_layer
        z_layer = self.lre.z_layer
        prompt_template = self.lre.prompt_template

        prompt_to_edit = functional.make_prompt(
            mt=mt, prompt_template=prompt_template, subject=subject_to_edit
        )
        prompt_target = functional.make_prompt(
            mt=mt, prompt_template=prompt_template, subject=subject_target
        )
        with models.set_padding_side(self.lre.mt, padding_side="left"):
            inputs = self.mt.tokenizer(
                [prompt_to_edit, prompt_target],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                return_offsets_mapping=True,
            )

        offset_mapping = inputs.pop("offset_mapping")
        _, subject_edit_index = tokenizer_utils.find_token_range(
            prompt_to_edit,
            subject_to_edit,
            offset_mapping=offset_mapping[0],
        )
        subject_edit_index -= 1

        hiddens = functional.compute_hidden_states(
            mt=self.lre.mt,
            layers=[z_layer],
            prompt=[prompt_to_edit, prompt_target],
        )

        z_original = hiddens.hiddens[0][0, -1, ..., None]
        z_target = hiddens.hiddens[0][1, -1, ..., None]

        weight_inv = self._low_rank_pinv()
        bias = self._bias()
        delta = weight_inv @ (z_target - z_original - bias)

        def edit_output(output):  # type: ignore
            if output[0].shape[1] == 1:
                return output
            output[:, subject_edit_index] += delta.squeeze()
            return output

        [h_layer_name] = models.determine_layer_paths(mt, layers=[h_layer])
        with baukit.Trace(mt.model, h_layer_name, edit_output=edit_output):
            outputs = mt.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

        probs = outputs.logits[:, -1].float().softmax(dim=-1)
        topk = probs.topk(k=5, dim=-1)
        return EditResult(
            predictions=[
                functional.PredictedToken(
                    token=mt.tokenizer.decode(token_id),
                    prob=prob,
                )
                for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist())
            ]
        )
