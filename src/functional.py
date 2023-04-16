from typing import Any, NamedTuple, Sequence

from src import models
from src.utils.typing import ModelInput, ModelOutput, StrSequence

import baukit
import torch


class Order1ApproxOutput(NamedTuple):
    """A first-order approximation of an LM."""

    weight: torch.Tensor
    bias: torch.Tensor

    h: torch.Tensor
    h_layer: int
    h_index: int

    z: torch.Tensor
    z_layer: int
    z_index: int

    inputs: ModelInput
    logits: torch.Tensor


@torch.no_grad()
def order_1_approx(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str,
    h_layer: int,
    h_index: int,
    z_layer: int | None = None,
    z_index: int | None = None,
    inputs: ModelInput | None = None,
) -> Order1ApproxOutput:
    """Compute a first-order approximation of the LM between `h` and `z`.

    Very simply, this computes the Jacobian of z with respect to h, as well as
    z - Jh to approximate the bias.

    Args:
        mt: The model.
        prompt: Prompt to approximate.
        h_layer: Layer to take h from.
        h_index: Token index for h.
        z_layer: Layer to take z from.
        z_index: Token index for z.
        inputs: Precomputed tokenized inputs.

    Returns:
        The approximation.

    """
    if z_layer is None:
        z_layer = mt.model.config.n_layer - 1
    if z_index is None:
        z_index = -1
    if inputs is None:
        inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)

    # Precompute everything up to the subject, if there is anything before it.
    past_key_values = None
    input_ids = inputs.input_ids
    _h_index = h_index
    if _h_index > 0:
        outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)
        past_key_values = outputs.past_key_values
        input_ids = input_ids[:, _h_index:]
        _h_index = 0
    use_cache = past_key_values is not None

    # Precompute initial h and z.
    h_layer_name = f"transformer.h.{h_layer}"
    z_layer_name = f"transformer.h.{z_layer}"
    with baukit.TraceDict(mt.model, (h_layer_name, z_layer_name)) as ret:
        outputs = mt.model(
            input_ids=input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    h = ret[h_layer_name].output[0][0, _h_index]
    z = ret[z_layer_name].output[0][0, z_index]

    # Now compute J and b.
    def compute_z_from_h(h: torch.Tensor) -> torch.Tensor:
        def insert_h(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            output[0][0, _h_index] = h
            return output

        with baukit.TraceDict(
            mt.model, (h_layer_name, z_layer_name), edit_output=insert_h
        ) as ret:
            mt.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return ret[z_layer_name].output[0][0, -1]

    weight = torch.autograd.functional.jacobian(compute_z_from_h, h, vectorize=True)
    bias = z[None] - h[None].mm(weight.t())
    approx = Order1ApproxOutput(
        h=h,
        h_layer=h_layer,
        h_index=h_index,
        z=z,
        z_layer=z_layer,
        z_index=z_index,
        weight=weight,
        bias=bias,
        inputs=inputs.to("cpu"),
        logits=outputs.logits.cpu(),
    )
    return approx


class ComputeHiddenStatesOutput(NamedTuple):
    """The output of `compute_hidden_states`."""

    hiddens: list[torch.Tensor]
    outputs: ModelOutput


@torch.no_grad()
def compute_hidden_states(
    *,
    mt: models.ModelAndTokenizer,
    layers: Sequence[int],
    prompt: str | StrSequence | None = None,
    inputs: ModelInput | None = None,
    **kwargs: Any,
) -> ComputeHiddenStatesOutput:
    """Compute the hidden states for a given prompt.

    Args:
        mt: The model.
        layers: The layers to grab hidden states for.
        prompt: The prompt. Can alternatively pass tokenized `inputs`.
        inputs: Precomputed tokenized inputs. Can alternatively pass `prompt`.

    Returns:
        The hidden states and the model output.

    """
    if (prompt is None) == (inputs is None):
        raise ValueError("Must pass either `prompt` or `inputs`, not both.")

    if inputs is None:
        assert prompt is not None
        inputs = mt.tokenizer(
            prompt, return_tensors="pt", padding="longest", truncation=True
        ).to(mt.model.device)

    layer_paths = models.determine_layer_paths(mt, layers=layers, return_dict=True)
    with baukit.TraceDict(mt.model, layer_paths.values()) as ret:
        outputs = mt.model(**inputs, **kwargs)

    hiddens = [ret[layer_paths[layer]].output[0] for layer in layers]

    return ComputeHiddenStatesOutput(hiddens=hiddens, outputs=outputs)


class ComputeHZOutput(NamedTuple):
    """The output of `compute_h_z`."""

    h: torch.Tensor
    z: torch.Tensor


def compute_h_z(
    *,
    mt: models.ModelAndTokenizer,
    h_layer: int,
    h_index: int,
    z_layer: int,
    z_index: int,
    prompt: str | None = None,
    inputs: ModelInput | None = None,
    **kwargs: Any,
) -> ComputeHZOutput:
    [[hs, zs], _] = compute_hidden_states(
        mt=mt,
        layers=[h_layer, z_layer],
        prompt=prompt,
        inputs=inputs,
        **kwargs,
    )
    h = hs[0, h_index]
    z = zs[0, z_index]
    return ComputeHZOutput(h=h, z=z)
