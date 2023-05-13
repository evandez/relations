from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

from src import data, models
from src.utils import tokenizer_utils
from src.utils.typing import Layer, ModelInput, ModelOutput, StrSequence

import baukit
import torch
from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True, kw_only=True)
class Order1ApproxOutput:
    """A first-order approximation of an LM.

    Attributes:
        weight: The weight matrix.
        bias: The bias vector.
        h: The subject hidden state.
        h_layer: The layer of h.
        h_index: The token index of h.
        z: The (true) object hidden state.
        z_layer: The layer of z.
        z_index: The token index of z.
        inputs: The LM inputs used to compute the approximation.
        logits: The LM logits, shape (batch_size, length, vocab_size).
    """

    weight: torch.Tensor
    bias: torch.Tensor

    h: torch.Tensor
    h_layer: Layer
    h_index: int

    z: torch.Tensor
    z_layer: Layer
    z_index: int

    inputs: ModelInput
    logits: torch.Tensor

    metadata: dict = field(default_factory=dict)


@torch.no_grad()
@torch.inference_mode(mode=False)
def order_1_approx(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str,
    h_layer: Layer,
    h_index: int,
    z_layer: Layer | None = None,
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
        inputs: Precomputed tokenized inputs, recomputed if not set.

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
    [h_layer_name, z_layer_name] = models.determine_layer_paths(mt, [h_layer, z_layer])
    with baukit.TraceDict(mt.model, (h_layer_name, z_layer_name)) as ret:
        outputs = mt.model(
            input_ids=input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    h = untuple(ret[h_layer_name].output)[0, _h_index]
    z = untuple(ret[z_layer_name].output)[0, z_index]

    # Now compute J and b.
    def compute_z_from_h(h: torch.Tensor) -> torch.Tensor:
        def insert_h(output: tuple, layer: str) -> tuple:
            hs = untuple(output)
            if layer != h_layer_name:
                return output
            hs[0, _h_index] = h
            return output

        with baukit.TraceDict(
            mt.model, (h_layer_name, z_layer_name), edit_output=insert_h
        ) as ret:
            mt.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return untuple(ret[z_layer_name].output)[0, -1]

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
        metadata={
            "Jh": weight @ h,
        },
    )
    return approx


def low_rank_approx(*, matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Compute a low-rank approximation of a matrix.

    Args:
        matrix: The matrix to approximate.
        rank: The rank of the approximation.

    Returns:
        The approximation.

    """
    u, s, v = torch.svd(matrix.float())
    matrix_approx = u[:, :rank] @ torch.diag(s[:rank]) @ v[:, :rank].T
    return matrix_approx.to(matrix.dtype)


def low_rank_pinv(*, matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Compute a low-rank pseudo-inverse of a matrix.

    Args:
        matrix: The matrix to invert.
        rank: The rank of the approximation.

    Returns:
        The pseudo-inverse.

    """
    u, s, v = torch.svd(matrix.float())
    matrix_pinv = v[:, :rank] @ torch.diag(1 / s[:rank]) @ u[:, :rank].T
    return matrix_pinv.to(matrix.dtype)


class CornerGdOutput(NamedTuple):
    """The output of `corner_gd`."""

    corner: torch.Tensor
    losses: list[float]

    def plot(self, ticks: int = 10) -> None:
        """Plot the loss over time."""
        import matplotlib.pyplot as plt

        plt.rcdefaults()
        plt.plot(self.losses)
        plt.xticks(range(0, len(self.losses), ticks))
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.show()


@torch.inference_mode(mode=False)
def corner_gd(
    *,
    mt: models.ModelAndTokenizer,
    words: Sequence[str],
    lr: float = 5e-2,
    weight_decay: float = 2e-2,
    n_steps: int = 100,
    target_logit_value: float = 50.0,
    init_range: tuple[float, float] = (-1.0, 1.0),
) -> CornerGdOutput:
    """Estimate a "corner" of LM rep space where words are assigned equal prob.

    Uses gradient descent to find.

    Args:
        mt: The model.
        words: The words to try to assign equal probability.
        lr: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        n_steps: Number of optimization steps.
        target_logit_value: Optimize word logits to be close to this value.
        init_range: Initialize corner uniformly in this range.

    Returns:
        Estimated corner and metadata.

    """
    device = models.determine_device(mt)
    dtype = models.determine_dtype(mt)
    hidden_size = models.determine_hidden_size(mt)
    token_ids = models.tokenize_words(mt, words).to(device).input_ids[:, 0]

    parameters_requires_grad = []
    for parameter in mt.lm_head.parameters():
        parameter.requires_grad = True
        parameters_requires_grad.append(parameter)

    z = torch.empty(hidden_size, dtype=dtype, device=device)
    z.uniform_(*init_range)
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=weight_decay)

    losses = []
    for _ in range(n_steps):
        logits = mt.lm_head(z)
        current_logits = torch.gather(logits, 0, token_ids)
        target_logits = torch.zeros_like(current_logits) + target_logit_value
        loss = (target_logits - current_logits).square().mean() + logits.mean()

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    z.requires_grad = False
    for parameter in parameters_requires_grad:
        parameter.requires_grad = False

    return CornerGdOutput(corner=z.detach(), losses=losses)


class ComputeHiddenStatesOutput(NamedTuple):
    """The output of `compute_hidden_states`."""

    hiddens: list[torch.Tensor]
    outputs: ModelOutput


# TODO(evan): Syntacic sugar for when you only want one layer,
# or don't want outputs.
@torch.no_grad()
def compute_hidden_states(
    *,
    mt: models.ModelAndTokenizer,
    layers: Sequence[Layer],
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
        outputs = mt.model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **kwargs
        )

    hiddens = []
    for layer in layers:
        h = untuple(ret[layer_paths[layer]].output)
        hiddens.append(h)

    return ComputeHiddenStatesOutput(hiddens=hiddens, outputs=outputs)


@dataclass(frozen=True, kw_only=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: float

    def __str__(self) -> str:
        return f"{self.token} (p={self.prob:.3f})"


@torch.inference_mode()
def predict_next_token(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str | StrSequence,
    k: int = 5,
    batch_size: int = 64,
) -> list[list[PredictedToken]]:
    """Compute the next token."""
    if isinstance(prompt, str):
        prompt = [prompt]
    with models.set_padding_side(mt, padding_side="left"):
        inputs = mt.tokenizer(
            prompt, return_tensors="pt", padding="longest", truncation=True
        ).to(mt.model.device)
    with torch.inference_mode():
        batched_logits = []
        for i in range(0, len(inputs.input_ids), batch_size):
            batch_outputs = mt.model(
                input_ids=inputs.input_ids[i : i + batch_size],
                attention_mask=inputs.attention_mask[i : i + batch_size],
            )
            batched_logits.append(batch_outputs.logits)
        logits = torch.cat(batched_logits, dim=0)

    next_token_probs = logits[:, -1].float().softmax(dim=-1)
    next_token_topk = next_token_probs.topk(dim=-1, k=k)

    predictions = []
    for token_ids, token_probs in zip(next_token_topk.indices, next_token_topk.values):
        predictions.append(
            [
                PredictedToken(token=mt.tokenizer.decode(token_id), prob=prob.item())
                for token_id, prob in zip(token_ids, token_probs)
            ]
        )
    return predictions


def make_prompt(
    *,
    prompt_template: str,
    subject: str,
    examples: list[data.RelationSample] | None = None,
    mt: models.ModelAndTokenizer | None = None,
) -> str:
    """Build the prompt given the template and (optionally) ICL examples."""
    prompt = prompt_template.format(subject)

    if examples is not None:
        others = [x for x in examples if x.subject != subject]
        # TODO(evan): Should consider whether prompt wants the space at the end or not.
        prompt = (
            "\n".join(
                prompt_template.format(x.subject) + f" {x.object}" for x in others
            )
            + "\n"
            + prompt
        )

    prompt = models.maybe_prefix_eos(mt, prompt)

    return prompt


def find_subject_token_index(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str,
    subject: str,
    offset: int = -1,
) -> tuple[int, ModelInput]:
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        mt.model.device
    )
    offset_mapping = inputs.pop("offset_mapping")

    subject_i, subject_j = tokenizer_utils.find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0]
    )
    subject_token_index = tokenizer_utils.offset_to_absolute_index(
        subject_i, subject_j, offset
    )

    return subject_token_index, inputs


def any_is_nontrivial_prefix(predictions: StrSequence, target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)

def random_incorrect_targets(true_targets):
    """Returns an array of the same size as true_targets where each entry is
    changed to a random (but guaranteed different) value, drawn at random from
    true_targets."""
    result = []
    for t in true_targets:
        bad = t
        while bad == t:
            bad = random.choice(true_targets)
        result.append(bad)
    return result

def get_hidden_state_at_subject(mt, prompt, subject, h_layer):
    """"Runs a single prompt in inference and reads out the hidden state at the
    last subject token for the given subject, at the specified layer."""
    h_index, inputs = find_subject_token_index(
        mt=mt, prompt=prompt, subject=subject
    )
    [[hs], _] = compute_hidden_states(
        mt=mt, layers=[h_layer], inputs=inputs
    )
    return hs[:, h_index]

def untuple(x: Any) -> Any:
    """If `x` is a tuple, return the first element."""
    if isinstance(x, tuple):
        return x[0]
    return x
