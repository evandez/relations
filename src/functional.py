import copy
import gc
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Sequence

from src import data, models
from src.utils import tokenizer_utils
from src.utils.typing import Layer, Mamba, ModelInput, ModelOutput, StrSequence

import baukit
import torch
from dataclasses_json import DataClassJsonMixin
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 48  # Reduced to 48 to fit in A6000
DEFAULT_N_ICL_LM = 5
DEFAULT_N_TOP_LM = 1


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
    h: torch.Tensor | None = None,
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
        h: will calculate approximation based on this hidden state, if provided.
        z_layer: Layer to take z from.
        z_index: Token index for z.
        inputs: Precomputed tokenized inputs, recomputed if not set.

    Returns:
        The approximation.

    """
    if z_layer is None:
        z_layer = models.determine_layers(mt)[-1]
    if z_index is None:
        z_index = -1
    if inputs is None:
        inputs = mt.tokenizer(prompt, return_tensors="pt").to(
            models.determine_device(mt)
        )

    # Precompute everything up to the subject, if there is anything before it.
    past_key_values = None
    input_ids = inputs.input_ids
    _h_index = h_index
    if isinstance(mt.model, Mamba) == False:
        if _h_index > 0:
            outputs = mt(input_ids=input_ids[:, :_h_index], use_cache=True)
            past_key_values = outputs.past_key_values
            input_ids = input_ids[:, _h_index:]
            _h_index = 0
    use_cache = past_key_values is not None

    # Precompute initial h and z.
    [h_layer_name, z_layer_name] = models.determine_layer_paths(mt, [h_layer, z_layer])

    edit_output: function | None = None
    if h is not None:

        def edit_output(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            untuple(output)[:, _h_index] = h
            return output

    else:
        edit_output = None

    with baukit.TraceDict(
        mt.model, layers=(h_layer_name, z_layer_name), edit_output=edit_output
    ) as ret:
        outputs = mt(
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
            mt(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return untuple(ret[z_layer_name].output)[0, -1]

    def calculate_jacobian(function, h):
        h = h.to(models.determine_device(mt))
        print(f"{h.shape=} | {h.requires_grad=}")
        # h.requires_grad = True
        h.retain_grad()
        z_est = function(h)
        jacobian = []
        print("Calculating Jacobians ...")
        for idx in tqdm(range(h.shape[0])):
            mt.model.zero_grad()
            z_est[idx].backward(retain_graph=True)
            jacobian.append(copy.deepcopy(h.grad))
            h.grad.zero_()
        return torch.stack(jacobian)

    assert h is not None

    if isinstance(mt.model, Mamba):
        weight = calculate_jacobian(compute_z_from_h, h)
    else:
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
        logits=outputs.logits.cpu() if hasattr(outputs, "logits") else outputs.cpu(),
        metadata={
            "Jh": (weight @ h).detach().cpu(),
        },
    )

    # NB(evan): Something about the jacobian computation causes a lot of memory
    # fragmentation, or some kind of memory leak. This seems to help.
    torch.cuda.empty_cache()

    return approx


Svd = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def low_rank_approx(
    *, matrix: torch.Tensor, rank: int, svd: Svd | None = None
) -> torch.Tensor:
    """Compute a low-rank approximation of a matrix.

    Args:
        matrix: The matrix to approximate.
        rank: The rank of the approximation.

    Returns:
        The approximation.

    """
    if svd is None:
        svd = torch.svd(matrix.float())
    u, s, v = svd
    matrix_approx = u[:, :rank] @ torch.diag(s[:rank]) @ v[:, :rank].T
    return matrix_approx.to(matrix.dtype)


def low_rank_pinv(
    *, matrix: torch.Tensor, rank: int, svd: Svd | None = None
) -> torch.Tensor:
    """Compute a low-rank pseudo-inverse of a matrix.

    Args:
        matrix: The matrix to invert.
        rank: The rank of the approximation.

    Returns:
        The pseudo-inverse.

    """
    if svd is None:
        svd = torch.svd(matrix.float())
    u, s, v = svd
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
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(
            models.determine_device(mt)
        )

    layer_paths = models.determine_layer_paths(mt, layers=layers, return_dict=True)
    with baukit.TraceDict(mt.model, layer_paths.values()) as ret:
        outputs = mt(
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
        return f"'{self.token}' (p={self.prob:.3f})"


@torch.inference_mode()
def predict_next_token(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str | StrSequence,
    k: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[PredictedToken]]:
    """Compute the next token."""
    if isinstance(prompt, str):
        prompt = [prompt]
    with models.set_padding_side(mt, padding_side="left"):
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(
            models.determine_device(mt)
        )
    with torch.inference_mode():
        predictions = []
        for i in range(0, len(inputs.input_ids), batch_size):
            batch_outputs = mt(
                input_ids=inputs.input_ids[i : i + batch_size],
                attention_mask=inputs.attention_mask[i : i + batch_size],
            )

            batch_logits = (
                batch_outputs.logits[:, -1]
                if hasattr(batch_outputs, "logits")
                else batch_outputs[:, -1]
            )
            next_token_probs = batch_logits.float().softmax(dim=-1)
            next_token_topk = next_token_probs.topk(dim=-1, k=k)

            for token_ids, token_probs in zip(
                next_token_topk.indices, next_token_topk.values
            ):
                predictions.append(
                    [
                        PredictedToken(
                            token=mt.tokenizer.decode(token_id), prob=prob.item()
                        )
                        for token_id, prob in zip(token_ids, token_probs)
                    ]
                )
    return predictions


def make_prompt(
    *,
    prompt_template: str,
    subject: str,
    examples: Sequence[data.RelationSample] | None = None,
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


@torch.inference_mode()
def filter_relation_samples(
    *,
    mt: models.ModelAndTokenizer,
    relation: data.Relation,
    prompt_template: str,
    n_icl_lm: int = DEFAULT_N_ICL_LM,
    n_top_lm: int = DEFAULT_N_TOP_LM,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> data.Relation:
    """Filter samples down to only those that model knows.

    Most benchmarks rely on model knowing the relation at all. We say the model
    "knows" the sample if, given an ICL prompt for the relation, it predicts the
    correct answer in the top-1 position.
    """
    logger.debug(f'filtering for knowns using prompt "{prompt_template}"')
    prompts = []
    for sample in relation.samples:
        examples, _ = relation.without(sample).split(n_icl_lm)
        prompt = make_prompt(
            prompt_template=prompt_template,
            mt=mt,
            subject=sample.subject,
            examples=examples.samples,
        )
        prompts.append(prompt)
    predictions = predict_next_token(
        mt=mt, prompt=prompts, k=n_top_lm, batch_size=batch_size
    )

    # Helpful to see what the model predicted sometimes.
    for sample, topk in zip(relation.samples, predictions):
        logger.debug(f"{sample.subject=}, {sample.object=}, predicted={topk[0]}")

    known_samples = {
        sample
        for sample, topk in zip(relation.samples, predictions)
        if any_is_nontrivial_prefix([x.token for x in topk], sample.object)
    }

    # NB(evan): Need to sort to keep deterministic.
    return relation.set(samples=sorted(known_samples, key=lambda x: x.subject))


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


@torch.inference_mode()
def filter_relation_samples_based_on_provided_fewshots(
    *,
    mt: models.ModelAndTokenizer,
    test_relation: data.Relation,
    prompt_template: str,
    n_top_lm: int = DEFAULT_N_TOP_LM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    examples: Sequence[data.RelationSample] = [],
    subj_token_filter: Literal["all", "single", "multi"] = "all",
) -> data.Relation:
    """Filter samples down to only those that model knows.

    Most benchmarks rely on model knowing the relation at all. We say the model
    "knows" the sample if, given an ICL prompt for the relation, it predicts the
    correct answer in the top-1 position.
    """
    if len(examples) > 0:
        logger.debug("ICL examples: ", [str(sample) for sample in examples])
        prompt_template = make_prompt(
            mt=mt,
            prompt_template=prompt_template,
            subject="{}",
            examples=examples,
        )
    logger.debug(f'filtering for knowns using prompt "{prompt_template}"')

    test_prompts = [
        prompt_template.format(sample.subject) for sample in test_relation.samples
    ]
    predictions = predict_next_token(
        mt=mt, prompt=test_prompts, k=n_top_lm, batch_size=batch_size
    )

    # Helpful to see what the model predicted sometimes.
    filtered_samples = []
    for sample, prediction in zip(test_relation.samples, predictions):
        known_flag = is_nontrivial_prefix(
            prediction=prediction[0].token, target=sample.object
        )
        log_print = f"{sample.subject=}, {sample.object=}, predicted={prediction[0]}, known=({get_tick_marker(known_flag)})"
        if known_flag:
            if subj_token_filter == "all":
                filtered_samples.append(sample)
            else:
                require_multi = subj_token_filter == "multi"
                subj_single_token = (
                    models.tokenize_words(mt.tokenizer, sample.subject, spaces=True)
                    .input_ids[0]
                    .shape[0]
                    == 1
                )
                subj_token_flag = require_multi != subj_single_token
                log_print += (
                    f", {subj_token_filter}=({get_tick_marker(subj_token_flag)})"
                )
                if subj_token_flag:
                    filtered_samples.append(sample)
        logger.debug(log_print)

    return test_relation.set(samples=sorted(filtered_samples, key=lambda x: x.subject))


@torch.inference_mode()
def filter_dataset_samples(
    *,
    mt: models.ModelAndTokenizer,
    dataset: data.RelationDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_icl_lm: int = DEFAULT_N_ICL_LM,
    n_top_lm: int = DEFAULT_N_TOP_LM,
    n_trials: int = 3,
    min_knowns: int = 10,
    common_prompt_template: str | None = None,
    n_subj_tokens: Literal["single", "multi"] | None = None,
) -> data.RelationDataset:
    """Filter samples down to only those that model knows.

    Most benchmarks rely on model knowing the relation at all. We say the model
    "knows" the sample if, given an ICL prompt for the relation, it predicts the
    correct answer in the top-1 position.
    """
    logger.info("filtering dataset to knowns only...")

    if common_prompt_template is not None:
        assert (
            "{}" in common_prompt_template
        ), "common_prompt_template must contain {} to be filled with subject"

    relations = []
    for relation in dataset.relations:
        logger.debug(f"filtering samples for relation {relation.name}...")
        if common_prompt_template is not None:
            prompt_template = common_prompt_template
        else:
            prompt_template = relation.prompt_templates[0]

        counts: dict[data.RelationSample, int] = defaultdict(int)
        for _ in range(n_trials):
            filtered = filter_relation_samples(
                mt=mt,
                relation=relation,
                prompt_template=prompt_template,
                n_icl_lm=n_icl_lm,
                n_top_lm=n_top_lm,
                batch_size=batch_size,
            )
            for sample in filtered.samples:
                counts[sample] += 1

        known_samples = []
        for sample, count in counts.items():
            if count != n_trials:
                logger.debug(f"filtered out unknown sample: {sample}")
                continue
            known_samples.append(sample)

        if n_subj_tokens is None:
            filtered_relation = relation.set(samples=known_samples)
        else:
            subject_filtered_samples = []
            require_multi = n_subj_tokens == "multi"
            for sample in relation.samples:
                subj_single_token = (
                    models.tokenize_words(mt.tokenizer, sample.subject, spaces=True)
                    .input_ids[0]
                    .shape[0]
                    == 1
                )
                if require_multi != subj_single_token:
                    subject_filtered_samples.append(sample)
            filtered_relation = relation.set(samples=subject_filtered_samples)

        if "cuda" in str(models.determine_device(mt)):
            logger.debug(
                f"clearing cuda cache after filtering samples for -> {relation.name}"
            )
            torch.cuda.empty_cache()
            gc.collect()

        if len(filtered_relation.samples) < min_knowns:
            logger.debug(
                f'not enough known samples for relation "{relation.name}" '
                f"({len(known_samples)} < {min_knowns})"
            )
            continue
        relations.append(filtered_relation)

    return data.RelationDataset(relations)


def find_subject_token_index(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str,
    subject: str,
    offset: int = -1,
) -> tuple[int, ModelInput]:
    """Determine index of a specific subject token in prompt."""
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        models.determine_device(mt)
    )
    offset_mapping = inputs.pop("offset_mapping")
    if "token_type_ids" in inputs:  # llama tokenizer has this annoying field
        inputs.pop("token_type_ids")
    # Find the last occurrence of the subject
    subject_i, subject_j = tokenizer_utils.find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0], occurrence=-1
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


def random_incorrect_targets(true_targets: list[str]) -> list[str]:
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


def random_edit_targets(
    samples: list[data.RelationSample],
) -> dict[data.RelationSample, data.RelationSample]:
    """Pick random edit targets for each of the given samples.

    If there are no other samples with different subject and different object,
    then the sample is skipped.
    """
    targets = {}
    for sample in samples:
        others = [
            x
            for x in samples
            if x.subject != sample.subject and x.object != sample.object
        ]
        if not others:
            logger.debug(f"no valid edit target for {sample}, skipping")
            continue
        targets[sample] = random.choice(others)
    return targets


def compute_h(
    mt: models.ModelAndTokenizer, prompt: str, subject: str, h_layer: Layer
) -> torch.Tensor:
    """Runs a single prompt in inference and reads out the hidden state at the
    last subject token for the given subject, at the specified layer."""
    h_index, inputs = find_subject_token_index(mt=mt, prompt=prompt, subject=subject)
    [[hs], _] = compute_hidden_states(mt=mt, layers=[h_layer], inputs=inputs)
    return hs[:, h_index]


class HZBySubject(NamedTuple):
    """Subject h and z vectors, potentially from multiple different layers.

    Dict keys are subjects, values are either single layer tensor (if only one layer
    is specified) or dict of layer -> tensor (if multiple layers are specified).
    If h_layer/z_layer was None, dict will be empty.
    """

    h_by_subj: dict
    z_by_subj: dict


def compute_hs_and_zs(
    *,
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subjects: StrSequence,
    h_layer: Layer | Sequence[Layer] | None = None,
    z_layer: Layer | Sequence[Layer] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    examples: Sequence[data.RelationSample] | None = None,
) -> HZBySubject:
    """Precompute h for every subject at every layer."""
    if h_layer is None and z_layer is None:
        raise ValueError("Must specify at least one of h_layer and z_layer.")
    if z_layer == -1 or z_layer is None:
        z_layer = models.determine_layers(mt)[-1]

    prompts = [
        make_prompt(
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
        ).to(models.determine_device(mt))
    offset_mapping = inputs.pop("offset_mapping")

    z_by_subj = {}
    h_by_subj = {}

    h_layers = [h_layer] if (isinstance(h_layer, int) or h_layer == "emb") else h_layer
    z_layers = [z_layer] if (isinstance(z_layer, int) or z_layer == "ln_f") else z_layer

    assert isinstance(h_layers, list)
    assert isinstance(z_layers, list)
    layer_idx_to_name = {
        l: models.determine_layer_paths(mt, [l])[0] for l in h_layers + z_layers
    }

    for batch_start in range(0, len(inputs.input_ids), batch_size):
        with torch.no_grad():
            with baukit.TraceDict(
                mt.model, layers=layer_idx_to_name.values()
            ) as traces:
                outputs = mt(
                    inputs.input_ids[batch_start : batch_start + batch_size],
                    attention_mask=inputs.attention_mask[
                        batch_start : batch_start + batch_size
                    ],
                )

        for batch_index in range(batch_size):
            abs_index = batch_start + batch_index
            if abs_index >= len(inputs.input_ids):
                break
            subject = subjects[abs_index]

            if h_layer is not None:
                prompt = prompts[abs_index]
                _, h_index = tokenizer_utils.find_token_range(
                    prompt, subject, offset_mapping=offset_mapping[abs_index]
                )
                h_index -= 1
                if isinstance(h_layer, int) or h_layer == "emb":
                    h_by_subj[subject] = untuple(
                        traces[layer_idx_to_name[h_layer]].output
                    )[batch_index, h_index]
                else:
                    h_by_subj[subject] = {
                        hl: untuple(traces[layer_idx_to_name[hl]].output)[
                            batch_index, h_index
                        ]
                        for hl in h_layer
                    }

            if z_layer is not None:
                if isinstance(z_layer, int):
                    z_by_subj[subject] = untuple(
                        traces[layer_idx_to_name[z_layer]].output
                    )[batch_index, -1]
                else:
                    z_by_subj[subject] = {
                        zl: untuple(traces[layer_idx_to_name[zl]].output)[
                            batch_index, -1
                        ]
                        for zl in z_layer
                    }

    return HZBySubject(h_by_subj, z_by_subj)


def untuple(x: Any) -> Any:
    """If `x` is a tuple, return the first element."""
    if isinstance(x, tuple):
        return x[0]
    return x


from dataclasses import dataclass
from typing import Optional

from src.editors import LinearRelationEditResult
from src.functional import PredictedToken
from src.models import ModelAndTokenizer
from src.utils.typing import Layer


@dataclass
class EditConfig:
    layers: list[Layer]
    intervention: callable


# custom generate function for mamba
@torch.inference_mode()
def mamba_generate(
    mt: ModelAndTokenizer,
    prompt: Optional[list[str] | str] = None,
    input_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 10,
    topk: int = 5,
    edit_config: Optional[EditConfig] = None,
):
    assert prompt is not None or input_ids is not None
    if isinstance(prompt, str):
        prompt = [prompt]
    if input_ids is None:
        with models.set_padding_side(mt, padding_side="left"):
            input_ids = (
                mt.tokenizer(prompt, return_tensors="pt", padding="longest")
                .to(models.determine_device(mt))
                .input_ids
            )

    predicted_tokens: list[PredictedToken] = []
    model_logits: torch.Tensor | None = None
    generated_tokens: list[list[int]] = [[] for _ in range(len(input_ids))]

    for i in range(max_new_tokens):
        if i == 0 and edit_config is not None:
            with baukit.Trace(
                module=mt.model,
                layer=edit_config.layers[0],
                edit_output=edit_config.intervention,
            ):
                outputs = mt(input_ids=input_ids)
        else:
            outputs = mt(input_ids=input_ids)

        logits = (
            outputs.logits[:, -1, :]
            if hasattr(outputs, "logits")
            else outputs[:, -1, :]
        )

        next_token_probs = logits.float().softmax(dim=-1)
        next_topk = logits.topk(dim=-1, k=topk)
        next_token_probs_filtered_topk = next_topk.values.float().softmax(dim=-1)

        if i == 0:
            # save the logits and predicted tokens for the immidiate next token
            model_logits = logits[0].clone().cpu()
            for token_id in next_topk.indices[0]:
                predicted_tokens.append(
                    PredictedToken(
                        token=mt.tokenizer.decode(token_id),
                        prob=next_token_probs[0, token_id].item(),
                    )
                )

        # sample the next token
        next_token = torch.multinomial(next_token_probs_filtered_topk, num_samples=1)
        next_token = next_topk.indices.gather(dim=-1, index=next_token)

        for j in range(len(input_ids)):
            generated_tokens[j].append(next_token[j].item())

        # update the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_tokens = mt.tokenizer.batch_decode(generated_tokens)

    return LinearRelationEditResult(
        predicted_tokens=predicted_tokens,
        model_logits=model_logits,
        model_generations=[
            "".join(generated_tokens)
            if isinstance(cur_generation, list)
            else cur_generation
            for cur_generation in generated_tokens
        ],
    )
