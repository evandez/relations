"""Estimate relation operator using Jacobian."""
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import baukit
import torch
import torch.autograd.functional
import torch.nn
import transformers

Model: TypeAlias = transformers.GPT2LMHeadModel
ModelInput: TypeAlias = transformers.BatchEncoding
Tokenizer: TypeAlias = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping: TypeAlias = Sequence[tuple[int, int]]
Device: TypeAlias = int | str | torch.device


def _find_token_range(
    string: str,
    substring: str,
    tokenizer: Tokenizer | None = None,
    occurrence: int = 0,
    offset_mapping: TokenizerOffsetMapping | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:
        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...
        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        The start (inclusive) and end (exclusive) token idx.

    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    char_start = string.index(substring)
    for _ in range(occurrence):
        try:
            char_start = string.index(substring, char_start + 1)
        except ValueError as error:
            raise ValueError(
                f"could not find {occurrence} occurrences "
                f'of "{substring} in "{string}"'
            ) from error
    char_end = char_start + len(substring)

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def _determine_token_index(start: int, end: int, offset: int) -> int:
    """Determine absolute index of token in range given offset."""
    if offset < 0:
        assert offset >= -end
        index = end + offset
    else:
        assert offset < end - start
        index = start + offset
    return index


# TODO(evandez): Should maybe be a torch.nn.Module someday
@dataclass(frozen=True)
class RelationOperator:
    """Implements a relation operator for the given LM."""

    model: Model
    tokenizer: Tokenizer
    relation: str
    layer: int
    weight: torch.Tensor
    bias: torch.Tensor

    def overwrite(
        self,
        relation: str | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> "RelationOperator":
        """Overwrite one or more parameters of the operator."""
        return RelationOperator(
            model=self.model,
            tokenizer=self.tokenizer,
            relation=self.relation if relation is None else relation,
            layer=self.layer,
            weight=self.weight if weight is None else weight,
            bias=self.bias if bias is None else bias,
        )

    @property
    def lm_head(self) -> torch.nn.Module:
        """Return just the LM head part of the model."""
        return torch.nn.Sequential(
            self.model.transformer.ln_f,
            self.model.lm_head,
        )

    def __call__(
        self,
        subject: str,
        subject_token_index: int = -1,
        return_top_k: int = 5,
        device: Device | None = None,
    ) -> tuple[str, ...]:
        """Estimate the O in (S, R, O) given a new S.

        Args:
            subject: The S to estimate O for. E.g., "The Space Needle"
            subject_token_index: Subject token to use as h.
            return_top_k: Number this many top candidates for O.
            device: Send model and inputs to this device.

        Returns:
            Top predictions for O.

        """
        self.model.to(device)

        prompt = self.relation.format(subject)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True
        ).to(device)

        offset_mapping = inputs.pop("offset_mapping")
        subject_i, subject_j = _find_token_range(
            prompt, subject, offset_mapping=offset_mapping[0]
        )
        h_token_index = _determine_token_index(
            subject_i, subject_j, subject_token_index
        )

        layer_name = f"transformer.h.{self.layer}"
        with baukit.Trace(self.model, layer_name) as ret:
            self.model(**inputs)
        h = ret.output[0][:, h_token_index]
        z = h.mm(self.weight.t()) + self.bias
        logits = self.lm_head(z)
        dist = torch.softmax(logits.float(), dim=-1)

        topk = dist.topk(dim=-1, k=return_top_k)
        probs = topk.values.squeeze().tolist()
        token_ids = topk.indices.squeeze().tolist()
        words = [self.tokenizer.decode(token_id) for token_id in token_ids]

        return tuple(zip(words, probs))


@dataclass(frozen=True)
class RelationOperatorMetadata:
    """Metadata from estimating the relation operator."""

    subject: str
    prompt: str
    subject_token_index: int
    inputs: ModelInput
    logits: torch.Tensor


@torch.no_grad()
def relation_operator_from_sample(
    model: Model,
    tokenizer: Tokenizer,
    subject: str,
    relation: str,
    subject_token_index: int = -1,
    layer: int = 15,
    device: Device | None = None,
) -> tuple[RelationOperator, RelationOperatorMetadata]:
    """Estimate the r in (s, r, o) as a linear operator.

    Args:
        model: The language model. Only supports GPT-2 for now.
        tokenizer: The tokenizer.
        subject: The subject, e.g. "The Eiffel Tower".
        relation: The relation as a template string, e.g. "{} is located in"
        subject_token_index: Token index to use as h.
        layer: Layer to take h from.
        device: Send inputs and model to this device.

    Returns:
        The estimated operator and its metadata.

    """
    model.to(device)

    prompt = relation.format(subject)
    inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        device
    )

    offset_mapping = inputs.pop("offset_mapping")
    subject_i, subject_j = _find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0]
    )
    h_token_index = _determine_token_index(subject_i, subject_j, subject_token_index)

    # Precompute everything up to the subject, if there is anything before it.
    past_key_values = None
    input_ids = inputs.input_ids
    if subject_i > 0:
        outputs = model(input_ids=input_ids[:, :subject_i], use_cache=True)
        past_key_values = outputs.past_key_values
        input_ids = input_ids[:, subject_i:]
        h_token_index -= subject_i
    use_cache = past_key_values is not None

    # Precompute initial h and z.
    h_layer_name = f"transformer.h.{layer}"
    z_layer_name = f"transformer.h.{model.config.n_layer - 1}"
    with baukit.TraceDict(model, (h_layer_name, z_layer_name)) as ret:
        outputs = model(
            input_ids=input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    h = ret[h_layer_name].output[0][0, h_token_index]
    z = ret[z_layer_name].output[0][0, -1]

    # Now compute J and b.
    def compute_z_from_h(h: torch.Tensor) -> torch.Tensor:
        def insert_h(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            output[0][0, h_token_index] = h
            return output

        with baukit.TraceDict(
            model, (h_layer_name, z_layer_name), edit_output=insert_h
        ) as ret:
            model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return ret[z_layer_name].output[0][0, -1]

    weight = torch.autograd.functional.jacobian(compute_z_from_h, h, vectorize=True)
    bias = z[None] - h[None].mm(weight.t())
    operator = RelationOperator(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        relation=relation,
        weight=weight,
        bias=bias,
    )
    metadata = RelationOperatorMetadata(
        subject=subject,
        subject_token_index=subject_token_index,
        prompt=prompt,
        inputs=inputs.to("cpu"),
        logits=outputs.logits.cpu(),
    )
    return operator, metadata


@dataclass(frozen=True)
class RelationOperatorBatchMetadata:
    """Metadata from estimating a relation operator from a batch."""

    subject_for_weight: str
    prompt_for_weight: str

    operator_for_weight: RelationOperator
    operators_for_bias: Sequence[RelationOperator]

    metadata_for_weight: RelationOperatorMetadata
    metadata_for_bias: Sequence[RelationOperatorMetadata]


@torch.no_grad()
def relation_operator_from_batch(
    model: Model,
    tokenizer: Tokenizer,
    samples: Sequence[tuple[str, str]],
    relation: str,
    subject_token_index: int | Sequence[int] = -1,
    layer: int = 15,
    device: Device | None = None,
) -> tuple[RelationOperator, RelationOperatorBatchMetadata]:
    """Estimate a higher quality J and b from a batch of samples.

    J will be estimated from an ICL prompt consisting of all samples but one.
    The one that is left out will be selected as the one that has the highest
    probability under the LM given the prompt.

    The bias will be estimated for each subject individually and then averaged.

    Args:
        model: The language model.
        tokenizer: The language model's tokenizer.
        samples: Batch of subjects and their values for this relation.
            E.g., if the relation is "{} is located in", then this could be
            [("The Space Needle", "Seattle"), ("The Eiffel Tower", "Paris"), ...].
        relation: The relation template string, e.g. "{} is located in".
        subject_token_index: Token index to use as h. Can be list specifying which
            token for each subject.
        layer: The layer to take h from.
        device: Send model and inputs to this device.

    Returns:
        The relation operator and its metadata.

    """
    model.to(device)

    if isinstance(subject_token_index, int):
        subject_token_index = [subject_token_index] * len(samples)
    if len(subject_token_index) != len(samples):
        raise ValueError(
            f"subject_token_index has length {len(subject_token_index)}"
            f"which does not match samples length {len(samples)}"
        )

    # Estimate the biases, storing the confidence of the target token
    # along the way.
    operators_for_bias = []
    metadata_for_bias = []
    confidences = []
    for i, (subject, object) in enumerate(samples):
        operator, metadata = relation_operator_from_sample(
            model,
            tokenizer,
            subject,
            relation,
            subject_token_index=subject_token_index[i],
            layer=layer,
            device=device,
        )
        operators_for_bias.append(operator)
        metadata_for_bias.append(metadata)

        object_token_id = tokenizer(object).input_ids[0]
        logits = metadata.logits[0, -1]
        logps = torch.log_softmax(logits, dim=-1)
        confidences.append(logps[object_token_id])

    # Now estimate J, using an ICL prompt with the model's most-confident subject
    # used as the training example.
    chosen = torch.stack(confidences).argmax().item()
    assert isinstance(chosen, int)
    subject_for_weight, _ = sample = samples[chosen]
    others = list(set(samples) - {sample})
    prompt_for_weight = "\n".join(relation.format(s) + f" {o}." for s, o in others)
    prompt_for_weight += "\n" + relation
    operator_for_weight, metadata_for_weight = relation_operator_from_sample(
        model,
        tokenizer,
        subject_for_weight,
        prompt_for_weight,
        subject_token_index=subject_token_index[chosen],
        layer=layer,
        device=device,
    )

    # Package it all up.
    weight = operator_for_weight.weight
    bias = torch.stack([operator.bias for operator in operators_for_bias]).mean(dim=0)
    batch_operator = RelationOperator(
        model=model,
        tokenizer=tokenizer,
        relation=relation,
        layer=layer,
        weight=weight,
        bias=bias,
    )
    batch_metadata = RelationOperatorBatchMetadata(
        subject_for_weight=subject_for_weight,
        prompt_for_weight=prompt_for_weight,
        operator_for_weight=operator_for_weight,
        metadata_for_weight=metadata_for_weight,
        operators_for_bias=operators_for_bias,
        metadata_for_bias=metadata_for_bias,
    )
    return batch_operator, batch_metadata
