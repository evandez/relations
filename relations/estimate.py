"""Estimate relation operator using Jacobian."""
import argparse
from dataclasses import dataclass
from typing import Any, Sequence

import baukit
import torch
import torch.autograd.functional
import torch.nn
import transformers

Model = transformers.GPT2LMHeadModel
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
Device = int | str | torch.device


def find_token_range(
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


def determine_token_index(start: int, end: int, offset: int) -> int:
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
        subject_i, subject_j = find_token_range(
            prompt, subject, offset_mapping=offset_mapping[0]
        )
        h_token_index = determine_token_index(subject_i, subject_j, subject_token_index)

        layer_name = f"transformer.h.{self.layer}"
        with baukit.Trace(self.model, layer_name) as ret:
            self.model(**inputs)
        h = ret.output[0][:, h_token_index]
        z = h.mm(self.weight.t()) + self.bias
        logits = self.lm_head(z)
        token_ids = logits.topk(dim=-1, k=return_top_k).indices.squeeze().tolist()
        return self.tokenizer.convert_ids_to_tokens(token_ids)


def estimate_relation_operator(
    model: Model,
    tokenizer: Tokenizer,
    subject: str,
    relation: str,
    subject_token_index: int = -1,
    layer: int = 25,
    device: Device | None = None,
) -> RelationOperator:
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
        The estimated operator.

    """
    model.to(device)

    prompt = relation.format(subject)
    inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        device
    )

    offset_mapping = inputs.pop("offset_mapping")
    subject_i, subject_j = find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0]
    )
    h_token_index = determine_token_index(subject_i, subject_j, subject_token_index)

    h_layer_name = f"transformer.h.{layer}"
    z_layer_name = f"transformer.h.{model.config.n_layer - 1}"

    with baukit.TraceDict(model, (h_layer_name, z_layer_name)) as ret:
        model(**inputs)

    h = ret[h_layer_name].output[0][0, h_token_index]
    z = ret[z_layer_name].output[0][0, -1]

    def compute_z_from_h(h: torch.Tensor) -> torch.Tensor:
        def insert_h(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            output[0][0, h_token_index] = h
            return output

        with baukit.TraceDict(
            model, (h_layer_name, z_layer_name), edit_output=insert_h
        ) as ret:
            model(**inputs)
        return ret[z_layer_name].output[0][0, -1]

    weight = torch.autograd.functional.jacobian(compute_z_from_h, h, vectorize=True)
    bias = z[None] - h[None].mm(weight.t())
    return RelationOperator(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        relation=relation,
        weight=weight,
        bias=bias,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["gpt2-xl"], default="gpt2-xl", help="language model to use"
    )
    parser.add_argument("--k", type=int, default=5, help="number of top O's to show")
    parser.add_argument("--layer", type=int, default=30, help="layer to get h from")
    parser.add_argument("--device", help="device to run on")
    args = parser.parse_args()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer
    k = args.k

    print(f"loading {args.model}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Example 1: "is located in"
    print("--- is located in ---")
    is_located_in = estimate_relation_operator(
        model,
        tokenizer,
        "The Space Needle",
        "{} is located in the country of",
        layer=layer,
        device=device,
    )
    for subject, subject_token_index in (
        ("The Space Needle", -1),
        ("The Eiffel Tower", -2),
        ("The Great Wall", -1),
        ("Niagara Falls", -2),
    ):
        objects = is_located_in(
            subject,
            subject_token_index=subject_token_index,
            device=device,
            return_top_k=k,
        )
        print(f"{subject}: {objects}")

    # Example 2: "is CEO of"
    # This one is less sensitive to which h you choose; can usually just do last.
    print("--- is CEO of ---")
    is_ceo_of = estimate_relation_operator(
        model, tokenizer, "Indra Nooyi", "{} is CEO of", layer=layer, device=device
    )
    for subject in (
        "Indra Nooyi",
        "Sundar Pichai",
        "Elon Musk",
        "Mark Zuckerberg",
        "Satya Nadella",
        "Jeff Bezos",
        "Tim Cook",
    ):
        objects = is_ceo_of(subject, device=device, return_top_k=k)
        print(f"{subject}: {objects}")

    # Example 3: "is lead singer of"
    # Seems to *actually* find the "is lead singer of grunge rock group" relation.
    print("--- is lead singer of ---")
    is_lead_singer_of = estimate_relation_operator(
        model,
        tokenizer,
        "Chris Cornell",
        "{} is the lead singer of the band",
        layer=layer,
        device=device,
    )
    for subject in (
        "Chris Cornell",
        "Kurt Cobain",
        "Eddie Vedder",
        "Stevie Nicks",
        "Freddie Mercury",
    ):
        objects = is_lead_singer_of(subject, device=device, return_top_k=k)
        print(f"{subject}: {objects}")

    # Example 4: "plays the sport of"
    # Does not work at all. Not sure why.
    print("--- plays the sport of ---")
    plays_sport_of = estimate_relation_operator(
        model,
        tokenizer,
        "Megan Rapinoe",
        "{} plays the sport of",
        layer=layer,
        device=device,
    )
    for subject in (
        "Megan Rapinoe",
        "Larry Bird",
        "John McEnroe",
        "Oksana Baiul",
        "Tom Brady",
        "Babe Ruth",
    ):
        objects = plays_sport_of(subject, device=device, return_top_k=k)
        print(f"{subject}: {objects}")
