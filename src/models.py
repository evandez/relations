"""Functions for loading and interacting with pretrained language models.

This module is designed to house all the annoying branching logic
that comes with supporting analysis of many slightly different model
implementations.
"""
import argparse
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence, overload

from src.utils import env_utils, tokenizer_utils
from src.utils.typing import Device, Model, ModelInput, Tokenizer

import torch
import transformers

logger = logging.getLogger(__name__)

GPT_J_NAME_SHORT = "gptj"  # A useful alias for the CLI.
GPT_J_NAME = "EleutherAI/gpt-j-6B"

GPT_NEO_X_NAME_SHORT = "neox"
GPT_NEO_X_NAME = "EleutherAI/gpt-neox-20b"

LLAMA_13B_NAME = "llama-13b"
LLAMA_30B_NAME = "llama-30b"
LLAMA_NAME_SHORT = "llama"

DOWNLOADABLE_MODELS = frozenset(
    {
        GPT_J_NAME,
        GPT_NEO_X_NAME,
    }
)


# @dataclass(frozen=True)
class ModelAndTokenizer:
    """A pretrained model and its tokenizer."""

    # model: Model
    # tokenizer: Tokenizer

    def __init__(self, model: Model, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.extract_relavent_fields_from_config()

    @property
    def lm_head(self) -> torch.nn.Module:
        """Return the LM head."""
        if isinstance(
            self.model, transformers.GPT2LMHeadModel | transformers.GPTJForCausalLM
        ):
            return torch.nn.Sequential(self.model.transformer.ln_f, self.model.lm_head)
        elif isinstance(self.model, transformers.GPTNeoXForCausalLM):
            return torch.nn.Sequential(
                self.model.gpt_neox.final_layer_norm,
                self.model.embed_out,
            )
        elif isinstance(self.model, transformers.LlamaForCausalLM):
            return torch.nn.Sequential(self.model.model.norm, self.model.lm_head)
        else:
            raise ValueError(f"unknown model type: {type(self.model).__name__}")

    def to_(self, device: Optional[Device]) -> None:
        """Send model to the device."""
        self.model.to(device)

    def eval_(self) -> None:
        """Set model to eval mode."""
        self.model.eval()

    # tested for GPT-j, galactica and LLaMa
    def extract_relavent_fields_from_config(self) -> None:
        """
        extracts a bunch of highly used fields from different model configurations
        """
        config = self.model.config
        self.vocab_size = config.vocab_size

        model_type = None
        if hasattr(self.model, "transformer"):
            model_type = "gpt2"
        elif hasattr(self.model, "gpt_neox"):
            model_type = "gpt-neox"
        elif "llama" in config._name_or_path:
            model_type = "llama"
        elif "galactica" in config._name_or_path:
            model_type = "galactica"
        else:
            warnings.warn(
                "unknown model type >> unable to extract relavent fields from config"
            )
            return

        self.n_layer: int
        self.n_embd: int
        self.n_attn_head: int
        self.max_seq_length: int

        self.layer_name_format: str
        self.layer_names: Sequence[str]
        self.mlp_module_name_format: str
        self.attn_module_name_format: str
        self.ln_f_name: str
        self.unembedder_name: str
        self.embedder_name: str

        self.model_type = model_type

        if model_type in ["llama", "galactica"]:
            self.n_layer = config.num_hidden_layers
            self.n_embd = config.hidden_size
            self.n_attn_head = config.num_attention_heads
            self.max_seq_length = config.max_sequence_length

            layer_name_prefix = "model"
            if model_type == "galactica":
                layer_name_prefix = "model.decoder"

            self.layer_name_format = layer_name_prefix + ".layers.{}"

            self.embedder_name = "model.embed_tokens"
            self.ln_f_name = (
                "model.norm"
                if model_type == "llama"
                else "model.decoder.final_layer_norm"
            )
            self.unembedder_name = "lm_head"

            if model_type == "llama":
                self.mlp_module_name_format = "model.layers.{}.mlp"
            else:
                self.mlp_module_name_format = "model.layers.{}.fc2"  # this is the output of mlp in galactica. the input is on model.layers.{}.fc1
            self.attn_module_name_format = "model.layers.{}.self_attn"

        elif model_type in ["gpt2", "gpt-neox"]:
            self.n_layer = config.n_layer
            self.n_embd = config.n_embd
            self.n_attn_head = config.n_head
            self.max_seq_length = config.n_positions

            self.layer_name_format = "transformer.h.{}"
            self.embedder_name = "transformer.wte"
            self.ln_f_name = "transformer.ln_f"
            self.unembedder_name = "lm_head"
            self.mlp_module_name_format = "transformer.h.{}.mlp"
            self.attn_module_name_format = "transformer.h.{}.attn"

        # print("num_layers >> ", self.num_layers)
        if model_type is not None:
            self.layer_names = [
                self.layer_name_format.format(i) for i in range(self.n_layer)
            ]


def unwrap_model(value: Model | ModelAndTokenizer) -> Model:
    """Unwrap the model if necessary."""
    if isinstance(value, ModelAndTokenizer):
        return value.model
    return value


def unwrap_tokenizer(tokenizer: ModelAndTokenizer | Tokenizer) -> Tokenizer:
    """Unwrap the tokenizer."""
    if isinstance(tokenizer, ModelAndTokenizer):
        return tokenizer.tokenizer
    return tokenizer


def determine_layers(model: ModelAndTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = unwrap_model(model)
    assert isinstance(model, Model)

    if isinstance(
        model, transformers.GPTNeoXForCausalLM | transformers.LlamaForCausalLM
    ):
        n_layer = model.config.num_hidden_layers
    else:
        n_layer = model.config.n_layer

    return (*range(n_layer),)


@overload
def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = ...,
    *,
    return_dict: Literal[False] = ...,
) -> Sequence[str]:
    """Determine layer path for each layer."""
    ...


@overload
def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = ...,
    *,
    return_dict: Literal[True],
) -> dict[int, str]:
    """Determine mapping from layer to layer path."""
    ...


def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = None,
    *,
    return_dict: bool = False,
) -> Sequence[str] | dict[int, str]:
    """Determine the absolute paths to the given layers in the model.

    Args:
        model: The model.
        layers: The specific layer (numbers) to look at. Defaults to all of them.
            Can be a negative number.
        return_dict: If True, return mapping from layer to layer path,
            otherwise just return list of layer paths in same order as `layers`.

    Returns:
        Mapping from layer number to layer path.

    """
    model = unwrap_model(model)

    if layers is None:
        layers = determine_layers(model)

    assert isinstance(model, Model), type(model)

    layer_paths = {}
    for layer in layers:
        if layer < 0:
            layer = len(determine_layers(model)) + layer

        if isinstance(model, transformers.GPTNeoXForCausalLM):
            layer_path = f"gpt_neox.layers.{layer}"
        elif isinstance(model, transformers.LlamaForCausalLM):
            layer_path = f"model.layers.{layer}"
        else:
            layer_path = f"transformer.h.{layer}"
        layer_paths[layer] = layer_path

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


def determine_hidden_size(model: ModelAndTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)
    return model.config.hidden_size


def determine_device(model: ModelAndTokenizer | Model) -> torch.device | None:
    """Determine device model is running on."""
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None


def determine_dtype(model: ModelAndTokenizer | Model) -> torch.dtype | None:
    """Determine dtype of model."""
    parameter = any_parameter(model)
    return parameter.dtype if parameter is not None else None


def any_parameter(model: ModelAndTokenizer | Model) -> torch.nn.Parameter | None:
    """Get any example parameter for the model."""
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)


def tokenize_words(
    tokenizer: ModelAndTokenizer | Tokenizer,
    words: str | Sequence[str],
    spaces: bool = True,
) -> ModelInput:
    """Return first token ID for word, accounting for whether model expects spaces."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if isinstance(words, str):
        words = [words]

    if spaces and isinstance(
        tokenizer,
        transformers.GPT2TokenizerFast | transformers.GPTNeoXTokenizerFast,
    ):
        words = [f" {word}" for word in words]

    return tokenizer(words, return_tensors="pt", padding=True)


@contextmanager
def set_padding_side(
    tokenizer: Tokenizer | ModelAndTokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """Wrap `tokenizer_utils.set_padding_side`."""
    tokenizer = unwrap_tokenizer(tokenizer)
    with tokenizer_utils.set_padding_side(tokenizer, padding_side=padding_side):
        yield


def map_to(
    orig: Any, device: Device | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Map all tensors in the given value to the device.

    Args:
        orig: Any sequence of or mapping to tensors, or just a tensor.
        device: Device to send to.

    Returns:
        Same value, but with all tensors moved to the device.

    """
    if device is None and dtype is None:
        return orig

    result = orig
    if isinstance(orig, torch.Tensor):
        result = orig.to(device=device, dtype=dtype)
    elif isinstance(orig, dict):
        result = {
            key: map_to(value, device=device, dtype=dtype)
            for key, value in orig.items()
        }
    elif isinstance(orig, (list, tuple)):
        result = orig.__class__(
            map_to(value, device=device, dtype=dtype) for value in orig
        )
    assert isinstance(result, orig.__class__), f"{type(result)}/{type(orig)}"
    return result


def load_model(
    name: str, device: Optional[Device] = None, fp16: Optional[bool] = None
) -> ModelAndTokenizer:
    """Load the model given its string name.

    Args:
        name: Name of the model or path to it.
        device: If set, send model to this device. Defaults to CPU.
        fp16: Whether to use half precision. If not set, depends on model.

    Returns:
        ModelAndTokenizer: Loaded model and its tokenizer.

    """
    if name == GPT_J_NAME_SHORT:
        name = GPT_J_NAME
    elif name == GPT_NEO_X_NAME_SHORT:
        name = GPT_NEO_X_NAME
    elif name == LLAMA_NAME_SHORT:
        name = LLAMA_13B_NAME

    # I usually save randomly initialized variants under the short name of the
    # corresponding real model (e.g. gptj_random, neox_random), so check here
    # if we are dealing with *any* variant of the big model.
    is_gpt_j_variant = name == GPT_J_NAME or GPT_J_NAME_SHORT in name
    is_neo_x_variant = name == GPT_NEO_X_NAME or GPT_NEO_X_NAME_SHORT in name
    is_llama_variant = (
        name in {LLAMA_13B_NAME, LLAMA_30B_NAME} or LLAMA_NAME_SHORT in name
    )

    if fp16 is None:
        fp16 = is_gpt_j_variant or is_neo_x_variant or is_llama_variant

    torch_dtype = torch.float16 if fp16 else None

    kwargs: dict = dict(torch_dtype=torch_dtype)
    if is_gpt_j_variant:
        kwargs["low_cpu_mem_usage"] = True
        if fp16:
            kwargs["revision"] = "float16"

    # If model is not automatically downloadable from huggingface, assume it is
    # available locally in the project models directory.
    if name not in DOWNLOADABLE_MODELS:
        models_dir = env_utils.determine_models_dir()
        logger.debug(f"{name} not downloadable, will look for weights in {models_dir}")

        path = Path(name)
        if not path.is_absolute() and not path.is_relative_to(models_dir):
            name = str(models_dir / name)

    logger.info(f"loading {name} (device={device}, fp16={fp16})")

    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(torch_dtype)
    model.to(device)
    model.eval()

    if is_llama_variant:
        tokenizer = transformers.LlamaTokenizerFast.from_pretrained(name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return ModelAndTokenizer(model, tokenizer)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add args needed to load a model.

    The args include:
        --model: The language model to load, defaulting to GPT-J.
        --device: The device to send model and inputs to.
        --fp16: Whether to use half precision version of the model.
            Note this is used as `--fp16 False` since default value depends on
            which model we are loading.
    """
    parser.add_argument(
        "--model",
        "-m",
        default=GPT_J_NAME_SHORT,
        help="model to edit",
    )
    parser.add_argument("--device", help="device to train on")
    parser.add_argument("--fp16", type=bool, help="set whether to use fp16")
