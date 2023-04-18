"""Unit tests for `src.utils.tokenizers`."""
from src.utils import tokenizer_utils

import pytest
import transformers

BERT_TOKENIZER = "bert-base-uncased"
GPT2_TOKENIZER = "gpt2"


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Return a GPT2 tokenizer for testing."""
    return transformers.AutoTokenizer.from_pretrained(GPT2_TOKENIZER, use_fast=True)


@pytest.fixture(scope="module")
def bert_tokenizer():
    """Return a BERT tokenizer for testing."""
    return transformers.AutoTokenizer.from_pretrained(BERT_TOKENIZER, use_fast=True)


@pytest.mark.parametrize(
    "tokenizer_name,string,substring,kwargs,expected",
    (
        (
            BERT_TOKENIZER,
            "The batman is the night.",
            "batman",
            {"add_special_tokens": False},
            (1, 2),
        ),
        (
            BERT_TOKENIZER,
            "The batman is the night.",
            "batman",
            {"add_special_tokens": True},
            (2, 3),
        ),
        (
            GPT2_TOKENIZER,
            "The batman is the night.",
            "batman",
            {"add_special_tokens": False},
            (1, 3),
        ),
        (
            GPT2_TOKENIZER,
            "The batman is the night.",
            "batman",
            {"add_special_tokens": True},
            (1, 3),
        ),
        (
            GPT2_TOKENIZER,
            "The batman is the night. The batman is cool.",
            "batman",
            {
                "occurrence": 1,
            },
            (8, 10),
        ),
    ),
)
def test_find_token_range(
    gpt2_tokenizer, bert_tokenizer, tokenizer_name, string, substring, kwargs, expected
):
    """Test find_token_range returns correct token range."""
    if tokenizer_name == GPT2_TOKENIZER:
        tokenizer = gpt2_tokenizer
    else:
        assert tokenizer_name == BERT_TOKENIZER
        tokenizer = bert_tokenizer
    actual = tokenizer_utils.find_token_range(string, substring, tokenizer, **kwargs)
    assert actual == expected


def test_find_token_range_no_substring(gpt2_tokenizer):
    """Test find_token_range dies when arg is not real substring."""
    bad = "foo bar"
    with pytest.raises(ValueError, match=f".*{bad}.*"):
        tokenizer_utils.find_token_range(
            "The batman is the night.", bad, gpt2_tokenizer
        )


@pytest.mark.parametrize(
    "start,end,offset,expected",
    (
        (4, 6, -1, 5),
        (4, 6, 0, 4),
        (4, 6, 1, 5),
    ),
)
def test_offset_to_absolute_index(start, end, offset, expected):
    """Test offset_to_absolute_index correctly resolves offset."""
    actual = tokenizer_utils.offset_to_absolute_index(start, end, offset)
    assert actual == expected


@pytest.mark.parametrize(
    "start,end,offset",
    (
        (-1, 0, 0),
        (5, 4, 0),
    ),
)
def test_offset_to_absolute_index_bad_range(start, end, offset):
    """Make sure offset_to_absolute_index dies on bad range."""
    with pytest.raises(ValueError):
        tokenizer_utils.offset_to_absolute_index(start, end, offset)


def test_offset_to_absolute_index_bad_offset():
    """Make sure offset_to_absolute_index dies on bad offset."""
    with pytest.raises(ValueError):
        tokenizer_utils.offset_to_absolute_index(0, 1, 2)
