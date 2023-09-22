"""Unit tests for `src.functional`."""
from src.functional import predict_next_token
from src.models import ModelAndTokenizer

import pytest
import transformers


@pytest.fixture(scope="module")
def gpt2():
    """Return a GPT2 model and tokenizer for testing."""
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    return ModelAndTokenizer(model, tokenizer)


def test_predict_next_token_single_item(gpt2):
    results = predict_next_token(
        mt=gpt2,
        prompt="Rome is located in the country of",
        k=1,
    )
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].token == " Italy"
    # snapshot probability to ensure it doesn't change unexpectedly in refactor
    assert results[0][0].prob == pytest.approx(0.072319, abs=0.001)


def test_predict_next_token_single_item_top_k(gpt2):
    results = predict_next_token(
        mt=gpt2,
        prompt="Rome is located in the country of",
        k=3,
    )
    assert len(results) == 1
    assert len(results[0]) == 3
    assert results[0][0].token == " Italy"
    assert results[0][1].token == " the"
    assert results[0][2].token == " Romania"


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 5],
)
def test_predict_next_token_single_multi(gpt2, batch_size):
    results = predict_next_token(
        mt=gpt2,
        prompt=[
            "Rome is located in the country of",
            "Tokyo is located in the country of",
            "Beijing is located in the country of",
        ],
        k=1,
        batch_size=batch_size,
    )
    assert len(results) == 3
    for result in results:
        assert len(result) == 1
        assert result[0].prob > 0.05
        assert result[0].prob < 0.5
    assert results[0][0].token == " Italy"
    assert results[1][0].token == " Japan"
    assert results[2][0].token == " China"
    # snapshot probabilities to ensure it doesn't change unexpectedly in refactor
    assert results[0][0].prob == pytest.approx(0.07232, abs=0.001)
    assert results[1][0].prob == pytest.approx(0.34312, abs=0.001)
    assert results[2][0].prob == pytest.approx(0.11155, abs=0.001)
