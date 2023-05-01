"""Unit tests for metrics module."""
import pytest


@pytest.mark.parametrize("case, predictions, targets, expected", (
    (
        "always correct",
        [["a", "b", "c"], ["b", "a", "c"], ["c", "a", "b"]],
        ["a", "b", "c"],
        [1.0, 1.0, 1.0],
    ),
    (
        "wrong@1, right@3",
        [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]],
        ["a", "b", "c"],
        [1 / 3, 2 / 3, 1.0],
    ),
    (
        "case insensitivity",
        [["A", "b", "c"], ["a", "B", "c"], ["a", "b", "C"]],
        ["a", "b", "c"],
        [1 / 3, 2 / 3, 1.0],
    ),
    (
        "whitespace insensitivity",
        [[" a", "b", "c"], [" a", " b", "c"], [" a", " b", " c"]],
        ["a", "b", "c"],
        [1 / 3, 2 / 3, 1.0],
    )
))
def test_recall(case, predictions, targets, expected):
    from src import metrics

    actual = metrics.recall(predictions, targets)
    assert actual == pytest.approx(expected), case
