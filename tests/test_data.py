"""Unit tests for data module."""
import json

import pytest

STANDALONE_RELATION_PROMPT_TEMPLATES = ["{} a"]
STANDALONE_RELATION_SAMPLES = [
    {"subject": "foo", "object": "baz"},
    {"subject": "bar", "object": "baz"},
]
STANDALONE_RELATION = {
    "name": "rel",
    "prompt_templates": STANDALONE_RELATION_PROMPT_TEMPLATES,
    "samples": STANDALONE_RELATION_SAMPLES,
    "domain": ["foo", "bar"],
    "range": ["baz"],
}

GROUPED_RELATION_1 = {
    "name": "gr1",
    "prompt_templates": ["{} b"],
    "samples": [
        {"subject": "foo", "object": "foo"},
        {"subject": "baz", "object": "foo"},
    ],
}
GROUPED_RELATION_2 = {
    "name": "gr2",
    "prompt_templates": ["{} c"],
    "samples": [
        {"subject": "bar", "object": "foo"},
        {"subject": "baz", "object": "foo"},
    ],
}


@pytest.fixture
def relation_file(tmp_path_factory):
    """Create a dummy relation file."""
    file = tmp_path_factory.mktemp("relations") / "relation.json"
    with file.open("w") as handle:
        json.dump(STANDALONE_RELATION, handle)
    return file


@pytest.fixture
def relation_dir(tmp_path_factory):
    """Create a dummy relation directory."""
    tmp_path = tmp_path_factory.mktemp("relations")

    dir = tmp_path / "group"
    dir.mkdir(exist_ok=True, parents=True)

    for relation in (GROUPED_RELATION_1, GROUPED_RELATION_2):
        name = relation["name"]
        file = dir / f"{name}.json"
        with file.open("w") as handle:
            json.dump(relation, handle)
    return tmp_path


def test_load_relation(relation_file):
    """Test loading a single relation."""
    from src.data import load_relation

    relation = load_relation(relation_file)
    assert relation.name == "rel"
    assert relation.prompt_templates == STANDALONE_RELATION_PROMPT_TEMPLATES
    assert [s.to_dict() for s in relation.samples] == STANDALONE_RELATION_SAMPLES
    assert relation.domain == {"foo", "bar"}
    assert relation.range == {"baz"}


def test_load_relation_invalid_key(relation_file):
    """Test loading a single relation with an invalid key."""
    from src.data import load_relation

    with relation_file.open("r") as handle:
        relation_dict = json.load(handle)
    relation_dict["invalid_key"] = "invalid"
    with relation_file.open("w") as handle:
        json.dump(relation_dict, handle)

    with pytest.raises(ValueError):
        load_relation(relation_file)


def test_load_dataset(relation_file, relation_dir):
    """Test loading a dataset."""
    from src.data import load_dataset

    dataset = load_dataset(relation_file, relation_dir)
    assert len(dataset) == 3
    assert dataset[0].name == "rel"
    assert dataset[1].name == "gr1"
    assert dataset[2].name == "gr2"
