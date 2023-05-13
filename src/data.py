import argparse
import json
import logging
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Sequence

from src.utils import env_utils
from src.utils.typing import PathLike

import torch.utils.data
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)

RelationFnType = Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"]


@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    """A single (subject, object) pair in a relation."""

    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} -> {self.object}"


@dataclass(frozen=True)
class RelationProperties(DataClassJsonMixin):
    """Some metadata about a relation."""

    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool
    fn_type: str
    disambiguating: bool


@dataclass(frozen=True)
class Relation(DataClassJsonMixin):
    """An abstract mapping between subjects and objects.

    Attributes:
        name: The name of the relation, used as an ID.
        prompt_templates: Prompts representing the relation, where the subject is
            represented by {}.
        samples: A list of (subject, object) pairs satisfying the relation.
        _domain: Explicit list of all possible subjects. Accessed via the @property
            `domain`, which guesses the domain from the samples if not provided.
        _range: Equivalent to `_domain`, but for objects.
    """

    name: str
    prompt_templates: list[str]
    samples: list[RelationSample]
    properties: RelationProperties

    _domain: list[str] | None = None
    _range: list[str] | None = None

    @property
    def domain(self) -> set[str]:
        if self._domain is not None:
            return set(self._domain)
        return {sample.subject for sample in self.samples}

    @property
    def range(self) -> set[str]:
        if self._range is not None:
            return set(self._range)
        return {sample.object for sample in self.samples}

    def split(self, size: int) -> tuple["Relation", "Relation"]:
        """Break into a train/test split."""
        if size > len(self.samples):
            raise ValueError(f"size must be <= len(samples), got: {size}")

        samples = self.samples.copy()
        random.shuffle(samples)
        train_samples = samples[:size]
        test_samples = samples[size:]

        return (
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                properties=self.properties,
                samples=train_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                properties=self.properties,
                samples=test_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
        )

    def set(
        self,
        name: str | None = None,
        prompt_templates: Sequence[str] | None = None,
        properties: RelationProperties | None = None,
        samples: Sequence[RelationSample] | None = None,
        domain: Sequence[str] | None = None,
        range: Sequence[str] | None = None,
    ) -> "Relation":
        """Return a copy of this relation with any specified fields overwritten."""
        return Relation(
            name=name if name is not None else self.name,
            prompt_templates=list(prompt_templates)
            if prompt_templates is not None
            else self.prompt_templates,
            properties=properties if properties is not None else self.properties,
            samples=list(samples) if samples is not None else self.samples,
            _domain=list(domain) if domain is not None else self._domain,
            _range=list(range) if range is not None else self._range,
        )


class RelationDataset(torch.utils.data.Dataset[Relation]):
    """A torch dataset of relations."""

    def __init__(self, relations: list[Relation]):
        self.relations = relations

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, index: int) -> Relation:
        return self.relations[index]

    def filter(
        self,
        relation_names: Sequence[str] | None = None,
        **properties: bool | Sequence[str],
    ) -> "RelationDataset":
        relations = list(self.relations)
        if relation_names is not None:
            logger.debug(f"filtering to only relations: {relation_names}")
            relations = [r for r in relations if r.name in set(relation_names)]

        for key, value in properties.items():
            if value is not None:
                if isinstance(value, bool):
                    logger.debug(f"filtering by property {key}={value}")
                    matches = lambda x: x == value
                else:
                    logger.debug(f"filtering by property {key} in {value}")
                    value_set = set(value)
                    matches = lambda x: (x in value_set)

                relations = [
                    r for r in relations if matches(getattr(r.properties, key))
                ]

        return RelationDataset(relations)


def get_relation_fn_type(relation_dict: dict) -> RelationFnType:
    """Determine the function type of a relation."""

    # Check if relation is one-to-many
    one_to_many = False
    sub2obj: dict[str, set[str]] = {}
    for sample in relation_dict["samples"]:
        cur = sub2obj.get(sample["subject"], set())
        cur.add(sample["object"])
        sub2obj[sample["subject"]] = cur
    for obj_set in sub2obj.values():
        if len(obj_set) > 1:
            one_to_many = True
            break

    # Check if relation is many-to-one
    many_to_one = False
    obj2sub: dict[str, set[str]] = {}
    for sample in relation_dict["samples"]:
        cur = obj2sub.get(sample["object"], set())
        cur.add(sample["subject"])
        obj2sub[sample["object"]] = cur
    for sub_set in obj2sub.values():
        if len(sub_set) > 1:
            many_to_one = True
            break

    # Determine relation type
    if one_to_many and many_to_one:
        return "MANY_TO_MANY"
    elif one_to_many:
        return "ONE_TO_MANY"
    elif many_to_one:
        return "MANY_TO_ONE"
    else:
        return "ONE_TO_ONE"


def load_relation_dict(file: PathLike) -> dict:
    """Load dict for a single relation from a json file."""
    file = Path(file)
    if file.suffix != ".json":
        raise ValueError(f"relation files must be json, got: {file}")
    with file.open("r") as handle:
        relation_dict = json.load(handle)
    for key in ("domain", "range"):
        if key in relation_dict:
            relation_dict[f"_{key}"] = relation_dict.pop(key)

    # Check that all keys are valid kwargs to Relation
    valid_keys = set(field.name for field in fields(Relation))
    for key in relation_dict.keys():
        if key not in valid_keys:
            raise ValueError(
                f"invalid key in relation file {file}: {key}. "
                f"valid keys are: {valid_keys}"
            )

    # Compute the type of relation function (injection, surjection, bijection, etc.)
    relation_dict["properties"]["fn_type"] = get_relation_fn_type(relation_dict)

    return relation_dict


def load_relation(file: PathLike) -> Relation:
    """Load a single relation from a json file."""
    return Relation.from_dict(load_relation_dict(file))


def load_dataset(*paths: PathLike) -> RelationDataset:
    """Load relations from json files in a folder.

    Accepts one or more directories or files. If a file, should be JSON format, and will
    be read as one relation. If a directory, will recursively search for all JSON files.
    """
    if not paths:
        data_dir = env_utils.determine_data_dir()
        logger.debug(f"no paths provided, using default data dir: {data_dir}")
        paths = (data_dir,)

    # Load all relation files
    files = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            logger.debug(f"found relation file: {path}")
            files.append(path)
        else:
            logger.debug(f"{path} is directory, globbing for json files...")
            for file in sorted(path.glob("**/*.json")):
                logger.debug(f"found relation file: {file}")
                files.append(file)

    logger.debug(f"found {len(files)} relation files total, loading...")
    relation_dicts = [load_relation_dict(file) for file in files]

    # Mark all disambiguating relations
    domain_range_pairs: dict[tuple[str, str], int] = {}
    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        cur = domain_range_pairs.get((d, r), 0)
        domain_range_pairs[(d, r)] = cur + 1

    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        relation_dict["properties"]["disambiguating"] = domain_range_pairs[(d, r)] > 1

    # Create Relation objects
    relations = [Relation.from_dict(relation_dict) for relation_dict in relation_dicts]

    return RelationDataset(relations)


def load_dataset_from_args(args: argparse.Namespace) -> RelationDataset:
    """Load a dataset based on args from `add_data_args`."""
    dataset = load_dataset()
    dataset = dataset.filter(
        relation_names=args.rel_names,
        domain_name=args.rel_domains,
        range_name=args.rel_ranges,
        disambiguating=args.rel_disamb,
        symmetric=args.rel_sym,
        fn_type=args.rel_fn_types,
    )
    if len(dataset.relations) == 0:
        raise ValueError("no relations found matching all criteria")
    return dataset


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset filtering args."""
    parser.add_argument(
        "--rel-names", "-r", nargs="+", type=str, help="filter by relation name"
    )
    parser.add_argument(
        "--rel-domains", nargs="+", type=str, help="filter by relation domain"
    )
    parser.add_argument(
        "--rel-ranges", nargs="+", type=str, help="filter by relation range"
    )
    parser.add_argument(
        "--rel-fn-types",
        choices=("ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"),
        nargs="+",
        type=str,
        help="filter by relation function type (ONE_TO_ONE, ONE_TO_MANY, etc.)",
    )
    parser.add_argument(
        "--rel-disamb",
        action=argparse.BooleanOptionalAction,
        help="filter by whether relation is disambiguating",
    )
    parser.add_argument(
        "--rel-sym",
        action=argparse.BooleanOptionalAction,
        help="filter by whether relation is symmetric",
    )
