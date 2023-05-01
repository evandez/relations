import json
import logging
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

from src.utils import env_utils
from src.utils.typing import PathLike

import torch.utils.data
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    """A single (subject, object) pair in a relation."""

    subject: str
    object: str


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
                samples=train_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                samples=test_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
        )

    def set(
        self,
        name: str | None = None,
        prompt_templates: Sequence[str] | None = None,
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


def load_relation(file: PathLike) -> Relation:
    """Load a single relation from a json file."""
    file = Path(file)
    if file.suffix != ".json":
        raise ValueError(f"relation files must be json, got: {file}")
    with file.open("r") as handle:
        relation_dict = json.load(handle)
    for key in ("domain", "range"):
        if key in relation_dict:
            relation_dict[f"_{key}"] = relation_dict.pop(key)

    # check that all keys are valid kwargs to Relation
    valid_keys = set(field.name for field in fields(Relation))
    for key in relation_dict.keys():
        if key not in valid_keys:
            raise ValueError(
                f"invalid key in relation file {file}: {key}. "
                f"valid keys are: {valid_keys}"
            )

    return Relation.from_dict(relation_dict)


def load_dataset(*paths: PathLike) -> RelationDataset:
    """Load relations from json files in a folder.

    Accepts one or more directories or files. If a file, should be JSON format, and will
    be read as one relation. If a directory, will recursively search for all JSON files.
    """
    if not paths:
        data_dir = env_utils.determine_data_dir()
        logger.debug(f"no paths provided, using default data dir: {data_dir}")
        paths = (data_dir,)

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
    relations = [load_relation(file) for file in files]
    return RelationDataset(relations)
