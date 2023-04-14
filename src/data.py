from dataclasses import dataclass
from pathlib import Path

from src.utils.typing import PathLike

import torch.utils.data
from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    subject: str
    object: str


@dataclass(frozen=True)
class Relation(DataClassJsonMixin):
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

    def filter(
        self,
        subject: str | None = None,
        object: str | None = None,
        prompt_template: str | None = None,
    ) -> "Relation":
        samples = self.samples
        prompt_templates = self.prompt_templates
        domain = self._domain
        range = self._range

        if subject is not None:
            samples = [sample for sample in samples if sample.subject == subject]
            if domain is not None:
                domain = [subject]

        if object is not None:
            samples = [sample for sample in samples if sample.object == object]
            if range is not None:
                range = [object]

        if prompt_template is not None:
            prompt_templates = [prompt_template]

        return Relation(
            name=self.name,
            prompt_templates=prompt_templates,
            samples=samples,
            _domain=domain,
            _range=range,
        )


class RelationDataset(torch.utils.data.Dataset[Relation]):
    def __init__(self, relations: list[Relation]):
        self.relations = relations

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, index: int) -> Relation:
        return self.relations[index]

    def filter(
        self,
        subject: str | None = None,
        object: str | None = None,
        relation_name: str | None = None,
    ) -> "RelationDataset":
        relations = self.relations
        if subject is not None:
            relations = [relation.filter(subject=subject) for relation in relations]
        if object is not None:
            relations = [relation.filter(object=object) for relation in relations]
        if relation_name is not None:
            relations = [
                relation for relation in relations if relation.name == relation_name
            ]
        relations = [relation for relation in relations if relation.samples]
        return RelationDataset(relations)


def load(path: PathLike) -> RelationDataset:
    path = Path(path)
    relations = []
    for file in path.glob("**/*.json"):
        with file.open("r") as handle:
            relation = Relation.from_json(handle.read())
            relations.append(relation)
    return RelationDataset(relations)
