"""Reads out hyperparameters from committed config files."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from src import data
from src.utils import env_utils
from src.utils.typing import PathLike

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)

HParamsT = TypeVar("HParamsT", bound="HParams")


@dataclass(frozen=True, kw_only=True)
class HParams(DataClassJsonMixin):
    def save_json_file(self, file: PathLike) -> None:
        file = Path(file)
        file.parent.mkdir(exist_ok=True, parents=True)
        with file.open("w") as handle:
            handle.write(self.to_json(indent=4))

    @classmethod
    def from_json_file(cls: type[HParamsT], file: PathLike) -> HParamsT:
        file = Path(file)
        with file.open("r") as handle:
            return cls.from_json(handle.read())


RelationHParamsT = TypeVar("RelationHParamsT", bound="RelationHParams")


@dataclass(frozen=True, kw_only=True)
class RelationHParams(HParams):
    relation_name: str
    h_layer: int
    beta: float
    rank: int | None = None
    z_layer: int | None = None
    model_name: str | None = None

    def save(self, file: PathLike | None = None) -> None:
        if file is None:
            file = self.default_relation_file(
                self.relation_name, model_name=self.model_name
            )
        logger.info(f'writing "{self.relation_name}" hparams to {file}')
        self.save_json_file(file)

    @classmethod
    def from_relation(
        cls: type[RelationHParamsT], relation: str | data.Relation
    ) -> RelationHParamsT:
        hparams_file = cls.default_relation_file(relation)
        if not hparams_file.exists():
            raise FileNotFoundError(
                f'expected {cls.__name__} file for relation "{relation}" at {hparams_file}'
            )
        return cls.from_json_file(hparams_file)

    @staticmethod
    def default_relation_file(
        relation: str | data.Relation, model_name: str | None = None
    ) -> Path:
        if isinstance(relation, data.Relation):
            relation = relation.name
        relation = relation.replace(" ", "_").replace("'", "")
        hparams_dir = env_utils.determine_hparams_dir()
        if model_name is not None:
            hparams_dir = hparams_dir / model_name
        hparams_file = hparams_dir / f"{relation}.json"
        return hparams_file


def get(relation: str | data.Relation) -> RelationHParams:
    """Get hyperparameters for a given relation."""
    return RelationHParams.from_relation(relation)
