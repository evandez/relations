"""Reads out hyperparameters from committed config files."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from src import data, models
from src.utils import env_utils
from src.utils.typing import Layer, PathLike

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
    model_name: str
    relation_name: str
    h_layer: Layer
    beta: float
    rank: int | None = None
    z_layer: int | None = None
    h_layer_edit: Layer | None = None

    def save(self, file: PathLike | None = None) -> None:
        if file is None:
            file = self.default_relation_file(self.model_name, self.relation_name)
        logger.info(
            f'writing {self.model_name}/"{self.relation_name}" hparams to {file}'
        )
        self.save_json_file(file)

    @classmethod
    def from_relation(
        cls: type[RelationHParamsT],
        model: str | models.ModelAndTokenizer,
        relation: str | data.Relation,
    ) -> RelationHParamsT | None:
        hparams_file = cls.default_relation_file(model, relation)
        if not hparams_file.exists():
            return None
        logger.info(f"reading hparams from {hparams_file}")
        hparams = cls.from_json_file(hparams_file)
        logger.info(
            f'{hparams.model_name}/"{hparams.relation_name}" hparams: {hparams}'
        )
        return hparams

    @staticmethod
    def default_relation_file(
        model: str | models.ModelAndTokenizer,
        relation: str | data.Relation,
    ) -> Path:
        if isinstance(model, models.ModelAndTokenizer):
            model = model.name
        if isinstance(relation, data.Relation):
            relation = relation.name
        relation = relation.replace(" ", "_").replace("'", "")
        hparams_dir = env_utils.determine_hparams_dir() / model
        hparams_file = hparams_dir / f"{relation}.json"
        return hparams_file


def get(
    model: models.ModelAndTokenizer, relation: str | data.Relation
) -> RelationHParams | None:
    """Get hyperparameters for a given relation."""
    return RelationHParams.from_relation(model, relation)
