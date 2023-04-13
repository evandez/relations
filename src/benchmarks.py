from dataclasses import dataclass

from src import data, operators


@dataclass(frozen=True, kw_only=True)
class FaithfulnessBenchmarkResults:
    pass


def faithfulness(
    estimator: operators.Estimator,
    dataset: data.RelationDataset,
) -> FaithfulnessBenchmarkResults:
    pass


@dataclass(frozen=True, kw_only=True)
class ReconstructionBenchmarkResults:
    pass


def reconstruction(
    estimator: operators.Estimator,
    dataset: data.RelationDataset,
) -> ReconstructionBenchmarkResults:
    pass
