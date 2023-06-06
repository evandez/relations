import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

from src import data, metrics
from src.data import RelationSample
from src.utils.typing import Layer

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger(__name__)


####################### sweep dataclasses #######################
@dataclass(frozen=True)
class SweepBetaResults(DataClassJsonMixin):
    beta: float
    recall: list[float]
    faithfulness_successes: list[data.RelationSample]


@dataclass(frozen=True)
class EfficacyTestPair(DataClassJsonMixin):
    source: data.RelationSample
    target: data.RelationSample


@dataclass(frozen=True)
class SweepRankResults(DataClassJsonMixin):
    rank: int
    efficacy: list[float]
    efficacy_successes: list[EfficacyTestPair]


@dataclass(frozen=True)
class SweepTrainResults(DataClassJsonMixin):
    samples: list[data.RelationSample]
    betas: list[SweepBetaResults]
    ranks: list[SweepRankResults]
    jh_norm: float

    def best_beta(self, k: int = 1) -> SweepBetaResults:
        """Return the best beta by given recall position."""
        return max(self.betas, key=lambda x: x.recall[k - 1])

    def best_rank(self, k: int = 1) -> SweepRankResults:
        """Return the best rank by efficacy."""
        assert self.ranks is not None
        return max(self.ranks, key=lambda x: x.efficacy[k - 1])

    def summarize(self) -> None:
        """Sumarize results in debug logs."""
        best_beta = self.best_beta()
        best_rank = self.best_rank()
        logger.info(
            "layer finished | "
            f"beta={best_beta.beta:.2f} | recall@1={best_beta.recall[0]:.2f} | "
            f"rank={best_rank.rank} | efficacy={best_rank.efficacy[0]:.2f} | "
            f"norm(Jh)={self.jh_norm:.2f} | "
            f"samples={[str(x) for x in self.samples]}"
        )


@dataclass(frozen=True)
class SweepLayerResults(DataClassJsonMixin):
    layer: Layer
    result: SweepTrainResults


@dataclass(frozen=True)
class SweepTrialResults(DataClassJsonMixin):
    prompt_template: str
    train_samples: list[data.RelationSample]
    layers: list[SweepLayerResults]
    n_test_samples: int
    efficacy_trials: list[EfficacyTestPair]


@dataclass(frozen=True)
class SweepLayerSummary(DataClassJsonMixin):
    layer: Layer
    beta: metrics.AggregateMetric
    recall: metrics.AggregateMetric
    rank: metrics.AggregateMetric
    efficacy: metrics.AggregateMetric


@dataclass(frozen=True)
class SweepRelationResults(DataClassJsonMixin):
    relation_name: str
    trials: list[SweepTrialResults]

    def by_layer(self, k: int = 1) -> dict[Layer, SweepLayerSummary]:
        """Return best layer and average beta for that layer."""
        results_by_layer = defaultdict(list)
        for trial in self.trials:
            for layer in trial.layers:
                best_beta = layer.result.best_beta()
                best_rank = layer.result.best_rank()
                results_by_layer[layer.layer].append(
                    (
                        layer.layer,
                        best_beta.beta,
                        best_beta.recall[k - 1],
                        best_rank.rank,
                        best_rank.efficacy[k - 1],
                    )
                )

        betas_by_layer = {
            layer: metrics.AggregateMetric.aggregate([x[1] for x in results])
            for layer, results in results_by_layer.items()
        }
        recalls_by_layer = {
            layer: metrics.AggregateMetric.aggregate([x[2] for x in results])
            for layer, results in results_by_layer.items()
        }
        ranks_by_layer = {
            layer: metrics.AggregateMetric.aggregate([x[3] for x in results])
            for layer, results in results_by_layer.items()
        }
        efficacies_by_layer = {
            layer: metrics.AggregateMetric.aggregate([x[4] for x in results])
            for layer, results in results_by_layer.items()
        }

        return {
            layer: SweepLayerSummary(
                layer=layer,
                beta=betas_by_layer[layer],
                recall=recalls_by_layer[layer],
                rank=ranks_by_layer[layer],
                efficacy=efficacies_by_layer[layer],
            )
            for layer in recalls_by_layer
        }

    def best_by_faithfulness(self, k: int = 1) -> SweepLayerSummary:
        """Return the best layer and average beta for that layer."""
        results_by_layer = self.by_layer(k=k)
        best_layer = max(
            results_by_layer, key=lambda x: results_by_layer[x].recall.mean
        )
        return results_by_layer[best_layer]

    def best_by_efficacy(self, k: int = 1) -> SweepLayerSummary:
        """Return the best layer and average beta for that layer."""
        results_by_layer = self.by_layer(k=k)
        best_layer = max(
            results_by_layer, key=lambda x: results_by_layer[x].efficacy.mean
        )
        return results_by_layer[best_layer]

    def summarize(self, k: int = 1) -> None:
        """Print a summary of what happened."""
        results_by_layer = self.by_layer(k=k)
        logger.debug(f'summarizing results for "{self.relation_name}"')
        for la, summ in results_by_layer.items():
            logger.info(
                f"layer={la} | beta={summ.beta.mean:.2f} | recall@{k}={summ.recall.mean:.2f} | "
                f"rank={summ.rank.mean:.2f} | efficacy={summ.efficacy.mean:.2f}"
            )


@dataclass(frozen=True)
class SweepResuts(DataClassJsonMixin):
    relations: list[SweepRelationResults]


####################### sweep dataclasses #######################

####################### causality baseline sweep dataclasses #######################


@dataclass(frozen=True)
class EfficacyBaselineLayerResult(DataClassJsonMixin):
    layer: Layer
    efficacy: float
    rank: int
    results: dict[str, float]


@dataclass(frozen=True)
class EfficacyBaselineTrialResult(DataClassJsonMixin):
    train_samples: list[data.RelationSample]
    prompt_template: str
    layerwise_baseline_results: list[EfficacyBaselineLayerResult]


@dataclass(frozen=True)
class EfficacyBaselineRelationResult(DataClassJsonMixin):
    relation_name: str
    trials: list[EfficacyBaselineTrialResult]


@dataclass(frozen=True)
class EfficacyBaselineResults(DataClassJsonMixin):
    relations: list[EfficacyBaselineRelationResult]


####################### causality baseline sweep dataclasses #######################


####################### read and parse the sweep results #######################
def parse_results(sweep_result: dict) -> SweepRelationResults:
    relation_results = SweepRelationResults(
        relation_name=sweep_result["relation_name"], trials=[]
    )

    for trial in sweep_result["trials"]:
        trial_results = SweepTrialResults(
            prompt_template=trial["prompt_template"],
            train_samples=[RelationSample.from_dict(s) for s in trial["train_samples"]],
            layers=[],
            n_test_samples=trial["n_test_samples"],
            efficacy_trials=[
                EfficacyTestPair(
                    source=RelationSample.from_dict(s["source"]),
                    target=RelationSample.from_dict(s["target"]),
                )
                for s in trial["efficacy_trials"]
            ]
            if "efficacy_trials" in trial
            else [],
        )
        for layer in trial["layers"]:
            train_results = SweepTrainResults(
                samples=[
                    RelationSample.from_dict(s) for s in layer["result"]["samples"]
                ],
                betas=[],
                ranks=[],
                jh_norm=layer["result"]["jh_norm"],
            )
            for beta in layer["result"]["betas"]:
                beta_results = SweepBetaResults(
                    beta=beta["beta"],
                    recall=beta["recall"],
                    faithfulness_successes=[
                        RelationSample.from_dict(s)
                        for s in beta["faithfulness_successes"]
                    ],
                )
                train_results.betas.append(beta_results)

            for rank in layer["result"]["ranks"]:
                rank_results = SweepRankResults(
                    rank=rank["rank"],
                    efficacy=rank["efficacy"]
                    if isinstance(rank["efficacy"], list)
                    else [rank["efficacy"]],
                    efficacy_successes=[
                        EfficacyTestPair(
                            source=RelationSample.from_dict(s["source"]),
                            target=RelationSample.from_dict(s["target"]),
                        )
                        for s in rank["efficacy_successes"]
                    ],
                )
                train_results.ranks.append(rank_results)

            layer_results = SweepLayerResults(
                layer=layer["layer"], result=train_results
            )

            trial_results.layers.append(layer_results)
        relation_results.trials.append(trial_results)
    return relation_results


def read_sweep_results(sweep_path: str) -> dict:
    sweep_results = {}

    for relation_folder in os.listdir(sweep_path):
        cur_sweep = f"{sweep_path}/{relation_folder}"
        if "results_all.json" not in os.listdir(cur_sweep):
            continue
        with open(f"{cur_sweep}/results_all.json") as f:
            res = json.load(f)["relations"]
            if len(res) == 0:
                continue
            res = res[0]
            sweep_results[res["relation_name"]] = res
    return sweep_results


####################### read and parse the sweep results #######################
