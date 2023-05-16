from typing import Literal

import numpy as np
from matplotlib import pyplot as plt


def plot_c_measure(
    c_measure: dict,
    measure: Literal["completeness", "contribution"],
    title: str | None = None,
) -> plt:
    for layer in c_measure.keys():
        c_measure[layer] = np.array(c_measure[layer])

    mean_richness = [c_measure[layer].mean() for layer in c_measure.keys()]
    low_richness = [c_measure[layer].min() for layer in c_measure.keys()]
    high_richness = [c_measure[layer].max() for layer in c_measure.keys()]

    plt.plot(mean_richness, color="blue")
    plt.fill_between(range(len(mean_richness)), low_richness, high_richness, alpha=0.2)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Layer")
    plt.ylabel(measure)
    plt.xticks(range(0, len(mean_richness), 2))
    if title is not None:
        plt.title(title)
    return plt


def plot_layer_wise_causal_tracing(
    causal_tracing_results: dict, title: str | None = None
) -> plt:
    for layer in causal_tracing_results.keys():
        causal_tracing_results[layer] = np.array(causal_tracing_results[layer])
    mean = [
        causal_tracing_results[layer].mean() for layer in causal_tracing_results.keys()
    ]
    # low = [causal_tracing_results[layer].min() for layer in mt.layer_names]
    # high = [causal_tracing_results[layer].max() for layer in mt.layer_names]

    plt.plot(mean, color="blue", linewidth=3)
    # plt.fill_between(range(len(mean)), low, high, alpha=0.2)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Layer")
    plt.ylabel("causal_score")
    plt.xticks(range(len(causal_tracing_results.keys()))[::2])
    if title is not None:
        plt.title(title)

    nrun = causal_tracing_results[list(causal_tracing_results.keys())[0]].shape[0]
    for run in range(nrun):
        arr = []
        for layer in causal_tracing_results.keys():
            arr.append(causal_tracing_results[layer][run])
        plt.plot(arr, alpha=0.2)
    return plt


def plot_layer_wise_faithfulness(
    faithfulness_result: dict, recall_upto: int = 3, title: str | None = None
) -> plt:
    linewidth: float = 3.0
    for recall_at in range(recall_upto):
        result = np.array(
            [
                np.array(faithfulness_result[layer])[recall_at]
                for layer in faithfulness_result.keys()
            ]
        )
        plt.plot(result, linewidth=linewidth, label=f"recall@{recall_at + 1}")
        linewidth /= 2

    plt.xticks(range(0, len(result), 2))
    plt.xlabel("layer")
    plt.ylabel("faithfulness")
    plt.legend()
    plt.title(title)

    return plt
