import itertools
import random

from src import models
from src.data import Relation, RelationSample
from src.functional import make_prompt
from src.lens import causal_tracing

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from tqdm import tqdm

H_PARAMS = {
    "gpt-j-6B": {
        "layer": {
            "default": 13,
            "max": 16,
            "good_range": (6, 16),
        }
    },
    "gpt2-xl": {
        "layer": {
            "default": 26,
            "max": 32,
            "good_range": (12, 32),
        }
    },
}


def sample_from_each_range(
    samples: list[RelationSample], n_sample: int = -1  # -1 means sample from all
) -> list[RelationSample]:
    traverse_order = np.random.permutation(range(len(samples)))
    drawn_samples = []
    drawn_range = set()
    for idx in traverse_order:
        if samples[idx].object not in drawn_range:
            drawn_samples.append(samples[idx])
            drawn_range.add(samples[idx].object)
            if len(drawn_range) == n_sample:
                break
    return drawn_samples


def select_layer(
    mt: models.ModelAndTokenizer,
    training_data: Relation,
    n_run: int = 20,
    n_icl: int = 3,
    knee_smooth_factor: int = 3,
    verbose: bool = False,
) -> int:
    if len(training_data.range) == 1:
        raise AssertionError("Range of training data is 1. Can't do causal tracing")
    model_name = mt.model.config._name_or_path.split("/")[-1]
    if model_name not in H_PARAMS:
        raise AssertionError(
            f"Unknown model {model_name} => can't select layer automatically"
        )

    good_layers = range(*H_PARAMS[model_name]["layer"]["good_range"])

    sample_pairs = []
    for idx in range(len(training_data.samples)):
        for jdx in range(idx + 1, len(training_data.samples)):
            if training_data.samples[idx].object != training_data.samples[jdx].object:
                sample_pairs.append(
                    (training_data.samples[idx], training_data.samples[jdx])
                )

    prompt_templates = []
    for p in training_data.prompt_templates:
        prompt_templates.append(p)
        if len(prompt_templates) * len(sample_pairs) >= n_run:
            break

    trace_config = list(itertools.product(sample_pairs, prompt_templates))
    random.shuffle(trace_config)

    causal_tracing_results: dict = {
        layer: [] for layer in models.determine_layer_paths(mt)
    }
    for sample_pair, prompt_template in tqdm(
        trace_config[: min(len(trace_config), n_run)],
        desc="searching for optimal layer",
    ):
        # print(sample_pair)
        icl_examples = sample_from_each_range(
            samples=list(set(training_data.samples) - set(sample_pair)), n_sample=n_icl
        )
        _prompt = (
            make_prompt(
                prompt_template=prompt_template,
                subject="{}",
                examples=icl_examples,
            )
            if len(icl_examples) > 0
            else prompt_template
        )
        # print(_prompt)
        cur_result = causal_tracing(
            mt,
            prompt_template=_prompt,
            subject_original=sample_pair[0].subject,
            subject_corruption=sample_pair[1].subject,
            object_original=sample_pair[0].object,
        )
        for layer in models.determine_layer_paths(mt):
            causal_tracing_results[layer].append(cur_result[layer])

    layer_causal_score = [
        np.array(causal_tracing_results[layer]).mean()
        for layer in models.determine_layer_paths(mt)
    ]

    # if(verbose):
    #     plot_layer_wise_causal_tracing(causal_tracing_results, title=f"causal tracing on {training_data.name}")
    #     plt.show()
    #     plt.plot([
    #         np.array(layer_causal_score[i - knee_smooth_factor : i]).mean()
    #         for i in range(len(layer_causal_score))
    #     ])
    #     plt.show()

    smoothed_causal_score = np.array(
        [
            np.array(layer_causal_score[i - knee_smooth_factor : i]).mean()
            for i in good_layers
        ]
    )

    kneedle = KneeLocator(
        x=list(good_layers),
        y=smoothed_causal_score,
        curve="concave",
        direction="decreasing",
    )

    if verbose:
        plt.xticks(range(len(smoothed_causal_score)), list(good_layers))
        plt.plot(smoothed_causal_score)
        print(f"Knee: {kneedle.knee}, Elbow: {kneedle.elbow}")

    if kneedle.knee is not None:
        return int(min(kneedle.knee, H_PARAMS[model_name]["layer"]["max"]))
    else:
        # can't find knee, fallback to default layer for the model
        return H_PARAMS[model_name]["layer"]["default"]
