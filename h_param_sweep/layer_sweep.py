import sys

sys.path.append("..")

import argparse
import json
import os
from typing import List

from src import data, models
from src.benchmarks import faithfulness
from src.functional import make_prompt, predict_next_token
from src.lens import causal_tracing, layer_c_measure
from src.operators import JacobianIclMeanEstimator

import numpy as np
import torch
from tqdm.auto import tqdm


def filter_by_model_knowledge(
    mt: models.ModelAndTokenizer,
    relation_prompt: str,
    relation_samples: List[data.RelationSample],
) -> List[data.RelationSample]:
    model_knows = []
    for sample in relation_samples:
        top_prediction = predict_next_token(
            mt=mt, prompt=relation_prompt.format(sample.subject)
        )[0][0].token
        tick = sample.object.strip().startswith(top_prediction.strip())
        if tick:
            model_knows.append(sample)
    return model_knows


def choose_sample_pairs(
    samples: List[data.RelationSample],
) -> List[data.RelationSample]:
    idx_pair = np.random.choice(range(len(samples)), 2, replace=False)
    sample_pair: list = [list(samples)[i] for i in idx_pair]
    if sample_pair[0].object != sample_pair[1].object:
        return sample_pair  # if the objects are different, return
    return choose_sample_pairs(samples)  # otherwise, draw again


def main(args: argparse.Namespace) -> None:
    ###################################################
    FILTER_RELATIONS: list = [
        "country capital city",
        # "occupation",
        "person superhero name",
        "plays pro sport",
        "task executor",
        "comparative",
        "past tense of verb",
        "gender of name",
        "religion of a name",
    ]
    ###################################################

    results_path = f"{args.results_dir}/{args.model}"
    os.makedirs(f"{args.results_dir}/{args.model}", exist_ok=True)

    print("running on relations")
    dataset = data.load_dataset()
    dataset = data.RelationDataset(
        relations=[r for r in dataset.relations if r.name in FILTER_RELATIONS]
    )
    for d in dataset.relations:
        print(f"{d.name} : {len(d.samples)}")

    print("\n\nloading model")
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device)
    layer_names = models.determine_layer_paths(mt)
    print(
        f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}"
    )

    for relation in tqdm(dataset.relations):
        print("\n####################################################################")
        print(f"{relation.name}")
        icl_indices = np.random.choice(range(len(relation.samples)), 3, replace=False)
        icl_samples = [relation.samples[i] for i in icl_indices]

        log_dict: dict = {}

        icl_prompt = make_prompt(
            prompt_template=relation.prompt_templates[0],
            subject="{}",
            examples=icl_samples,
        )

        relation_known = filter_by_model_knowledge(
            mt, relation_prompt=icl_prompt, relation_samples=relation.samples
        )
        log_dict["samples_known"] = len(relation_known)
        print(f"relation known: {len(relation_known)}/{len(relation.samples)}")

        print("calculating layer completeness and contributions")
        layer_completeness: dict = {layer: [] for layer in layer_names}
        layer_contributions: dict = {layer: [] for layer in layer_names}

        for sample in tqdm(relation_known[: min(args.n_runs, len(relation_known))]):
            cur_completeness = layer_c_measure(
                mt, icl_prompt, sample.subject, measure="completeness"
            )
            cur_contributions = layer_c_measure(
                mt, icl_prompt, sample.subject, measure="contribution"
            )
            for layer in layer_names:
                layer_completeness[layer].append(cur_completeness[layer])
                layer_contributions[layer].append(cur_contributions[layer])

        log_dict["layer_completeness"] = layer_completeness
        log_dict["layer_contributions"] = layer_contributions

        print("performing causal tracing")
        test_samples = set(relation_known) - set(icl_samples)
        causal_tracing_results: dict = {layer: [] for layer in layer_names}

        for run in tqdm(range(args.n_runs)):
            sample_pair = choose_sample_pairs(list(test_samples))
            cur_result = causal_tracing(
                mt,
                prompt_template=icl_prompt,
                subject_original=sample_pair[0].subject,
                subject_corruption=sample_pair[1].subject,
            )

            for layer in layer_names:
                causal_tracing_results[layer].append(cur_result[layer])
        log_dict["causal_tracing"] = causal_tracing_results

        print("recording faithfuless on each layer")

        faithfulness_results: dict = {layer: [] for layer in layer_names}
        for layer_idx, layer_name in enumerate(layer_names):
            mean_estimator = JacobianIclMeanEstimator(
                mt=mt,
                h_layer=layer_idx,
                # bias_scale_factor=0.2       # so that the bias doesn't knock out the prediction too much in the direction of training examples
            )
            n_train = 3
            cur_faithfulness = faithfulness(
                estimator=mean_estimator,
                dataset=data.RelationDataset(relations=[relation]),
                n_trials=7,  # hard coded -- dafault args.n_runs=20 take too long
                n_train=n_train,
                k=5,
                desc=layer_name,
            )
            faithfulness_results[layer_name] = cur_faithfulness.metrics.__dict__

        log_dict["faithfulness"] = faithfulness_results
        with open(f"{results_path}/{relation.name}.json", "w") as f:
            json.dump(log_dict, f, indent=4)
        print("--------------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gptj", help="language model to use"
    )
    parser.add_argument("--device", type=str, default=None, help="device to use")
    parser.add_argument(
        "--n_runs", type=int, default=20, help="Number of runs to average over"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/local_arnab/Codes/relations/results/layer_sweep",
        help="results dir",
    )
    args = parser.parse_args()
    main(args)
