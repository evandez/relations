import sys

sys.path.append("..")

import argparse
import copy
import json
import os
from typing import List

from h_param_sweep.utils import select_subset_from_relation
from src import data, models
from src.benchmarks import faithfulness
from src.functional import make_prompt, predict_next_token
from src.lens import causal_tracing, layer_c_measure
from src.operators import JacobianIclMeanEstimator
from src.select_hparams import select_layer

import numpy as np
import torch
from tqdm.auto import tqdm


def main(args: argparse.Namespace) -> None:
    ###################################################
    FILTER_RELATIONS: list = [
        # "country capital city",
        # "occupation",
        "person superhero name",
        "plays pro sport",
        "landmark in country",
        "outside color of fruits and vegetables",
        # "work location",
        # "task done by person NEEDS REVISION",
        # "word comparative",
        # "word past tense",
        # "name gender",
        # "name religion",
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
    print(
        f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}"
    )

    for relation in tqdm(dataset.relations):
        print("\n####################################################################")
        print(f"relation name: {relation.name}")

        log_dict: dict = {}

        layer_selection_samples = select_subset_from_relation(
            relation=relation, n=min(len(relation.samples), args.n_layer_select)
        )

        optimal_layer = select_layer(
            mt=mt, training_data=layer_selection_samples, n_run=args.n_layer_select
        )
        print(f"optimal layer: {optimal_layer}")

        log_dict["optimal_layer"] = int(optimal_layer)
        log_dict["results"] = {}

        eval_relation = (
            relation
            if (
                args.max_eval_samples == -1
                or len(relation.samples) <= args.max_eval_samples
            )
            else select_subset_from_relation(relation, args.max_eval_samples)
        )

        for bias_scale_factor in np.linspace(0.1, 1.0, args.n_bias_steps):
            mean_estimator = JacobianIclMeanEstimator(
                mt=mt,
                h_layer=optimal_layer,
                bias_scale_factor=bias_scale_factor,
            )
            cur_faithfulness = faithfulness(
                estimator=mean_estimator,
                dataset=data.RelationDataset(relations=[eval_relation]),
                n_trials=7,
                n_train=3,
                k=5,
                desc=f"layer {optimal_layer}, bias scale: {bias_scale_factor}",
            )
            log_dict["results"][
                float(np.round(bias_scale_factor, 3))
            ] = cur_faithfulness.metrics.__dict__

            # clear memory
            del mean_estimator
            torch.cuda.empty_cache()

        with open(f"{results_path}/{relation.name}.json", "w") as f:
            json.dump(log_dict, f, indent=4)
        print("------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gptj", help="language model to use"
    )
    parser.add_argument("--device", type=str, default=None, help="device to use")
    parser.add_argument(
        "--n_layer_select",
        type=int,
        default=20,
        help="number of examples to use to select the layer",
    )
    parser.add_argument(
        "--n_bias_steps",
        type=int,
        default=10,
        help="number of bias scale steps to take between 0.1 and 1.0",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=200,
        help="maximum number of samples to use from each relation (-1 for all)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results/bias_scale_sweep",
        help="results dir",
    )
    args = parser.parse_args()
    print(args)
    main(args)
