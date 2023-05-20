"""Test which samples the model knows.

This script is mostly used for testing and data curation.
"""
import argparse

from src import data, functional, models
from src.utils import logging_utils

import torch

KEPT_MARKER = "✅"
REMOVED_MARKER = "❌"


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args=args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, fp16=args.fp16, device=args.device)
    dataset = data.load_dataset_from_args(args)
    with torch.device(device):
        filtered = functional.filter_dataset_samples(
            mt=mt, dataset=dataset, n_icl_lm=args.n_icl, n_trials=args.n_trials
        )

        relations_by_name = {r.name: r for r in dataset.relations}
        filtered_by_name = {r.name: r for r in filtered.relations}
        for name in relations_by_name:
            if name not in filtered_by_name:
                print(f"{REMOVED_MARKER} {name}")
                continue
            print(f"{KEPT_MARKER} {name}")

            relation_samples = set(relations_by_name[name].samples)
            filtered_samples = set(filtered_by_name[name].samples)
            for sample in relation_samples:
                if sample in filtered_samples:
                    marker = KEPT_MARKER
                else:
                    marker = REMOVED_MARKER
                print("\t" + f"{marker} {sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test model's knowledge")
    parser.add_argument(
        "--n-icl",
        type=int,
        default=functional.DEFAULT_N_ICL_LM,
        help="number of icl examples",
    )
    parser.add_argument("--n-trials", type=int, default=3, help="number of trials")
    data.add_data_args(parser)
    logging_utils.add_logging_args(parser)
    models.add_model_args(parser)
    args = parser.parse_args()
