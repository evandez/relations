import argparse
import json
import logging
import os
from typing import Sequence

from scripts.faithfulness_baselines import evaluate, get_h, load_raw_results
from src import data, functional, models
from src.operators import Word2VecIclEstimator
from src.utils import experiment_utils, logging_utils, tokenizer_utils
from src.utils.sweep_utils import read_sweep_results, relation_from_dict
from src.utils.typing import Layer

import baukit
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, fp16=args.fp16, device=device)
    dataset = data.load_dataset()
    baseline_results = load_raw_results(
        model_name=args.model, results_path=args.faith_base_dir
    )

    relation_result_updated = []
    for relation_result in tqdm(baseline_results):
        prompt_template = relation_result["prompt_template"]
        relation_name = relation_result["relation_name"]
        if args.rel_names and relation_name not in args.rel_names:
            logger.warning(f"Skipping {relation_name}")
            continue

        h_layer = relation_result["h_layer"]
        h_layer_name = models.determine_layer_paths(mt, [h_layer])[0]
        relation_data = dataset.filter(
            relation_names=[relation_result["relation_name"]]
        )[0]
        logger.info(
            f"{relation_name} | h_layer={h_layer}({h_layer_name}) | {prompt_template=}"
        )
        for idx in range(len(relation_result["trials"])):
            logger.info(f"Trial {idx}/{len(relation_result['trials'])}")
            trial = relation_result["trials"][idx]
            train = relation_data.set(
                samples=[
                    data.RelationSample.from_dict(sample) for sample in trial["train"]
                ]
            )
            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                examples=train.samples,
                subject="{}",
            )
            test = relation_data.set(
                samples=list(set(relation_data.samples) - set(train.samples))
            )
            test = functional.filter_relation_samples_based_on_provided_fewshots(
                mt=mt, test_relation=test, prompt_template=icl_prompt
            )
            translation_estimator = Word2VecIclEstimator(
                mt=mt, h_layer=h_layer, mode="icl"
            )
            translation_operator = translation_estimator(train)
            trial["icl"]["translation"] = evaluate(
                operator=translation_operator,
                test_set=test,
                layer_name=h_layer_name,
            )

            hs_by_subj_zs = {
                sample.subject: get_h(
                    mt=mt,
                    prompt_template=mt.tokenizer.eos_token + " " + prompt_template,
                    subject=sample.subject,
                    layer_names=models.determine_layer_paths(mt, ["emb", h_layer]),
                )
                for sample in test.samples
            }
            trial["zero_shot"]["translation"] = evaluate(
                operator=translation_operator,
                test_set=test,
                hs_by_subj=hs_by_subj_zs,
                layer_name=models.determine_layer_paths(mt, [h_layer])[0],
            )
        relation_result_updated.append(relation_result)
        logger.info(
            "-----------------------------------------------------------------------------"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model}.json")
    logger.info(f"Saving updated results to {save_path}")
    with open(save_path, "w") as f:
        json.dump(relation_result_updated, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faithfulness baselines")
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--faith-base-dir",
        type=str,
        default="results/faithfulness_baselines",
        help="directory to find faithfulness baseline results, will update the results there",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/faithfulness_baselines_updated",
        help="directory to save results updated with translation baselines",
    )

    parser.add_argument(
        "--rel-names", "-r", nargs="+", type=str, help="filter by relation name"
    )

    args = parser.parse_args()
    logging_utils.configure(args=args)

    logger.info(args)
    main(args)
