import argparse
import logging
import os
import random
import sys
from typing import Literal

from src import data, functional, models
from src.utils import experiment_utils, logging_utils, typing
from src.utils.sweep_utils import read_sweep_results, relation_from_dict

import torch

logger = logging.getLogger(__name__)


def main(
    relation_name: str,
    model_name: Literal["gptj", "llama-13b", "mamba-3b"] = "mamba-3b",
    h_layer: typing.Layer | None = None,
    n_icl: int = 5,
    limit_approxes=40,
    save_dir: str = "results/cached_o1_approxes",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" not in device:
        logger.warning("!!! running on CPU, this will be slow !!!")
    mt = models.load_model(name=model_name, fp16=False, device=device)
    relation = data.load_dataset().filter(relation_names=[relation_name])[0]
    prompt_template = relation.prompt_templates[0]
    relation = relation.set(prompt_templates=[prompt_template])

    assert h_layer is not None

    relation = functional.filter_relation_samples(
        mt=mt,
        relation=relation,
        prompt_template=prompt_template,
        n_icl_lm=n_icl,
    )

    logger.info(
        f"filtered {len(relation.samples)} test samples with {n_icl} ICL examples"
    )

    assert len(relation.samples) > 1, "not enough number of test samples to interpolate"
    save_dir = os.path.join(
        save_dir, model_name, relation_name.lower().replace(" ", "_"), str(h_layer)
    )
    os.makedirs(save_dir, exist_ok=True)

    samples = relation.samples
    limit_approxes = min(limit_approxes, len(samples))
    num_saved = 0
    num_skipped = 0
    random.shuffle(samples)
    for sample in samples:
        icl_examples = random.choices(
            list(set(samples) - {sample}), k=min(2, len(samples) - 1)
        )
        prompt = functional.make_prompt(
            mt=mt,
            prompt_template=prompt_template,
            subject=sample.subject,
            examples=icl_examples,
        )

        prediction = functional.predict_next_token(
            mt=mt,
            prompt=prompt,
        )
        top_pred = prediction[0][0]
        prediction_info = f"{sample.subject} -> {sample.object} | prediction: `{top_pred.token}` [p={top_pred.prob:.2f}]"
        known = functional.is_nontrivial_prefix(
            prediction=prediction[0][0].token, target=sample.object
        )
        tick = functional.get_tick_marker(known)
        prediction_info += f" ({tick})"
        logger.info(prediction_info)
        if not known:
            logger.info(" ---- skipping unknown prediction ---- ")
            num_skipped += 1
            continue

        h_index, inputs = functional.find_subject_token_index(
            mt=mt,
            prompt=prompt,
            subject=sample.subject,
        )
        logger.debug(f"note that subject={sample.subject}, h_index={h_index}")

        order_1_approx = functional.order_1_approx(
            mt=mt,
            prompt=prompt,
            h_layer=h_layer,
            h_index=h_index,
            inputs=inputs,
        )

        functional.save_linear_operator(
            approx=order_1_approx,
            file_name=sample.subject.lower().replace(" ", "_"),
            path=save_dir,
            metadata={
                "icl_examples": [example.to_dict() for example in icl_examples],
                "sample": sample.to_dict(),
            },
        )
        num_saved += 1

        logger.info(
            f">>>>>> saved {num_saved}/{limit_approxes} approxes [skipped {num_skipped}] <<<<<<<<"
        )

        if num_saved == limit_approxes:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cache Order-1 approximations for faster sweeps"
    )
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument("--rel", type=str, help="filter by relation name")

    parser.add_argument(
        "--n-icl",
        type=int,
        default=3,
        help="number of few-shot examples to provide",
    )

    parser.add_argument(
        "--h-layer",
        type=int,
        default=20,
        help="h layer",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/cached_o1_approxes",
        help="directory to cache values",
    )

    parser.add_argument(
        "--limit-approxes",
        type=int,
        default=5,
        help="number of approxes to cache per relation",
    )

    args = parser.parse_args()
    logging_utils.configure(args=args)
    experiment = experiment_utils.setup_experiment(args)

    logger.info(args)

    main(
        relation_name=args.rel,
        model_name=args.model,
        h_layer=args.h_layer,
        n_icl=args.n_icl,
        save_dir=args.save_dir,
        limit_approxes=args.limit_approxes,
    )