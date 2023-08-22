import argparse
import copy
import json
import logging
import os
import sys

from src import data, functional, lens, metrics, models, operators, utils
from src.utils import experiment_utils, logging_utils, typing
from src.utils.sweep_utils import read_sweep_results, relation_from_dict

import baukit
import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def save_order_1_approx(
    approx: functional.Order1ApproxOutput | operators.LinearRelationOperator,
    file_name: str = "order_1_approx",
    path: str = "../results/interpolation",
) -> None:
    os.makedirs(path, exist_ok=True)
    detached = {}
    for k, v in approx.__dict__.items():
        if k == "mt":  # will save the whole model and weights otherwise
            continue
        if isinstance(v, torch.Tensor):
            detached[k] = v.detach().cpu().numpy()
        else:
            detached[k] = v
    if file_name.endswith(".npz") == False:
        file_name = file_name + ".npz"
    np.savez(f"{path}/{file_name}", **detached)


def normalize_on_sphere(h: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    lh = (h - h.mean(dim=0)) / h.std(dim=0)
    return scale * lh / lh.norm(dim=0) if scale is not None else lh


def main(
    relation_name: str,
    h_layer: typing.Layer | None = None,
    interpolation_steps: int = 100,
    n_few_shot: int = 5,  # filter known samples from test set with n_few_shot examples
    n_trials: int = 10,
) -> None:
    mt = models.load_model(name="gptj", fp16=True, device="cuda")
    relation = data.load_dataset().filter(relation_names=[relation_name])[0]
    prompt_template = relation.prompt_templates[0]
    # prompt_template = " {} :"  # bare prompt with colon
    relation = relation.set(prompt_templates=[prompt_template])

    if h_layer is None:
        sweep_path = f"results/sweep-24-trials/gptj"
        relation_result_raw = read_sweep_results(
            sweep_path, relation_names=[relation_name], economy=True
        )[relation_name]
        relation_result = relation_from_dict(relation_result_raw)
        hparams = relation_result.best_by_efficacy(beta=2.25)
        h_layer = hparams.layer

    assert h_layer is not None

    train, test = relation.split(n_few_shot)
    icl_prompt = functional.make_prompt(
        prompt_template=train.prompt_templates[0],
        subject="{}",
        examples=train.samples,
        mt=mt,
    )
    logger.info(icl_prompt)
    test = functional.filter_relation_samples_based_on_provided_fewshots(
        mt=mt, test_relation=test, prompt_template=icl_prompt, batch_size=4
    )
    logger.info(f"filtered {len(test.samples)} test samples")

    assert len(test.samples) > 1, "not enough number of test samples to interpolate"

    if len(test.samples) < n_trials:
        logger.warning(
            f"not enough number of test samples ({len(test.samples)}), n_trials={n_trials}"
        )
        n_trials = len(test.samples)

    test_targets = functional.random_edit_targets(test.samples)

    test_subjects = [sample.subject for sample in test.samples]
    hs_and_zs = functional.compute_hs_and_zs(
        mt=mt,
        prompt_template=train.prompt_templates[0],
        subjects=[sample.subject for sample in relation.samples],
        h_layer=h_layer,
        z_layer=-1,
        examples=train.samples,
    )

    s1_samples = np.random.choice(test.samples, size=n_trials, replace=False)  # type: ignore

    for trial in range(n_trials):
        logger.info(f"Trial: {trial+1}/{n_trials}")
        # s1, s2 = np.random.choice(test_subjects, size=2, replace=False)
        sample_1 = s1_samples[trial]
        sample_2 = test_targets[s1_samples[trial]]
        logger.info(f"{sample_1} <to> {sample_2}")

        s1, s2 = sample_1.subject, sample_2.subject
        h1, h2 = [hs_and_zs.h_by_subj[s] for s in [s1, s2]]
        logger.info(f"distance: {torch.dist(h1, h2)}")
        H: list[torch.Tensor] = []
        for alpha in np.linspace(0, 1, interpolation_steps):
            H.append(h1 * (1 - alpha) + h2 * alpha)
        H = torch.stack(H)  # type: ignore
        h_index, inputs = functional.find_subject_token_index(
            mt=mt, prompt=icl_prompt.format(s1), subject=s1
        )

        for idx, h in enumerate(H):
            approx = functional.order_1_approx(
                mt=mt,
                prompt=icl_prompt.format(s1),
                h_layer=h_layer,
                h_index=h_index,
                h=normalize_on_sphere(
                    h, scale=65.0  # scale is selected by eyeballing some examples
                ),
            )
            save_order_1_approx(
                approx,
                file_name=f"approx_{idx+1}",
                path=f"results/interpolation/{relation_name}/{s1}-{s2}",
            )
            w_norm = approx.weight.norm()
            b_norm = approx.bias.norm()
            jh_norm = approx.metadata["Jh"].norm()
            top_predictions, _ = lens.logit_lens(mt, approx.z, get_proba=True, k=3)
            logger.info(
                f"{idx+1} => {w_norm=:.2f} | {b_norm=:.2f} | {jh_norm=:.2f} | {top_predictions=}"
            )
        logger.info("------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate h, z, and dF for poiints interpolated from s1 to s2"
    )
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/faithfulness_baselines_updated",
        help="directory to cache values",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="number of interpolation steps between s1 and s2",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="number of s1, s2 pairs to interpolate between",
    )

    parser.add_argument(
        "--n-icl",
        type=int,
        default=10,
        help="number of few-shot examples to provide",
    )

    parser.add_argument(
        "--rel-names", "-r", nargs="+", type=str, help="filter by relation name"
    )

    args = parser.parse_args()
    logging_utils.configure(args=args)
    experiment = experiment_utils.setup_experiment(args)

    logger.info(args)

    for relation_name in args.rel_names:
        logger.info(f"#################### {relation_name} ####################")
        main(
            relation_name=relation_name,
            interpolation_steps=args.n_steps,
            n_few_shot=args.n_icl,
            n_trials=args.n_trials,
        )
        logger.info("------------------------------------------------------")
