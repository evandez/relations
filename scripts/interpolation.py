import copy
import json
import logging
import os
import sys

from src import data, functional, lens, metrics, models, operators, utils
from src.utils import experiment_utils, logging_utils

import baukit
import numpy as np
import torch
from tqdm.auto import tqdm

experiment_utils.set_seed(123456)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
    stream=sys.stdout,
)


def save_order_1_approx(
    approx: functional.Order1ApproxOutput,
    file_name: str = "order_1_approx",
    path: str = "../results/interpolation",
) -> None:
    os.makedirs(path, exist_ok=True)
    detached = {}
    for k, v in approx.__dict__.items():
        if isinstance(v, torch.Tensor):
            detached[k] = v.detach().cpu().numpy()
        else:
            detached[k] = v
    if file_name.endswith(".npz") == False:
        file_name = file_name + ".npz"
    np.savez(f"{path}/{file_name}", **detached)


def main(
    relation_name: str,
    h_layer: int = 8,
    interpolation_steps: int = 100,
    n_training: int = 5,
    n_trials: int = 10,
) -> None:
    mt = models.load_model(name="gptj", fp16=True, device="cuda")
    relation = (
        data.load_dataset()
        .filter(relation_names=[relation_name])[0]
        .set(prompt_templates=[" {}:"])
    )
    train, test = relation.split(n_training)
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
    logger.info(f"filtered {len(test.samples)} test samples based on provided fewshots")
    test_subjects = [sample.subject for sample in test.samples]
    hs_and_zs = functional.compute_hs_and_zs(
        mt=mt,
        prompt_template=train.prompt_templates[0],
        subjects=[sample.subject for sample in relation.samples],
        h_layer=h_layer,
        z_layer=-1,
        examples=train.samples,
    )

    for trial in range(n_trials):
        logger.info(f"Trial: {trial+1}/{n_trials}")
        s1, s2 = np.random.choice(test_subjects, size=2, replace=False)
        h1, h2 = [hs_and_zs.h_by_subj[s] for s in [s1, s2]]
        logger.info(f"Subjects: {s1}, {s2} | distance: {torch.dist(h1, h2)}")
        H: list[torch.Tensor] = []
        for alpha in np.linspace(0, 1, interpolation_steps):
            H.append(h1 * (1 - alpha) + h2 * alpha)
        H = torch.stack(H)
        h_index, inputs = functional.find_subject_token_index(
            mt=mt, prompt=icl_prompt.format(s1), subject=s1
        )

        for idx, h in tqdm(enumerate(H)):
            approx = functional.order_1_approx(
                mt=mt,
                prompt=icl_prompt.format(s1),
                h_layer=h_layer,
                h_index=h_index,
                h=h,
            )
            save_order_1_approx(
                approx,
                file_name=f"approx_{idx+1}",
                path=f"results/interpolation/{s1}-{s2}",
            )
            w_norm = approx.weight.norm()
            b_norm = approx.bias.norm()
            jh_norm = approx.metadata["Jh"].norm()
            top_predictions, _ = lens.logit_lens(mt, approx.z, get_proba=True, k=3)
            logger.info(
                f"{idx+1} => {w_norm=:.2f} | {b_norm=:.2f} | {jh_norm=:.2f} | {top_predictions=}"
            )
        print("------------------------------------------------------")
        print()


if __name__ == "__main__":
    main(relation_name="country capital city")
