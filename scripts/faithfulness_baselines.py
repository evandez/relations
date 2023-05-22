import argparse
import json
import logging
import os

from src import data, functional, metrics, models
from src.operators import (
    JacobianEstimator,
    JacobianIclMeanEstimator,
    LearnedLinearEstimatorBaseline,
    LinearRelationOperator,
    OffsetEstimatorBaseline,
)
from src.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def evaluate(
    operator: LinearRelationOperator, test_set: data.Relation, k: int = 10
) -> dict:
    pred_objects = []
    test_objects = [x.object for x in test_set.samples]
    subject_to_pred = {}
    for sample in test_set.samples:
        preds = operator(subject=sample.subject, k=k)
        pred_objects.append([p.token for p in preds.predictions])
        subject_to_pred[sample.subject] = [p.token for p in preds.predictions]
    return {
        "recall": metrics.recall(pred_objects, test_objects),
        "predictions": subject_to_pred,
    }


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args=args)

    dataset = data.load_dataset()
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, fp16=args.fp16, device=device)
    logger.info(
        f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}"
    )

    hparams_path = f"{args.hparams_path}/{args.model}"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    N_TRIALS = args.n_trials
    N_TRAINING = args.n_training

    all_results = []

    for relation_hparams in os.listdir(hparams_path):
        with open(os.path.join(hparams_path, relation_hparams), "r") as f:
            hparams = json.load(f)
        logger.info(
            f"{hparams['relation_name']} | h_layer: {hparams['h_layer']} | beta: {hparams['beta']}"
        )
        result = {
            "relation_name": hparams["relation_name"],
            "h_layer": hparams["h_layer"],
            "beta": hparams["beta"],
        }
        cur_relation_dataset = dataset.filter(
            relation_names=[hparams["relation_name"]],
        )
        cur_relation_known_dataset = functional.filter_dataset_samples(
            mt=mt, dataset=cur_relation_dataset, n_icl_lm=N_TRAINING
        )
        if len(cur_relation_known_dataset.relations) == 0:
            logger.info("Skipping relation with no known samples")
            continue

        cur_relation = cur_relation_dataset[0]
        cur_relation_known = cur_relation_known_dataset[0]

        logger.info(
            f"known samples: {len(cur_relation_known.samples)}/{len(cur_relation.samples)}"
        )
        result["known_samples"] = len(cur_relation_known.samples)
        result["total_samples"] = len(cur_relation.samples)
        result["trials"] = []

        prompt_template = cur_relation_known.prompt_templates[0]
        logger.info(f"prompt template: {prompt_template}")
        result["prompt_template"] = prompt_template
        logger.info("")

        for trial in range(N_TRIALS):
            logger.info(f"trial {trial + 1}/{N_TRIALS}")
            train, test = cur_relation_known.split(size=N_TRAINING)
            logger.info(f"train: {[str(sample) for sample in train.samples]}")

            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                examples=train.samples,
                subject="{}",
            )

            trial_results = {
                "icl_prompt": icl_prompt,
                "train": [
                    {
                        "subject": sample.subject,
                        "object": sample.object,
                    }
                    for sample in train.samples
                ],
                "zero_shot": {},  # W_r and b_r calculated without any ICL examples
                "logit_lens": {},  # F(h) = h
                "corner": {},  # F(h) = h + b
                "learned_linear": {},  # F(h) = Wh + b, W is learned with linear regression
                "icl_mean_emb": {},  # ICL-Mean but h set to embedding
                "icl_mean": {},  # flagship method
            }

            zero_shot_estimator = JacobianEstimator(
                mt=mt,
                h_layer=hparams["h_layer"],
                beta=hparams["beta"],
            )
            zero_shot_operator = zero_shot_estimator.estimate_for_subject(
                subject=train.samples[0].subject,
                prompt_template=prompt_template,
            )
            zero_shot_recall = evaluate(zero_shot_operator, test)
            logger.info(f"zero shot recall: {zero_shot_recall['recall']}")
            trial_results["zero_shot"] = zero_shot_recall

            logit_lens_operator = LinearRelationOperator(
                mt=mt,
                h_layer=hparams["h_layer"],
                weight=None,
                bias=None,
                prompt_template=icl_prompt,
                z_layer=-1,
            )
            logit_lens_recall = evaluate(logit_lens_operator, test)
            logger.info(f"logit lens recall: {logit_lens_recall['recall']}")
            trial_results["logit_lens"] = logit_lens_recall

            offset_estimator = OffsetEstimatorBaseline(
                mt=mt,
                h_layer=hparams["h_layer"],
            )
            offset_operator = offset_estimator(train)
            offset_recall = evaluate(offset_operator, test)
            logger.info(f"offset recall: {offset_recall['recall']}")
            trial_results["corner"] = offset_recall

            learned_estimator = LearnedLinearEstimatorBaseline(
                mt=mt,
                h_layer=hparams["h_layer"],
            )
            learned_operator = learned_estimator(train)
            learned_recall = evaluate(learned_operator, test)
            logger.info(f"learned recall: {learned_recall['recall']}")
            trial_results["learned_linear"] = learned_recall

            mean_emb_estimator = JacobianIclMeanEstimator(
                mt=mt,
                h_layer="emb",
                beta=hparams["beta"],
            )
            mean_emb_operator = mean_emb_estimator(train)
            mean_emb_recall = evaluate(mean_emb_operator, test)
            logger.info(f"icl mean recall (emb): {mean_emb_recall['recall']}")
            trial_results["icl_mean_emb"] = mean_emb_recall

            mean_estimator = JacobianIclMeanEstimator(
                mt=mt,
                h_layer=hparams["h_layer"],
                beta=hparams["beta"],
            )
            mean_operator = mean_estimator(train)
            mean_recall = evaluate(mean_operator, test)
            logger.info(f"icl mean recall: {mean_recall['recall']}")
            trial_results["icl_mean"] = mean_recall

            result["trials"].append(trial_results)
            logger.info("")

        all_results.append(result)
        logger.info(
            "-----------------------------------------------------------------------"
        )
        logger.info("\n\n")

        with open(f"{save_dir}/gptj.json", "w") as f:
            json.dump(all_results, f, indent=4)

        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    parser.add_argument(
        "--hparams-path",
        type=str,
        default="hparams",
        help="path to hparams",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/faithfulness_baselines",
        help="path to save results",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="number of trials",
    )

    parser.add_argument(
        "--n-training",
        type=int,
        default=functional.DEFAULT_N_ICL_LM,
        help="number of training samples",
    )

    experiment_utils.add_experiment_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
