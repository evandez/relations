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
from src.utils import experiment_utils, logging_utils, tokenizer_utils

import baukit
import torch

logger = logging.getLogger(__name__)


def get_h(
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subject: str,
    layer_names: list[str],
) -> dict[str, torch.Tensor]:
    prompt = prompt_template.format(subject)
    device = models.determine_device(mt)

    inputs = mt.tokenizer(
        [prompt],
        return_tensors="pt",
        padding="longest",
        return_offsets_mapping=True,
    ).to(device)

    with baukit.TraceDict(mt.model, layers=layer_names) as traces:
        outputs = mt.model(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    _, h_index = tokenizer_utils.find_token_range(
        prompt, subject, tokenizer=mt.tokenizer
    )
    h_index -= 1
    return {
        layer_name: functional.untuple(traces[layer_name].output)[0][h_index]
        for layer_name in layer_names
    }


def evaluate(
    operator: LinearRelationOperator,
    test_set: data.Relation,
    k: int = 10,
    hs_by_subj: dict[str, dict[str, torch.Tensor]] | None = None,
    layer_name: str = "emb",
) -> dict:
    pred_objects = []
    test_objects = [x.object for x in test_set.samples]
    subject_to_pred = {}
    for sample in test_set.samples:
        if hs_by_subj is None:
            preds = operator(subject=sample.subject, k=k)
        else:
            h = hs_by_subj[sample.subject][layer_name][None]
            preds = operator(subject=sample.subject, h=h, k=k)
        pred_objects.append([p.token for p in preds.predictions])
        subject_to_pred[sample.subject] = [p.token for p in preds.predictions]
    return {
        "recall": metrics.recall(pred_objects, test_objects),
        "predictions": subject_to_pred,
    }


def get_zero_shot_results(
    mt: models.ModelAndTokenizer,
    h_layer: int,
    test: data.Relation,
    operators: dict[str, LinearRelationOperator],
    hs_by_subj: dict[str, dict[str, torch.Tensor]],
) -> dict:
    print("------------ Zero Shot ------------")
    results: dict = {
        "logit_lens": {},  # F(h) = h
        "corner": {},  # F(h) = h + b
        "learned_linear": {},  # F(h) = Wh + b, W is learned with linear regression
        "lre_emb": {},  # ICL-Mean but h set to embedding
        "lre": {},  # ICL, don't do mean as it's zero shot
    }
    emb_layer_name, h_layer_name = models.determine_layer_paths(mt, ["emb", h_layer])
    for operator_name in operators:
        operator = operators[operator_name]
        layer_name = emb_layer_name if operator_name == "lre_emb" else h_layer_name
        recall = evaluate(operator, test, hs_by_subj=hs_by_subj, layer_name=layer_name)
        print(f"{operator_name}: {recall['recall']}")
        results[operator_name] = recall
    return results


def get_icl_results(
    mt: models.ModelAndTokenizer,
    h_layer: int,
    beta: float,
    train: data.Relation,
    test: data.Relation,
    icl_prompt: str,
    hs_by_subj: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict, dict]:
    print("------------ ICL ------------")
    results: dict = {
        "logit_lens": {},  # F(h) = h
        "corner": {},  # F(h) = h + b
        "learned_linear": {},  # F(h) = Wh + b, W is learned with linear regression
        "lre_emb": {},  # ICL-Mean but h set to embedding
        "lre": {},  # ICL, don't do mean as it's zero shot
    }
    emb_layer_name, h_layer_name = models.determine_layer_paths(mt, ["emb", h_layer])
    logit_lens_operator = LinearRelationOperator(
        mt=mt,
        h_layer=h_layer,
        weight=None,
        bias=None,
        prompt_template=icl_prompt,
        z_layer=-1,
    )
    logit_lens_recall = evaluate(
        logit_lens_operator, test, hs_by_subj=hs_by_subj, layer_name=h_layer_name
    )
    print(f"logit lens: {logit_lens_recall['recall']}")
    results["logit_lens"] = logit_lens_recall

    offset_estimator = OffsetEstimatorBaseline(mt=mt, h_layer=h_layer, mode="icl")
    offset_operator = offset_estimator(train)
    offset_recall = evaluate(
        offset_operator, test, hs_by_subj=hs_by_subj, layer_name=h_layer_name
    )
    print(f"corner: {offset_recall['recall']}")
    results["corner"] = offset_recall

    learned_estimator = LearnedLinearEstimatorBaseline(
        mt=mt,
        h_layer=h_layer,
        mode="icl",
    )
    learned_operator = learned_estimator(train)
    learned_recall = evaluate(
        learned_operator, test, hs_by_subj=hs_by_subj, layer_name=h_layer_name
    )
    print(f"learned: {learned_recall['recall']}")
    results["learned_linear"] = learned_recall

    lre_icl_emb_estimator = JacobianIclMeanEstimator(
        mt=mt,
        h_layer="emb",
        beta=beta,
    )
    lre_icl_emb_operator = lre_icl_emb_estimator(train)
    lre_emb_recall = evaluate(
        lre_icl_emb_operator, test, hs_by_subj=hs_by_subj, layer_name=emb_layer_name
    )
    print(f"LRE (emb): {lre_emb_recall['recall']}")
    results["lre_emb"] = lre_emb_recall

    lre_estimator = JacobianIclMeanEstimator(
        mt=mt,
        h_layer=h_layer,
        beta=beta,
    )
    lre_operator = lre_estimator(train)
    mean_recall = evaluate(
        lre_operator, test, hs_by_subj=hs_by_subj, layer_name=h_layer_name
    )
    print(f"LRE: {mean_recall['recall']}")
    results["lre"] = mean_recall

    return results, {
        "logit_lens": logit_lens_operator,
        "corner": offset_operator,
        "learned_linear": learned_operator,
        "lre_emb": lre_icl_emb_operator,
        "lre": lre_operator,
    }


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args=args)

    dataset = data.load_dataset_from_args(args)
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, fp16=args.fp16, device=device)
    print(
        f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}"
    )

    hparams_path = f"{args.hparams_path}/{mt.name}"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    N_TRIALS = args.n_trials
    N_TRAINING = args.n_training

    all_results = []

    for relation_hparams in os.listdir(hparams_path):
        with open(os.path.join(hparams_path, relation_hparams), "r") as f:
            hparams = json.load(f)
        print(
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
            print("Skipping relation with no known samples")
            continue

        cur_relation = cur_relation_dataset[0]
        cur_relation_known = cur_relation_known_dataset[0]

        print(
            f"known samples: {len(cur_relation_known.samples)}/{len(cur_relation.samples)}"
        )
        result["known_samples"] = len(cur_relation_known.samples)
        result["total_samples"] = len(cur_relation.samples)
        result["trials"] = []

        prompt_template = cur_relation_known.prompt_templates[0]
        print(f"prompt template: {prompt_template}")
        result["prompt_template"] = prompt_template
        print("")

        for trial in range(N_TRIALS):
            print(f"trial {trial + 1}/{N_TRIALS}")
            train, test = cur_relation_known.split(size=N_TRAINING)
            print(f"train: {[str(sample) for sample in train.samples]}")

            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                examples=train.samples,
                subject="{}",
            )
            print(icl_prompt)

            trial_results = {
                "icl_prompt": icl_prompt,
                "train": [
                    {
                        "subject": sample.subject,
                        "object": sample.object,
                    }
                    for sample in train.samples
                ],
                "zero_shot": {},
                "icl": {},
            }

            hs_by_subj_icl = {
                sample.subject: get_h(
                    mt=mt,
                    prompt_template=icl_prompt,
                    subject=sample.subject,
                    layer_names=models.determine_layer_paths(
                        mt, ["emb", hparams["h_layer"]]
                    ),
                )
                for sample in test.samples
            }

            trial_results["icl"], operators = get_icl_results(
                mt=mt,
                h_layer=hparams["h_layer"],
                beta=hparams["beta"],
                train=train,
                test=test,
                icl_prompt=icl_prompt,
                hs_by_subj=hs_by_subj_icl,
            )

            hs_by_subj_zs = {
                sample.subject: get_h(
                    mt=mt,
                    prompt_template=mt.tokenizer.eos_token + "{}",
                    subject=sample.subject,
                    layer_names=models.determine_layer_paths(
                        mt, ["emb", hparams["h_layer"]]
                    ),
                )
                for sample in test.samples
            }

            trial_results["zero_shot"] = get_zero_shot_results(
                mt=mt,
                h_layer=hparams["h_layer"],
                test=test,
                operators=operators,
                hs_by_subj=hs_by_subj_zs,
            )

            print("")

            result["trials"].append(trial_results)
            print("")

        all_results.append(result)
        print("-----------------------------------------------------------------------")
        print("\n\n")

        with open(f"{save_dir}/{mt.name}.json", "w") as f:
            json.dump(all_results, f, indent=4)

        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data.add_data_args(parser)
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--hparams-path",
        type=str,
        default="hparams",
        help="path to hparams",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/faithfulness_baselines_test",
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

    args = parser.parse_args()
    print(args)
    main(args)
