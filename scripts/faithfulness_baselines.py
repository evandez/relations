import argparse
import json
import logging
import os
from typing import Sequence

from src import data, functional, metrics, models
from src.operators import (
    JacobianIclMeanEstimator,
    LearnedLinearEstimatorBaseline,
    LinearRelationOperator,
    OffsetEstimatorBaseline,
)
from src.utils import experiment_utils, logging_utils, tokenizer_utils
from src.utils.sweep_utils import read_sweep_results, relation_from_dict
from src.utils.typing import Layer

import baukit
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_h(
    mt: models.ModelAndTokenizer,
    prompt_template: str,
    subject: str,
    layer_names: Sequence[str],
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
        .detach()
        .cpu()
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
            h = hs_by_subj[sample.subject][layer_name][None].to(
                operator.mt.model.device
            )
            preds = operator(subject=sample.subject, h=h, k=k)
        logger.debug(
            f"testing {str(sample)} | preds={[str(p) for p in preds.predictions[:3]]}"
        )
        pred_objects.append([p.token for p in preds.predictions])
        subject_to_pred[sample.subject] = [p.token for p in preds.predictions]
    return {
        "recall": metrics.recall(pred_objects, test_objects),
        "predictions": subject_to_pred,
    }


def get_zero_shot_results(
    mt: models.ModelAndTokenizer,
    h_layer: Layer,
    test: data.Relation,
    operators: dict[str, LinearRelationOperator],
    hs_by_subj: dict[str, dict[str, torch.Tensor]],
) -> dict:
    logger.info("------------ Zero Shot ------------")
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
        logger.info(f"{operator_name}: {recall['recall']}")
        results[operator_name] = recall
    return results


def get_icl_results(
    mt: models.ModelAndTokenizer,
    h_layer: Layer,
    beta: float,
    train: data.Relation,
    test: data.Relation,
    icl_prompt: str,
    hs_by_subj: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict, dict]:
    logger.info("------------ ICL ------------")
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
    logger.info(f"logit lens: {logit_lens_recall['recall']}")
    results["logit_lens"] = logit_lens_recall

    offset_estimator = OffsetEstimatorBaseline(mt=mt, h_layer=h_layer, mode="icl")
    offset_operator = offset_estimator(
        train.set(samples=train.samples + test.samples)  # access to the full range
    )
    offset_recall = evaluate(
        offset_operator, test, hs_by_subj=hs_by_subj, layer_name=h_layer_name
    )
    logger.info(f"corner: {offset_recall['recall']}")
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
    logger.info(f"learned: {learned_recall['recall']}")
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
    logger.info(f"LRE (emb): {lre_emb_recall['recall']}")
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
    logger.info(f"LRE: {mean_recall['recall']}")
    results["lre"] = mean_recall

    return results, {
        "logit_lens": logit_lens_operator,
        "corner": offset_operator,
        "learned_linear": learned_operator,
        "lre_emb": lre_icl_emb_operator,
        "lre": lre_operator,
    }


from scripts.efficacy_baselines import filter_not_in_train_samples


def main(args: argparse.Namespace) -> None:
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, fp16=args.fp16, device=device)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    N_TRIALS = args.n_trials
    N_TRAINING = args.n_training

    sweep_results_dir = f"{args.sweep_results_dir}/{args.model}"
    sweep_results = read_sweep_results(sweep_results_dir, relation_names=args.rel_names)

    logger.info("found %d relations", len(sweep_results))
    logger.info(json.dumps(list(sweep_results.keys()), indent=4))

    dataset = data.load_dataset()

    all_relation_results = {}

    # these relations will go out of memory with large N_TRAINING is > 6
    OOM_relations = [
        "person father",
        "person mother",
    ]
    relation_sweeps = {}
    for trial in range(N_TRIALS):
        logger.info(
            "################################################################################"
        )
        logger.info(f"trial {trial + 1}/{N_TRIALS}")
        logger.info(
            "################################################################################"
        )
        for relation_name, sweep_result in tqdm(sweep_results.items()):
            if args.rel_names is not None and relation_name not in args.rel_names:
                logger.info("skipping %s", relation_name)
                continue
            logger.info("relation: %s", relation_name)
            if relation_name not in relation_sweeps:
                relation_sweeps[relation_name] = relation_from_dict(sweep_result)
            relation_sweep = relation_sweeps[relation_name]
            if len(relation_sweep.trials) < 3:
                logger.info(f"skipping {relation_name}, not enough trials")
                continue
            hparams = relation_sweep.best_by_faithfulness()
            logger.info(
                f"{relation_name} | h_layer: {hparams.layer} | beta: {hparams.beta.mean} +/- {hparams.beta.stderr} |>> expected lre recall: {hparams.recall.mean} +/- {hparams.recall.stderr}"
            )
            h_layer = hparams.layer
            beta = hparams.beta.mean
            # prompt_template = relation_known.prompt_templates[0]
            prompt_template = " {} :"
            relation = dataset.filter(relation_names=[relation_sweep.relation_name])[0]
            relation = relation.set(prompt_templates=[prompt_template])

            logger.info(
                f"total samples = {len(relation.samples)}, prompt template: {prompt_template}"
            )

            if relation_name not in all_relation_results:
                all_relation_results[relation_name] = {
                    "relation_name": relation_name,
                    "total_samples": len(relation.samples),
                    "prompt_template": prompt_template,
                    "h_layer": h_layer,
                    "beta": beta,
                    "expected_recall": hparams.recall.mean,
                    "trials": [],
                }

            relation_result = all_relation_results[relation_name]

            # Runs the numbers with the exact same train/test split as the n'th trial of the sweep
            # sweep_trial = relation_results.trials[trial]
            # train_samples = sweep_trial.train_samples
            # test_samples = [
            #     sample
            #     for sample in relation.samples
            #     if filter_not_in_train_samples(sample, train_samples)
            # ]
            # train_relation = relation.set(
            #     samples=train_samples, prompt_templates=[prompt_template]
            # )
            # test_relation = relation.set(
            #     samples=test_samples, prompt_templates=[prompt_template]
            # )

            # sample random train/test split for each trial
            train_relation, test_relation = relation.split(
                N_TRAINING
                if (relation_name not in OOM_relations or N_TRAINING < 6)
                else 6
            )

            logger.info(f"train: {[str(sample) for sample in train_relation.samples]}")

            icl_prompt = functional.make_prompt(
                mt=mt,
                prompt_template=prompt_template,
                examples=train_relation.samples,
                subject="{}",
            )
            logger.info(icl_prompt)

            test_relation = (
                functional.filter_relation_samples_based_on_provided_fewshots(
                    mt=mt,
                    test_relation=test_relation,
                    prompt_template=icl_prompt,
                    subj_token_filter="all",
                )
            )

            logger.info(
                f"known samples: {len(test_relation.samples)}/{len(relation.samples)}"
            )

            trial_results = {
                "icl_prompt": icl_prompt,
                "known": len(test_relation.samples),
                "train": [
                    {
                        "subject": sample.subject,
                        "object": sample.object,
                    }
                    for sample in train_relation.samples
                ],
                "zero_shot": {},
                "icl": {},
            }

            hs_by_subj_icl = {
                sample.subject: get_h(
                    mt=mt,
                    prompt_template=icl_prompt,
                    subject=sample.subject,
                    layer_names=models.determine_layer_paths(mt, ["emb", h_layer]),
                )
                for sample in test_relation.samples
            }

            trial_results["icl"], operators = get_icl_results(
                mt=mt,
                h_layer=h_layer,
                beta=beta,
                train=train_relation,
                test=test_relation,
                icl_prompt=icl_prompt,
                hs_by_subj=hs_by_subj_icl,
            )

            hs_by_subj_zs = {
                sample.subject: get_h(
                    mt=mt,
                    prompt_template=mt.tokenizer.eos_token + " {} :",
                    subject=sample.subject,
                    layer_names=models.determine_layer_paths(mt, ["emb", h_layer]),
                )
                for sample in test_relation.samples
            }

            trial_results["zero_shot"] = get_zero_shot_results(
                mt=mt,
                h_layer=h_layer,
                test=test_relation,
                operators=operators,
                hs_by_subj=hs_by_subj_zs,
            )

            relation_result["trials"].append(trial_results)

        logger.info(
            "-----------------------------------------------------------------------"
        )
        logger.info("\n\n")

        with open(f"{save_dir}/{mt.name}.json", "w") as f:
            json.dump(list(all_relation_results.values()), f, indent=4)

        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faithfulness baselines")

    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--sweep-results-dir",
        type=str,
        default="results/sweep",
        help="directory to find sweep results",
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
        default=24,
        help="number of trials",
    )

    parser.add_argument(
        "--n-training",
        type=int,
        default=functional.DEFAULT_N_ICL_LM,
        help="number of training samples",
    )

    parser.add_argument(
        "--rel-names", "-r", nargs="+", type=str, help="filter by relation name"
    )

    args = parser.parse_args()
    logging_utils.configure(args=args)

    logger.info(args)
    main(args)
