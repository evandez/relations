import argparse
import logging

from scripts.caching.interpolation import save_order_1_approx
from src import data, functional, models, operators
from src.utils import experiment_utils, logging_utils, typing
from src.utils.sweep_utils import read_sweep_results, relation_from_dict

logger = logging.getLogger(__name__)


def main(
    relation_name: str,
    h_layer: typing.Layer | None = None,
    n_few_shot: int = 5,
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

    # compute LRE approximations
    estimator = operators.JacobianIclMeanEstimator(mt=mt, h_layer=h_layer, beta=2.25)
    operator = estimator(
        relation.set(samples=train.samples, prompt_templates=[prompt_template])
    )

    assert operator.weight is not None
    assert operator.bias is not None

    # save LRE approximations
    save_order_1_approx(
        approx=operator,
        file_name=relation_name.lower().replace(" ", "_"),
        path=args.save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache LRE approximations for the attribute lens demo"
    )
    models.add_model_args(parser)
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/LRE_cached",
        help="directory to cache values",
    )

    parser.add_argument(
        "--n-icl",
        type=int,
        default=8,
        help="number of few-shot examples to provide",
    )

    parser.add_argument("--rel-name", type=str, help="filter by relation name")
    args = parser.parse_args()
    logging_utils.configure(args=args)
    experiment = experiment_utils.setup_experiment(args)

    logger.info(args)

    if args.rel_name is not None:
        main(
            relation_name=args.rel_name,
            n_few_shot=args.n_icl,
        )
    else:
        dataset = data.load_dataset()
        for relation in dataset.relations:
            logger.info(f"#################### {relation.name} ####################")
            main(
                relation_name=relation.name,
                n_few_shot=args.n_icl,
            )
            logger.info("------------------------------------------------------")
