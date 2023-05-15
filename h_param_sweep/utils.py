import sys

sys.path.append("..")

import argparse
import copy
import json
import os
from typing import List

from src import data, models
from src.benchmarks import faithfulness
from src.functional import make_prompt, predict_next_token
from src.lens import causal_tracing, layer_c_measure
from src.operators import JacobianIclMeanEstimator

import numpy as np
import torch
from tqdm.auto import tqdm


def filter_by_model_knowledge(
    mt: models.ModelAndTokenizer,
    relation_prompt: str,
    relation_samples: List[data.RelationSample],
) -> List[data.RelationSample]:
    model_knows = []
    for sample in relation_samples:
        top_prediction = predict_next_token(
            mt=mt, prompt=relation_prompt.format(sample.subject)
        )[0][0].token
        tick = sample.object.strip().startswith(top_prediction.strip())
        if tick:
            model_knows.append(sample)
    return model_knows


def choose_sample_pairs(
    samples: List[data.RelationSample],
) -> List[data.RelationSample]:
    idx_pair = np.random.choice(range(len(samples)), 2, replace=False)
    sample_pair: list = [list(samples)[i] for i in idx_pair]
    if sample_pair[0].object != sample_pair[1].object:
        return sample_pair  # if the objects are different, return
    return choose_sample_pairs(samples)  # otherwise, draw again


def select_subset_from_relation(relation: data.Relation, n: int) -> data.Relation:
    indices = np.random.choice(
        range(len(relation.samples)), min(len(relation.samples), n), replace=False
    )
    samples = [relation.samples[i] for i in indices]
    subset_relation = copy.deepcopy(relation.__dict__)
    subset_relation["samples"] = samples
    return data.Relation(**subset_relation)
