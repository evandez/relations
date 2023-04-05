import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")

from typing import Any, List, Sequence

from src import estimate
from src.utils import model_utils


def filter_correct_predictions_by_model(
    model,
    tokenizer,
    relation_prompt,
    cf_relation,
):
    correct_predict = []
    for c in tqdm(cf_relation):
        # print(c['subject'], " target: ", c['target_true']['str'], end = ", ")
        txt, ret_dict = model_utils.generate_fast(
            model,
            tokenizer,
            [relation_prompt.format(c["subject"])],
            argmax_greedy=True,
            max_new_tokens=5,
            get_answer_tokens=True,
        )
        answer = ret_dict["answer"][0]["top_token"]
        ok = c["target_true"]["str"].startswith(answer.strip())
        if ok:
            correct_predict.append(c)
    return correct_predict


def evaluate(
    relation_id: str,
    relation_operator: estimate.RelationOperator,
    validation_set: List[tuple] = None,
    precision_at: int = 1,
):
    """
    evaluates a relation_operator
    Params:
        relation_id: counterfact/wikidata identifier of the relation
        relation_operator: RelationOperator to be evaluated
        validation_set: list of subject, object tuples [(s1, o1), (s2, o2), ...] with which to evaluate the relation_operator
                        if set to None, will evaluate with all the requests of the counterfact dataset.
        precision_at:
    """
    model = relation_operator.model
    tokenizer = relation_operator.tokenizer
    relation_prompt = relation_operator.relation

    ret_dict = {}
    if validation_set == None:
        with open("../data/counterfact.json") as f:
            counterfact = json.load(f)
        cf_relation = [
            c["requested_rewrite"]
            for c in counterfact
            if c["requested_rewrite"]["relation_id"] == relation_id
        ]
        print(
            f"{relation_id} >> number of requests in counterfact = {len(cf_relation)}"
        )

        print("Checking correct prediction with normal calculation ...")
        correct_predict = filter_correct_predictions_by_model(
            model=model,
            tokenizer=tokenizer,
            relation_prompt=relation_prompt,
            cf_relation=cf_relation,
        )
        print(f"Number of correctly predicted requests = {len(correct_predict)}")
        validation_set = [
            (c["subject"], -1, c["target_true"]["str"]) for c in correct_predict
        ]

    assert len(validation_set) != 0, "length of the validation set can't be zero"

    ret_dict["validation_set"] = validation_set
    ret_dict["out_of"] = len(validation_set)

    tick = 0
    track_predictions = []
    print(f"validating on {len(validation_set)} subject --> object associations")
    for subject, subject_token_index, target in tqdm(validation_set):
        output = relation_operator(
            subject,
            subject_token_index=subject_token_index,
            device=model.device,
            return_top_k=max(precision_at, 5),
        )
        top_predictions = [(p.token, p.prob) for p in output.predictions]
        ok = False
        for o in top_predictions[0:precision_at]:
            _o = o.strip()
            if len(_o) == 0:
                continue
            if target.startswith(_o):
                ok = True
                break
        # print(f"{subject}, target: {target}   ==>   predicted: {top_predictions} >>> {ok}")
        tick += ok
        track_predictions.append(top_predictions)

    ret_dict["predictions"] = track_predictions
    ret_dict["tick"] = tick

    return tick / len(validation_set), ret_dict
