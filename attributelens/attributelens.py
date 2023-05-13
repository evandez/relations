from collections import defaultdict
from typing import Any

import attributelens.utils as lens_utils
from src.functional import compute_hidden_states
from src.models import ModelAndTokenizer, determine_layer_paths
from src.operators import LinearRelationOperator
from src.utils import tokenizer_utils

import numpy as np
import torch
from baukit import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer


class Attribute_Lens:
    def __init__(
        self,
        mt: ModelAndTokenizer,
        top_k: int = 10,
        layer_output_tmp: str = "h.{}",
    ):
        self.mt = mt
        self.top_k = top_k
        self.layers = determine_layer_paths(mt)
        self.layer_output_tmp = layer_output_tmp

    def apply_attribute_lens(
        self,
        subject: str,
        relation_operator: LinearRelationOperator,
    ) -> dict[str, Any]:
        print("subject  : ", subject)
        print("prompt_template : ", relation_operator.prompt_template)

        prompt = relation_operator.prompt_template.format(subject)
        inputs = self.mt.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True
        ).to(self.mt.model.device)

        offset_mapping = inputs.pop("offset_mapping")
        subject_start, subject_end = tokenizer_utils.find_token_range(
            prompt, subject, offset_mapping=offset_mapping[0]
        )

        prompt_tokenized = [self.mt.tokenizer.decode(t) for t in inputs.input_ids[0]]

        print(
            "subject mapping: ",
            subject_start,
            subject_end,
            " >> ",
            prompt_tokenized[subject_start:subject_end],
        )

        v_space_reprs: list = []

        for sub_idx in range(subject_start, subject_end):
            v_space_reprs.append(defaultdict(list))
            for layer_idx in range(len(self.layers)):
                # print(">>> ", sub_idx, layer_idx)

                [[hs], _] = compute_hidden_states(
                    mt=self.mt, layers=[layer_idx], inputs=inputs
                )

                predictions = relation_operator(
                    subject=subject,
                    k=self.top_k,
                    h=hs[:, sub_idx],
                ).predictions

                v_space_reprs[sub_idx - subject_start][
                    self.layer_output_tmp.format(layer_idx)
                ] = [(p.token, p.prob) for p in predictions]

        ret_dict = dict()
        ret_dict["prompt_tokenized"] = prompt_tokenized
        ret_dict["prompt_template"] = relation_operator.prompt_template
        ret_dict["v_space_reprs"] = v_space_reprs
        ret_dict["subject_range"] = (subject_start, subject_end)

        return ret_dict
