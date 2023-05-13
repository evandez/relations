from typing import Callable, List, Literal, Tuple

from src import models
from src.models import ModelAndTokenizer
from src.operators import _compute_h_index
from src.utils.supplimentary import untuple

import baukit
import torch


######################### utils #########################
def interpret_logits(
    mt: ModelAndTokenizer,
    logits: torch.Tensor,
    top_k: int = 10,
    get_proba: bool = False,
) -> List[Tuple[str, float]]:
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=top_k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=top_k).values.squeeze().tolist()
    return [
        (mt.tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)
    ]


def logit_lens(
    mt: ModelAndTokenizer,
    h: torch.Tensor,
    interested_tokens: List[int] = [],
    get_proba: bool = False,
) -> Tuple[List[Tuple[str, float]], dict]:
    logits = mt.lm_head(h)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    candidates = interpret_logits(mt, logits)
    interested_logits = {
        t: (logits[t].item(), mt.tokenizer.decode(t)) for t in interested_tokens
    }
    return candidates, interested_logits


######################### layerwise contribution/completeness #########################
def layer_c_measure(
    mt: ModelAndTokenizer,
    relation_prompt: str,
    subject: str,
    verbose: bool = False,
    measure: Literal["completeness", "contribution"] = "completeness",
) -> dict:
    tokenized = relation_prompt.format(subject)
    with baukit.TraceDict(mt.model, layers=models.determine_layer_paths(mt)) as traces:
        output = mt.model(
            **mt.tokenizer(tokenized, return_tensors="pt", padding=True).to(
                mt.model.device
            )
        )

    object_id = output.logits[0][-1].argmax().item()
    object = mt.tokenizer.decode(object_id)
    base_score = torch.nn.functional.softmax(output.logits[0][-1], dim=-1)[
        object_id
    ].item()

    if verbose:
        print(f"object ==> {object} [{object_id}], base = {base_score}")

    layer_contributions = {}

    prev_score = 0
    for layer in models.determine_layer_paths(mt):
        h = untuple(traces[layer].output)[0][-1]
        candidates, interested_logits = logit_lens(mt, h, [object_id], get_proba=True)
        layer_score = interested_logits[object_id][0]
        sub_score = base_score if measure == "completeness" else prev_score
        cur_layer_contribution = (layer_score - sub_score) / base_score

        layer_contributions[layer] = cur_layer_contribution

        if verbose:
            print(f"layer: {layer}, diff: {cur_layer_contribution}")

        prev_score = layer_score

    return layer_contributions


######################### causal tracing #########################
def get_replace_intervention(
    intervention_layer: str, intervention_tok_idx: int, h_intervention: torch.Tensor
) -> Callable:
    def intervention(
        output: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor, layer: str
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if layer != intervention_layer:
            return output
        output[0][0][intervention_tok_idx] = h_intervention
        return output

    return intervention


def causal_tracing(
    mt: ModelAndTokenizer,
    prompt_template: str,
    subject_original: str,
    subject_corruption: str,
    verbose: bool = False,
) -> dict:
    h_idx_orig, tokenized_orig = _compute_h_index(
        mt=mt,
        prompt=prompt_template.format(subject_original),
        subject=subject_original,
        offset=-1,
    )

    h_idx_corr, tokenized_corr = _compute_h_index(
        mt=mt,
        prompt=prompt_template.format(subject_corruption),
        subject=subject_corruption,
        offset=-1,
    )

    layer_names = models.determine_layer_paths(mt)
    with baukit.TraceDict(mt.model, layer_names) as traces_o:
        output_o = mt.model(**tokenized_orig)

    answer, p_answer = interpret_logits(mt, output_o.logits[0][-1], get_proba=True)[0]
    answer_t = (
        mt.tokenizer(answer, return_tensors="pt").to(mt.model.device).input_ids[0]
    )

    if verbose:
        print(f"answer: {answer}[{answer_t.item()}], p(answer): {p_answer:.3f}")

    result = {}
    for intervention_layer in layer_names:
        with baukit.TraceDict(
            mt.model,
            layers=layer_names,
            edit_output=get_replace_intervention(
                intervention_layer=intervention_layer,
                intervention_tok_idx=h_idx_corr,
                h_intervention=untuple(traces_o[intervention_layer].output)[0][
                    h_idx_orig
                ],
            ),
        ) as traces_i:
            output_i = mt.model(
                **mt.tokenizer(
                    prompt_template.format(subject_corruption), return_tensors="pt"
                ).to(mt.model.device)
            )

        z = untuple(traces_i[layer_names[-1]].output)[0][-1]
        candidates, interested = logit_lens(mt, z, [answer_t], get_proba=True)
        layer_p = interested[answer_t][0]

        if verbose:
            print(intervention_layer, layer_p)
        result[intervention_layer] = (layer_p - p_answer) / p_answer

    return result