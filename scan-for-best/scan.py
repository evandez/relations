import torch
import numpy as np
import json
from tqdm import tqdm

import os
import sys
import random
sys.path.append('..')

from relations import estimate
from util import model_utils
from dsets.counterfact import CounterFactDataset


###########################################################################
MODEL_NAME = "EleutherAI/gpt-j-6B"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
layer = 25
consider_residual = False
approximate_rank = -1
num_test_cases = 50
###########################################################################

print("loading ", MODEL_NAME)
mt = model_utils.ModelAndTokenizer(MODEL_NAME, low_cpu_mem_usage=False)

model = mt.model
tokenizer = mt.tokenizer
tokenizer.pad_token = tokenizer.eos_token

counterfact = CounterFactDataset("../data/")
print("loaded counterfact dataset")

###########################################################################
relation_dct = {
    'P17'   : {'relation': '{} is located in the country of', 'correct_predict': None, 'cached_JB': None},
    'P641'  : {'relation': '{} plays the sport of', 'correct_predict': None, 'cached_JB': None},
    'P103'  : {'relation': 'The mother tongue of {} is', 'correct_predict': None, 'cached_JB': None},
    'P176'  : {'relation': '{} is produced by', 'correct_predict': None, 'cached_JB': None},
}
###########################################################################

for relation_id in relation_dct:
    save_dir = f"gpt-j/{relation_id}/"

    relation = relation_dct[relation_id]["relation"]
    os.makedirs(save_dir, exist_ok=True)

    print("\n\n")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(relation_id, " <> ", relation)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    cf_relation = [c['requested_rewrite'] for c in counterfact if c["requested_rewrite"]["relation_id"] == relation_id]
    print(relation_id, "filtered >>> ", len(cf_relation))

    print("Checking correct prediction with normal calculation ...")
    if(relation_dct[relation_id]['correct_predict'] == None):
        correct_predict = []
        for c in tqdm(cf_relation):
            # print(c['subject'], " target: ", c['target_true']['str'], end = ", ")
            txt, ret_dict = model_utils.generate_fast(
                model, tokenizer,
                [relation.format(c['subject'])],
                argmax_greedy = True,
                max_out_len= 20,
                get_answer_tokens=True,
            )
            answer = ret_dict['answer'][0]['top_token']
            # print("predict: ", answer, end = " >>>> ")

            ok = c['target_true']['str'].startswith(answer.strip())
            # print(ok)

            if(ok):
                correct_predict.append(c)

            with open(f"{save_dir}correct_prediction_{relation_id}.json", "w") as f:
                json.dump(correct_predict, f)
    else:
        print("Skipped checking, loading from file ==> ", relation_dct[relation_id]['correct_predict'])
        with open(relation_dct[relation_id]['correct_predict']) as f:
            correct_predict = json.load(f)


    print(f"Correctly predicted {len(correct_predict)} targets")
    test_cases = [(c["subject"], -1, c["target_true"]['str']) for c in random.sample(correct_predict, min(num_test_cases, len(correct_predict)))]

    print()
    print("Calculating Jacobian weights and biases at all subject tokens")
    if(relation_dct[relation_id]['cached_JB'] == None):
        calculated_relation_collections = []
        for c in tqdm(correct_predict):
            relation_collection = estimate.estimate_relation_operator__for_all_subject_tokens(
                model, tokenizer,
                c["subject"],
                relation,
                layer=layer,
                device=model.device,
                consider_residual = True,
                approximate_rank = approximate_rank
            )

            calculated_relation_collections.append({
                "subject": c["subject"],
                "request": c,
                "all_weights_and_biases": [
                    {
                        "weight": calc.weight.cpu().numpy(),
                        "bias": calc.bias.cpu().numpy(),
                        "misc": calc.misc
                    } for calc in relation_collection
                ]
            })
        
        print("Calculating completed .. saving results")
        np.savez(
            f"{save_dir}jacobian_calculations__all_sub_toks__{relation_id}__layer_{layer}.npz", 
            jacobians = calculated_relation_collections, 
            allow_pickle = True
        )
    else:
        print("Skipping calculation for Jacobians and Biases, loading from => ", relation_dct[relation_id]['cached_JB'])
        calculated_relation_collections = np.load(
            relation_dct[relation_id]['cached_JB'],
            allow_pickle= True
        )["jacobians"]

    print("\n")

    with open(f"{save_dir}scan_results_all_sub.txt", "w") as f:
        for cur_operators in tqdm(calculated_relation_collections):
            print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", file=f)
            print(f'(s = {cur_operators["request"]["subject"]}, r = {relation} [{relation_id}], o = {cur_operators["request"]["target_true"]["str"]})', file=f)
            print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", file=f)
            for calc in cur_operators["all_weights_and_biases"]:
                print("----------------------------------------------------------------------------------------------------", file=f)
                print(calc["misc"], file=f)
                print("----------------------------------------------------------------------------------------------------", file=f)

                is_located_in = estimate.RelationOperator(
                    model = model, tokenizer= tokenizer,
                    layer = layer, relation = relation,
                    weight= torch.tensor(calc['weight'], device = model.device), 
                    bias= torch.tensor(calc['bias'], device=model.device)
                )

                tick = 0
                for subject, subject_token_index, target in test_cases:
                    objects = is_located_in(
                        subject,
                        subject_token_index=subject_token_index,
                        device=model.device,
                        return_top_k=5,
                    )
                    print(f"{subject}, target: {target}   ==>   predicted: {objects}", file=f)
                    ok = False
                    for o in objects:
                        if target.startswith(o.strip()):
                            ok = True
                            break
                    tick += ok
                print("----------------------------------------------------------------------------------------------------", file=f)
                print(f"{tick}/{len(test_cases)}", file=f)
                print("----------------------------------------------------------------------------------------------------\n", file=f)

            print("\n", file=f)
