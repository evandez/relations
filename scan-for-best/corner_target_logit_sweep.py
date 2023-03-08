import torch
import numpy as np
import json
from tqdm.auto import tqdm
import random
import transformers

import os
import sys
sys.path.append('..')

from relations import estimate
from util import model_utils
from baukit import nethook
from operator import itemgetter



cut_off = 50 # minimum number of correct predictions

###########################################################################
relation_dct = {
    'P17'   : {'relation': '{} is located in the country of', 'correct_predict': None, 'cached_JB': None},
    'P641'  : {'relation': '{} plays the sport of', 'correct_predict': None, 'cached_JB': None},
    'P103'  : {'relation': 'The mother tongue of {} is', 'correct_predict': None, 'cached_JB': None},
    'P176'  : {'relation': '{} is produced by', 'correct_predict': None, 'cached_JB': None},
    'P140'  : {'relation': 'The official religion of {} is', 'correct_predict': None, 'cached_JB': None},
    'P1303' : {'relation': '{} plays the instrument', 'correct_predict': None, 'cached_JB': None},
    'P190'  : {'relation': 'What is the twin city of {}? It is', 'correct_predict': None, 'cached_JB': None},
    'P740'  : {'relation': '{} was founded in', 'correct_predict': None, 'cached_JB': None},
    'P178'  : {'relation': '{} was developed by', 'correct_predict': None, 'cached_JB': None},
    'P495'  : {'relation': '{}, that originated in the country of', 'correct_predict': None, 'cached_JB': None},
    'P127'  : {'relation': '{} is owned by', 'correct_predict': None, 'cached_JB': None},
    'P413'  : {'relation': '{} plays in the position of', 'correct_predict': None, 'cached_JB': None},
    'P39'   : {'relation': '{}, who holds the position of', 'correct_predict': None, 'cached_JB': None},
    'P159'  : {'relation': 'The headquarter of {} is located in', 'correct_predict': None, 'cached_JB': None},
    'P20'   : {'relation': '{} died in the city of', 'correct_predict': None, 'cached_JB': None},
    'P136'  : {'relation': 'What does {} play? They play', 'correct_predict': None, 'cached_JB': None},
    'P106'  : {'relation': 'The profession of {} is', 'correct_predict': None, 'cached_JB': None},
    'P30'   : {'relation': '{} is located in the continent of', 'correct_predict': None, 'cached_JB': None},
    'P937'  : {'relation': '{} worked in the city of', 'correct_predict': None, 'cached_JB': None},
    'P449'  : {'relation': '{} was released on', 'correct_predict': None, 'cached_JB': None},
    'P27'   : {'relation': '{} is a citizen of', 'correct_predict': None, 'cached_JB': None},
    'P101'  : {'relation': '{} works in the field of', 'correct_predict': None, 'cached_JB': None},
    'P19'   : {'relation': '{} was born in', 'correct_predict': None, 'cached_JB': None},
    'P37'   : {'relation': 'In {}, an official language is', 'correct_predict': None, 'cached_JB': None},
    'P138'  : {'relation': '{}, named after', 'correct_predict': None, 'cached_JB': None},
    'P131'  : {'relation': '{} is located in', 'correct_predict': None, 'cached_JB': None},
    'P407'  : {'relation': '{} was written in', 'correct_predict': None, 'cached_JB': None},
    'P108'  : {'relation': '{}, who is employed by', 'correct_predict': None, 'cached_JB': None},
    'P36'   : {'relation': 'The capital of {} is', 'correct_predict': None, 'cached_JB': None},
}
###########################################################################

jacobian_cache_path = "/mnt/39a89eb4-27b7-4fce-a6ab-a9d203443a7c/relation_cached/gpt-j/jacobians_averaged_collection/before__ln_f/{}.npz"
root_path = "gpt-j"
save_results_path = "corner_target_logit_sweep/target_as_multiple_of_J_norm__current.json"
cache_path = "corner_target_logit_sweep/average/sweep_loss_without_second_term_averaged.json"
cached_results = None
with open(cache_path) as f:
    cached_results = json.load(f)

pop_track = []
for relation in relation_dct:
    path = f"{root_path}/{relation}"
    with open(f"{path}/correct_prediction_{relation}.json") as f:
        correct_predictions = json.load(f)
    if(len(correct_predictions) < cut_off):
    # if "performance" not in os.listdir(path):
        print("skipped ", relation)
        pop_track.append(relation)
    
for r in pop_track:
    relation_dct.pop(r)


print("loading the model")
MODEL_NAME = "EleutherAI/gpt-j-6B"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
mt = model_utils.ModelAndTokenizer(MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32)

model = mt.model
tokenizer = mt.tokenizer
tokenizer.pad_token = tokenizer.eos_token

print(f"{MODEL_NAME} ==> device: {model.device}, memory: {model.get_memory_footprint()}")


print("loading dataset")
from dsets.counterfact import CounterFactDataset
counterfact = CounterFactDataset("../data/")


from relations import corner, evaluate

jacobians_calculated_after_layer = 15
precision_at = 3

corner_estimator = corner.CornerEstimator(model = model, tokenizer = tokenizer)
# target_logit_range = range(3, 82, 3)
target_multiple_of_jacobian = range(1, 10)
performance_track = {}

for relation_id in tqdm(relation_dct):
    print(f"relation_id >> {relation_id}")
    print("------------------------------------------------------------------------------------------------------")
    objects = [c['requested_rewrite'] for c in counterfact if c["requested_rewrite"]['relation_id'] == relation_id]
    objects = [" "+ o['target_true']['str'] for o in objects]
    objects = list(set(objects))
    print("unique objects: ", len(objects), objects[0:min(5, len(objects))])

    cached_jbs = np.load(
        jacobian_cache_path.format(relation_id),
        allow_pickle= True
    )
    cached_jacobian = torch.stack(
                            [wb['weight'] for wb in cached_jbs['weights_and_biases']]
                    ).mean(dim=0).to(model.dtype).to(model.device)

    performance_track[relation_id] = {"jacobian_norm": cached_jacobian.norm().item()}
    validation_set = None
    if cached_results is not None:
        print("using cached validation set")
        validation_set = cached_results[relation_id]['validation_set']

    # for target_logit in tqdm(target_logit_range):
    for scale in tqdm(target_multiple_of_jacobian):
        target_logit = cached_jacobian.norm().item() * scale
        
        current_corner = corner_estimator.estimate_corner_with_gradient_descent(
            objects, target_logit_value=target_logit
        )
        vocab_repr = corner_estimator.get_vocab_representation(current_corner, get_logits=True)
        print(f"{target_logit} >> {vocab_repr}")

        performance_track[relation_id][target_logit] = {
            'jacobian': -1, 'identity': -1,
            'vocab_repr': vocab_repr,
            'corner_norm': current_corner.norm().item()
        }

        relation_identity = estimate.RelationOperator(
            model = model,
            tokenizer = tokenizer,
            relation = relation_dct[relation_id]['relation'],
            layer = jacobians_calculated_after_layer,
            weight = torch.eye(model.config.n_embd).to(model.dtype).to(model.device),
            bias = current_corner
        )
        precision, ret_dict = evaluate.evaluate(
            relation_id= relation_id,
            relation_operator= relation_identity,
            precision_at = precision_at,
            validation_set = validation_set
        )

        if(validation_set is None):
            validation_set = ret_dict["validation_set"]
            performance_track[relation_id]['validation_set'] = validation_set

        print("w = identity >> ", precision)
        ret_dict.pop('validation_set')
        ret_dict.pop('predictions')
        performance_track[relation_id][target_logit]['identity'] = (precision, ret_dict)


        relation_jacobian = estimate.RelationOperator(
            model = model,
            tokenizer = tokenizer,
            relation = relation_dct[relation_id]['relation'],
            layer = jacobians_calculated_after_layer,
            weight = cached_jacobian,
            bias = current_corner
        )
        precision, ret_dict = evaluate.evaluate(
            relation_id= relation_id,
            relation_operator= relation_jacobian,
            precision_at = precision_at,
            validation_set = validation_set
        )
        
        print("w = jacobian >> ", precision)
        ret_dict.pop('validation_set')
        ret_dict.pop('predictions')
        performance_track[relation_id][target_logit]['jacobian'] = (precision, ret_dict)
        print()
        

    print("saving results")
    with open(save_results_path, "w") as f:
        json.dump(performance_track, f)
    print("############################################################################################################")