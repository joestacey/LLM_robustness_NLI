
import time
import os
path = os.path.abspath(os.getcwd())

import json
import argparse
from openai import OpenAI
import datasets
import transformers
import torch

from create_train_data import format_data
from utils import load_hf_dataset
import random

import numpy as np


def get_args():
    """
    Parse command line arguments
    :return params:
    """
    parser = argparse.ArgumentParser(description="Training model parameters")

    parser.add_argument(
        "--num_samples_firstsplit",
        type=int,
        default=10000,
        help="size of firstsplit",
    )

    parser.add_argument(
        "--num_samples_nextsplit",
        type=int,
        default=10000,
        help="size of nextsplit",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="default",
        help="openai model id if using fine-tuned model",
    )

    parser.add_argument(
        "--model_type", type=str, default="default", help="gpt4mini"
    )

    # Filename 
    parser.add_argument(
        "--name_id", type=str, default="default", help="text for saving results"
    )

    # Temperature / top p
    parser.add_argument(
        "--temperature", type=float, default=0.00001, help="temperature for LLM calls"
    )

    parser.add_argument("--top_p", type=float, default=0.00001, help="top p for LLM calls")

    parser.add_argument(
        "--random_seed", type=int, default=13, help="Set our random seed for inference"
    )

    parser.add_argument("--num_runs", type=int, default=1,
                    help="How many inference steps to do")

    parser.add_argument(
        "--make_predictions_on_training_data", type=int, default=0, help="Are we making predictions on training data (used by sampling strategies)?"
    )

    parser.add_argument(
        "--training_data_to_make_preds_on", type=str, default="default", help="name of our training data (e.g. snli)")

    parser.add_argument(
        "--aug_data", type=str, default="default", help="Unlabelled dataset to make predictions on")

    parser.add_argument("--unlabelled", type=int, default=0, help="Do we make predictions on unlabelled data")

    parser.add_argument("--filter_type", type=str, default="default", help="'nextsplit' if making predictions on the additional potential 10k training examples")

    parser.add_argument("--gpt_soft_probs", type=int, default=0,
                        help="Should GPT return soft probs")

    params, _ = parser.parse_known_args()

    return params

def namespace_to_dict(namespace):

    return {k: v for k, v in vars(namespace).items()}


def set_seeds(seed_value: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed_value: chosen random seed
    """
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    transformers.set_seed(seed_value)

def _gpt_call(obs, params):

    client = OpenAI()

    model = params.model_id

    if not params.gpt_soft_probs:
        
        response = client.chat.completions.create(
            temperature=params.temperature,
            top_p=params.top_p,
            model=model,
            messages=obs["messages"],
            logprobs=False,
        )

        soft_probs = {}

    else:

        response = client.chat.completions.create(
            temperature=params.temperature,
            top_p=params.top_p,
            model=model,
            messages=obs["messages"],
            logprobs=True,
            top_logprobs=10,
        )

        soft_probs = {'contr': 0, 'neutral': 0, 'ent': 0}

        for idx, log_prob in enumerate(response.choices[0].logprobs):
            assert isinstance(log_prob[0], str)
            if log_prob[1]:
                assert idx < 1
                # We consider log_prob[1][0] with 0 index as we want the first token
                for single_logprob in log_prob[1][0].top_logprobs:
                    if single_logprob.token in soft_probs:
                        soft_probs[single_logprob.token] = np.exp(single_logprob.logprob)

            total = soft_probs['contr'] + soft_probs['neutral'] + soft_probs['ent']
            soft_probs['contr'] = soft_probs['contr'] / total
            soft_probs['neutral'] = soft_probs['neutral'] / total
            soft_probs['ent'] = soft_probs['ent'] / total

    full_response = response.choices[0].message.content

    return full_response, soft_probs


def get_pred(dataset, params, run_num):

    set_seeds(params.random_seed + run_num)

    output_dict = {}

    for i, obs in enumerate(dataset):

        if params.model_type == "gpt4mini":
            no_tries = 0

            response_valid = False

            max_no_tries = 1

            while not response_valid and no_tries < max_no_tries:
                response, soft_probs = _gpt_call(obs, params)

                if response.lower() in ['neutral', 'contradiction', 'entailment']: 
                    response_valid = True
                else:
                    response_valid = False

                no_tries += 1

        if params.model_type == 'gpt4mini' and params.gpt_soft_probs:

            output_dict[i] = {
                "label": obs["label"],
                "full_response": response,
                "response_valid": response_valid,
                "soft_probs": soft_probs
            }

        else:

            output_dict[i] = {
                "label": obs["label"],
                "full_response": response,
                "response_valid": response_valid,
            }

    return output_dict


if __name__ == "__main__":

    params = get_args()

    # Setting boolean params
    params.make_predictions_on_training_data = bool(params.make_predictions_on_training_data)
    params.unlabelled = bool(params.unlabelled)
    params.gpt_soft_probs = bool(params.gpt_soft_probs)

    if params.make_predictions_on_training_data:
        assert params.filter_type in ['nextsplit']

    if params.filter_type in ['nextsplit']:
        assert params.make_predictions_on_training_data

    print(params)

    assert params.name_id != "default"
    assert params.temperature <= 2 and params.temperature >= 0
    assert params.top_p <= 1 and params.top_p >= 0
    assert params.model_type in ["gpt4mini"]
    
    all_data_tuples = [
            ('snli', 'test'),
            ('snli', 'validation'),
            ('multi_nli', 'validation_matched'),
            ('multi_nli', 'validation_mismatched'),
            ('anli', 'test_r1'),
            ('anli', 'test_r2'),
            ('anli', 'test_r3'),
            ('allenai/scitail', 'test'),
            ('pietrolesci/nli_fever', 'dev'),
            ('copa_nli', 'test'),
            ('inli', 'nli'),
            ('inli', 'implied'),
            ('alisawuffles/WANLI', 'test')
            ]

    if params.training_data_to_make_preds_on != 'default':
        assert params.make_predictions_on_training_data

    if params.make_predictions_on_training_data:
        assert params.training_data_to_make_preds_on != 'default'
        assert not params.unlabelled
        all_data_tuples = [(params.training_data_to_make_preds_on, 'train')]

    elif params.unlabelled:
        assert params.aug_data != 'default'
        all_data_tuples = [('unlabelled', 'train')]


    all_output_dict = {}

    for data_tuple in all_data_tuples:

        for run in range(params.num_runs):

            run += 1
            data_desc, data_split = data_tuple

            loaded_dataset = load_hf_dataset(data_desc, data_split, aug_data_location=params.aug_data, shuffle_dataset=True)
            print("dataset loaded from huggingface")

            if params.make_predictions_on_training_data: # We make predictions on the next 10k potential training examples 

                loaded_dataset = loaded_dataset.filter(lambda example: example['label'] in [0, 1, 2])

                print("Size before filtering:", len(loaded_dataset))
                loaded_dataset = loaded_dataset.filter(
                    lambda example, index: index < params.num_samples_nextsplit + params.num_samples_firstsplit and index >= params.num_samples_firstsplit, with_indices=True
                )
                print("Size after filtering:", len(loaded_dataset))

            loaded_dataset =  format_data(loaded_dataset, params, inference=True)
            loaded_dataset = datasets.Dataset.from_list(loaded_dataset)
            output_dict = get_pred(loaded_dataset, params, run)

            exp_name = "run_" + str(run) + "_" + data_desc + "_" + data_split 
            exp_name = exp_name.replace("/","_")

            # We save the predictions already made
            if params.make_predictions_on_training_data:
                output_file = "saved_baseline_preds/" + params.name_id
            elif params.unlabelled:
                output_file = "unlabelled_data_predictions/" + params.name_id
            else:
                output_file = "inference_outputs/" + params.name_id

            if not os.path.exists(output_file): 
                os.makedirs(output_file)
    
            with open(output_file + '/' + exp_name + '.json', "w") as f:
                json.dump(output_dict, f)
