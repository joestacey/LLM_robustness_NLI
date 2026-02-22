
import os
path = os.path.abspath(os.getcwd())

import json
import argparse
from openai import OpenAI
import datasets
import transformers
import torch

import time
from utils import load_hf_dataset
import random

from huggingface_hub import login

def get_args():
    """
    Parse command line arguments
    :return params:
    """
    parser = argparse.ArgumentParser(description="Training model parameters")

    # Data locations

    parser.add_argument(
        "--model_type", type=str, default="default", help="'gpt4mini'"
    )

    # Temperature / top p

    parser.add_argument(
        "--temperature", type=float, default=1, help="temperature for LLM calls"
    )

    parser.add_argument("--top_p", type=float, default=1, help="top p for LLM calls")

    # Do we save or not

    parser.add_argument(
        "--random_seed", type=int, default=13, help="Set our random seed for inference"
    )

    parser.add_argument("--num_runs", type=int, default=8,
                    help="How many inference steps to do")

    parser.add_argument("--dataset_location", type=str, default="default",
                        help="Default location")

    params, _ = parser.parse_known_args()

    return params

def namespace_to_dict(namespace):

    return {k: v for k, v in vars(namespace).items()}

def get_pred_from_response(pred):

    # Some code to correct the few-shot predictions
    error = False
    if pred.count("Answer:") > 1:
        error = True

    reduced_pred = None
    if "Answer: Entailment" in pred:
        reduced_pred = "entailment"
    elif "Answer: Neutral" in pred:
        reduced_pred = "neutral"
    elif "Answer: Contradiction" in pred:
        reduced_pred = "contradiction"
    else:
        error = True

    return reduced_pred, error

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

    response = client.chat.completions.create(
        temperature=params.temperature,
        top_p=params.top_p,
        model=model,
        messages=obs["messages"],
        logprobs=False,
    )

    full_response = response.choices[0].message.content

    return full_response


def get_pred(dataset, params, run_num):

    set_seeds(params.random_seed + run_num)

    output_dict = {}

    for i, obs in enumerate(dataset):

        no_tries = 0
        response_valid = False

        while not response_valid and no_tries < 3:
            if params.model_type == "gpt4mini":
                response = _gpt_call(obs, params)

            reduced_pred, error = get_pred_from_response(response)

            if not error:
                response_valid = True
            
            no_tries += 1
        
        output_dict[i] = {
            "full_response": response,
            "response_valid": response_valid,
            "reduced_pred": reduced_pred
        }

    return output_dict

def format_data(train_data, params, hypothesis_only=False, inference=False):

    all_data = []

    for i, obs in enumerate(train_data):

        user_prompt = create_user_prompt(obs, params, hypothesis_only=hypothesis_only)
        sys_prompt = None
        messages = create_inf_data(user_prompt, sys_prompt, obs, params, inference=inference)
        all_data.append({"messages": messages})

    return all_data

def create_user_prompt(obs, params, hypothesis_only=False, inference=False):

    prompt_to_user = """You are undertaking the task of Natural Language Inference, a task that involves predicting one of three classes: 1) Entailment (where the hypothesis is entailed by the premise), 2) Contradiction (where the hypothesis contradicts the premise) and 3) Neutral (where there is neither Entailment nor Contradiction). 

You need to provide calculations that show why the chosen class is correct, before concluding what the answer should be. Don't say the correct answer before your conclusion at the end. You must finish with 'Answer: Neutral', 'Answer: Contradiction', or 'Answer: Entailment'.

"""

    prompt_to_user += "Now it's your turn:\n\n"
    prompt_to_user += "\nPremise: " + obs['premise']
    prompt_to_user += '\nHypothesis: ' + obs['hypothesis']

    return prompt_to_user

def create_inf_data(user_prompt, sys_prompt, obs, params, inference=False):

    messages = []

    messages.append({
            "role": "user",
            "content": user_prompt})

    return messages


def filter_improve_data_quality(obs, params):

    accepted = True
    premise = obs['premise']
    hypothesis = obs['hypothesis']

    if 'premise' in premise.lower():
        accepted = False

    if 'hypothesis' in premise.lower():
        accepted = False

    if 'premise' in hypothesis.lower():
        accepted = False

    if 'hypothesis' in hypothesis.lower():
        accepted = False

    if 'entailment' in hypothesis.lower() or \
        'neutral' in hypothesis.lower() or \
        'contradiction' in hypothesis.lower() or \
        'implied' in hypothesis.lower() or \
        'implies' in hypothesis.lower() or \
        'entailment' in premise.lower() or \
        'neutral' in premise.lower() or \
        'contradiction' in premise.lower() or \
        'implied' in premise.lower() or \
        'implies' in premise.lower() or \
        'sure!' in premise.lower() or \
        'can i help' in premise.lower() or \
        'happy to help' in premise.lower() or \
        'no problem' in premise.lower():
            accepted = False

    return accepted

if __name__ == "__main__":

    params = get_args()

    assert params.model_type in ['gpt4mini']

    print(params)

    # Checking the validity of the chosen params
    assert params.temperature >= 1
    assert params.top_p == 1

    assert params.model_type in ["gpt4mini"]
    
    if params.model_type == 'gpt4mini':
        params.model_id = "gpt-4o-mini-2024-07-18"

    # We can make predictions on our unlabelled LLM generated data
    all_data_tuples = [('unlabelled_' + params.dataset_location, 'train')]

    all_output_dict = {}

    for data_tuple in all_data_tuples:

        for run in range(params.num_runs):

            run += 1

            data_desc, data_split = data_tuple

            # Load our data
            loaded_dataset = load_hf_dataset(data_desc, data_split)

            loaded_dataset = loaded_dataset.filter(
                    lambda example: filter_improve_data_quality(example, params))

            loaded_dataset =  format_data(loaded_dataset, params, hypothesis_only=False, inference=True)
            loaded_dataset = datasets.Dataset.from_list(loaded_dataset)
            output_dict = get_pred(loaded_dataset, params, run)
            
            all_output_dict["run_" + str(run) + "_" + data_desc + "_" + data_split] = output_dict


    output_file = 'label_predictions_' + params.dataset_location + '.json'

    labels = {}

    for idx in range(len(loaded_dataset)):
        neut_preds = 0
        cont_preds = 0
        ent_preds = 0

        for run, all_preds_dict in all_output_dict.items():

            pred = all_preds_dict[idx]['reduced_pred']

            if pred is None:
                print("No prediction")
            elif pred.lower() == 'entailment':
                ent_preds += 1
            elif pred.lower() == 'neutral':
                neut_preds += 1
            elif pred.lower() == 'contradiction':
                cont_preds += 1

        # We apply the if in doubt, discard validation
        if ent_preds > 0 and neut_preds == 0 and cont_preds == 0:
            labels[idx] = 0
        elif ent_preds == 0 and neut_preds > 0 and cont_preds == 0:
            labels[idx] = 1
        elif ent_preds == 0 and neut_preds == 0 and cont_preds > 0:
            labels[idx] = 2
        else:
            labels[idx] = -1

    loaded_dataset = load_hf_dataset(data_desc, data_split)
    loaded_dataset = loaded_dataset.filter(
                    lambda example: filter_improve_data_quality(example, params))

    loaded_dataset = loaded_dataset.remove_columns("label")
    loaded_dataset = loaded_dataset.add_column("label", labels.values())

    loaded_dataset.save_to_disk("labelled_generated_data/" + output_file)
