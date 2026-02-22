import time
import os
path = os.path.abspath(os.getcwd())

import json
import argparse
from openai import OpenAI
import datasets
import transformers
import torch
from datasets import Dataset

from utils import load_hf_dataset
import random


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

    parser.add_argument(
        "--random_seed", type=int, default=13, help="Set our random seed for inference"
    )

    parser.add_argument("--premise_file", type=str, default="default",
            help="File name to load with premises")

    parser.add_argument("--name_id", type=str, default="default",
                    help="For saving the dataset")

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

    response = client.chat.completions.create(
        temperature=params.temperature,
        top_p=params.top_p,
        model=model,
        messages=obs["messages"],
        logprobs=False,
    )

    full_response = response.choices[0].message.content

    return full_response


def get_pred(params, premise, lab):

    assert lab in ['entailment', 'neutral', 'contradiction']

    output_dict = {}

    obs = create_data(params, premise, lab)

    print("\n\n\n\n** obs:", obs)

    if params.model_type == "gpt4mini":

        response = _gpt_call(obs, params)

        print("\n\n\npremise:", premise)
        print("hypothesis:", response)
        print("lab:", lab)

    return response

def create_data(params, premise, lab):

    all_data = []

    user_prompt = create_user_prompt_premise(params, premise, lab)
    messages = create_premise_data(user_prompt)
    
    return {"messages": messages}


def create_user_prompt_premise(params, premise, lab):

    prompt_to_user = "Premise: " + premise

    if lab == 'contradiction':
        if random.sample([True, False], 1)[0]:
            prompt_to_user += """From the premise above, randomly pick two sentences, label them sentence 1 and sentence 2. Then construct a sentence that contradicts sentence 1, does not contradict sentence 2, but mentions something related to sentence 2. Label this third sentence ‘output:’"""
        else:
            prompt_to_user += """From the premise above, randomly pick two sentences, label them sentence 1 and sentence 2. Then construct a two sentence output (label it 'output:') which contains one sentence implied by sentences 1 and 2, and another sentence that contradicts one of them."""

    elif lab == 'neutral':
        if random.sample([True, False], 1)[0]:
            prompt_to_user += """From the premise above, randomly pick two sentences, label them sentence 1 and sentence 2. Then construct a sentence that relates to both sentences 1 and 2, but is not implied by them. Label this third sentence ‘output:’"""
        else:
            prompt_to_user += """From the premise above, randomly pick two sentences, label them sentence 1 and sentence 2. Then construct a two sentence output (label it 'output:') which contains one sentence implied by sentences 1 and 2, and another sentence that is not implied by them."""

    elif lab == 'entailment':
        if random.sample([True, False], 1)[0]:
            prompt_to_user += """From the premise above, randomly pick two sentences, label them sentence 1 and sentence 2. Then construct a sentence that is implied by both sentences 1 and 2, relating to both of these. Label this third sentence ‘output:’"""
        else:
            prompt_to_user += """Provide a very short, concise fact that is strictly implied by a specific part of the premise above (label it 'output:'). Don't do something obvious."""

    return prompt_to_user

def create_premise_data(user_prompt):

    messages = [{
            "role": "user",
            "content": user_prompt}]

    return messages

if __name__ == "__main__":

    params = get_args()

    # Setting boolean params

    params.model_id = "invalid"

    print(params)

    # Checking the validity of the chosen params
    assert params.temperature >= 1
    assert params.top_p == 1

    assert params.model_type in ["gpt4mini"]
 
    set_seeds(params.random_seed)

    with open(params.premise_file + ".json") as f:
        all_premises = json.load(f)

    output_dict = {}

    if params.model_type == 'gpt4mini':
        params.model_id = "gpt-4o-mini-2024-07-18"

    counter = 0
    premise_list = []
    
    # We avoid hypotheses being generated for similar premises (with the same 3 words at the beginning)
    # In practice, we run this data generation script 5 times with different seeds (to speed up the data generation process)
    # So the premises with the same first 3 words will be repeated at max 5 times in the generated synthetic data
    prem_include_int = 3

    for idx, prem_dict in all_premises.items():
        
        premise = prem_dict['premise']

        if premise[0] == '"' and premise[-1] == '"':
            premise = premise[1:-1]

        if "*" not in premise and premise.strip()[-1] == "." and '"' not in premise:
            
            output_dict[counter] = {
                        "premise": premise,
                        "hypothesis": {}}

            # Avoid generating hypotheses with premises starting in the same way
            end_of_prem_start = min(len(premise.split(" ")), prem_include_int)
            prem_start = premise.split(" ")[:end_of_prem_start]

            if prem_start in premise_list:
                print("Already generated hypothesis for a similar premise")

            else:
                premise_list.append(prem_start)

                for lab in ['contradiction', 'neutral', 'entailment']:
                    
                    hypothesis = get_pred(params, premise, lab)
                    output_dict[counter]["hypothesis"][lab] = hypothesis
                    print("Valid hypothesis:", hypothesis)

                counter += 1

        else:
            print("Invalid premise:", premise)

    output_file = "hypcomplex_" + params.name_id

    premises = []
    hypotheses = []
    labels = []

    for idx in output_dict:

        if output_dict[idx]['hypothesis'] != {}:

            premises.append(output_dict[idx]['premise'])

            hypotheses.append(output_dict[idx]['hypothesis']['entailment'])
            labels.append(0)

            premises.append(output_dict[idx]['premise'])
            hypotheses.append(output_dict[idx]['hypothesis']['contradiction'])
            labels.append(2)

            premises.append(output_dict[idx]['premise'])
            hypotheses.append(output_dict[idx]['hypothesis']['neutral'])
            labels.append(1)


    my_dict = {'premise': premises, 'hypothesis': hypotheses, 'label': labels}

    dataset = Dataset.from_dict(my_dict)

    dataset.save_to_disk("unlabelled_data/" + output_file)
