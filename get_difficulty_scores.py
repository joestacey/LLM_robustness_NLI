
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


def get_args():
    """
    Parse command line arguments
    :return params:
    """
    parser = argparse.ArgumentParser(description="Training model parameters")

    # Data locations

    parser.add_argument(
        "--filter", type=int, default=0, help="Do we only evaluate on num_samples"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="description of data for huggingface",
    )


    parser.add_argument(
        "--model_type", type=str, default="default", help="'gpt' or 'llama'"
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

    params, _ = parser.parse_known_args()

    return params


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

        if params.model_type == "gpt4mini":

            response = _gpt_call(obs, params)

        output_dict[i] = {
            "full_response": response,
        }

    return output_dict

def format_data(train_data, params, inference=False):

    all_data = []

    for i, obs in enumerate(train_data):

        user_prompt = create_user_prompt(obs, params)
        messages = create_inf_data(user_prompt)
        all_data.append({"messages": messages})

    return all_data

def create_user_prompt(obs, params, inference=False):

    prompt_to_user = """

Your job is to rate how good Natural Language Inference examples are. Natural Language Inference are pairs of premises and hypotheses where either: the label is 'entailment' and the hypothesis is implied by the premise, the label is 'contradiction' and the hypothesis contradicts the premise, or otherwise the label is 'neutral'.

You must rate the examples from 0 (worse) to 10 (best) on three criteria:

The correctness of the label
The difficulty of the example
The written fluency of the example
The plausibility of the example
Here are two examples:

<Example 1>
Premise: A man holding up a red sign that offers the honest trade of a good poem for a Giant's ticket.
Hypothesis: A man is trying to barter his artistic work for a ticket to a sporting event
Label: Entailment

Correctness: 9
Difficulty: 8
Fluency: 8
Plausibility: 6
</Example 1>

<Example 2>
Premise: Two men in a room looking at a computer screen.
Hypothesis: Two men work on computers.
Label: Entailment

Correctness: 9
Difficulty: 1
Fluency: 7
Plausibility: 10
</Example 2>

Your turn:

"""

    prompt_to_user += "\nPremise: " + obs['premise']
    prompt_to_user += '\nHypothesis: ' + obs['hypothesis']

    if obs['label'] == 0:
        prompt_to_user += '\nLabel: Entailment'
    elif obs['label'] == 1:
        prompt_to_user += '\nLabel: Neutral'
    elif obs['label'] == 2:
        prompt_to_user += '\nLabel: Contradiction'

    prompt_to_user += """

    You must answer in the form:

    Correctness: score
    Difficulty: score
    Fluency: score
    Plausibility: score
    """

    return prompt_to_user

def create_inf_data(user_prompt):

    messages = []

    messages.append({
            "role": "user",
            "content": user_prompt})

    return messages


if __name__ == "__main__":

    params = get_args()

    # Setting boolean params
    params.filter = bool(params.filter)

    # Login to huggingface (unless working offline)
    print(params)

    assert params.name_id != "default"
    assert params.model_type in ["gpt4mini"]
    
    if params.model_type == 'gpt4mini':
        params.model_id = "gpt-4o-mini-2024-07-18"

    all_data_tuples = [('snli', 'train')]

    all_output_dict = {}

    for data_tuple in all_data_tuples:

        for run in range(params.num_runs):

            run += 1

            data_desc, data_split = data_tuple

            loaded_dataset = load_hf_dataset(data_desc, data_split, shuffle_dataset=True)

            print("len of training data:", len(loaded_dataset))

            loaded_dataset = loaded_dataset.filter(lambda example: example['label'] in [0, 1, 2])

            if params.filter:
                print("Size before filtering:", len(loaded_dataset))
                loaded_dataset = loaded_dataset.filter(
                    lambda example, index: index < params.num_samples, with_indices=True
                )
                print("Size after filtering:", len(loaded_dataset))

            for obs in loaded_dataset:
                assert obs['label'] in [0, 1, 2], str(obs['label'])

            loaded_dataset =  format_data(loaded_dataset, params, inference=True)

            loaded_dataset = datasets.Dataset.from_list(loaded_dataset)

            output_dict = get_pred(loaded_dataset, params, run)
            
            all_output_dict["run_" + str(run) + "_" + data_desc + "_" + data_split] = output_dict

    output_file = "training_preds_baseline/" + params.name_id

    output_file += '.json'

    print("Saving to:", output_file)

    with open(output_file, "w") as f:
        json.dump(all_output_dict, f)
