import time
import os
path = os.path.abspath(os.getcwd())

import json
import argparse
from openai import OpenAI
import datasets
import transformers
import torch

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

    parser.add_argument(
        "--sentences", type=str, default="default", help="'one' or 'four'"
    )

    # Filename 
    parser.add_argument(
        "--name_id", type=str, default="default", help="name for saving"
    )

    # Temperature / top p
    parser.add_argument(
        "--temperature", type=float, default=1.2, help="temperature for LLM calls"
    )

    parser.add_argument("--top_p", type=float, default=1, help="top p for LLM calls")

    parser.add_argument(
        "--random_seed", type=int, default=13, help="Set our random seed for inference"
    )

    parser.add_argument("--num_tries", type=int, default=1000, help="How many examples do we collect")

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

    if params.sentences == 'one':
        max_tokens = 50
    else:
        max_tokens = 200

    response = client.chat.completions.create(
        temperature=params.temperature,
        top_p=params.top_p,
        model=model,
        messages=obs["messages"],
        logprobs=False,
        max_tokens=max_tokens
    )

    full_response = response.choices[0].message.content

    return full_response

def get_pred(params):

    output_dict = {}

    for i in range(params.num_tries):

        obs = create_data(params)

        if params.model_type == "gpt4mini":
            response = _gpt_call(obs, params)

        print("response:", response)

        output_dict[i] = {
            "premise": response
        }


    return output_dict

def create_data(params):

    all_data = []

    user_prompt = create_user_prompt_premise(params)
    print("\n\nPrompt:", user_prompt)
    messages = create_premise_data(user_prompt)

    return {"messages": messages}


def create_user_prompt_premise(params):

    if params.sentences == 'one':
        context = 'Provide a single sentence '
    elif params.sentences == 'four':
        context = 'Provide an output with four sentences '
    all_prompts = [
            'about a workplace',
            'about the founding fathers',
            'from a review of a book',
            'overheard from radio podcast',
            'that you might say to a tourist when recommending a specific location',
            'describing your favourite place in the city',
            'that you would never want to hear',
            'that you might overhear on public transport',
            'about someone you admire',
            'about someone historically significant',
            'from a boring website',
            'to explain a complicated concept',
            'that you only hear in films',
            'from a not so famous speech',
            'that you might read in a blog about geography',
            'that might surprise me',
            'you remember from when someone read you the riot act',
            'from some song lyrics',
            'extract from a made-up newspaper',
            'describing a musician',
            'that you might hear in a battle',
            'from a propaganda leaflet',
            'with a fact that surprised you',
            'from a Shakespeare play',
            'containing a joke',
            'extract from Poirot',
            'explaining investment banking to a graduate student',
            'describing the history of mathematics',
            'about railways',
            'that you might read in a formal legal contract',
            'describing a medical procedure',
            'that you could hear in an airplane cabin',
            'giving orders on a submarine',
            'with technical analysis',
            'describing the plants grown in a garden',
            'explaining an animal found in the rainforest',
            'found in the instructions for a electrical item',
            'from a poem',
            'from a detailed a technical report',
            'in slang',
            "describing a tactical analysis of the team's strategy",
            'overheard between two people in a pub',
            'from a review of an NLP paper submitted for peer review',
            'with a surprising historical fact',
            'that you could have overheard in the Great Exhibition',
            'about farming',
            'that you could hear in an undergraduate lecture',
            'describing the properties of an element in the periodic table',
            'that might be interesting to an economist',
            'describing a scientific advancement',
            'to describe an object in the British Museum']


    prompt_to_user = context + random.choice(all_prompts)

    if params.sentences == 'one':
        prompt_to_user += ". Don't give a general response first, immediately start the sentence without any quotation marks."
    elif params.sentences == 'four':
        prompt_to_user += ". Don't give a general response first, immediately start the four sentences. Write the four sentences back to back without any quotation marks."

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

    assert params.name_id != "default"
    assert params.model_type in ["gpt4mini"]
    
    if params.model_type == 'gpt4mini':
        params.model_id = "gpt-4o-mini-2024-07-18"

    # Note, it is possible for refusals to be included in the generated premises
    output_dict = get_pred(params)
        
    output_file = "premises_" + str(params.random_seed) + "_" + params.name_id + "_" + params.sentences

    output_file += '.json'

    with open(output_file, "w") as f:
        json.dump(output_dict, f)
