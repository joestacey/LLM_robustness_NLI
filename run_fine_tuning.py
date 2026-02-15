import os

path = os.path.abspath(os.getcwd())

from transformers import TrainingArguments, set_seed, TrainerCallback
import torch
import argparse
import json
from datasets import load_dataset

from openai import OpenAI
from create_train_data_refactored import create_train_data

import time

def get_args():
    """
    Parse command line arguments
    :return params:
    """
    parser = argparse.ArgumentParser(description="Training model parameters")

    # Data locations
    parser.add_argument("--id", type=str, default="default", help="File to fine-tune")

    parser.add_argument(
        "--model_type", type=str, default="default", help="Model to finetune with"
    )

    parser.add_argument("--random_seed", type=int, default=13, help="Random seed")

    params, _ = parser.parse_known_args()

    return params


if __name__ == "__main__":

    params = get_args()
    
    params.force_epochs = int(params.force_epochs)

    # Checking valid choice of params
    assert params.model_type != "default"
    assert params.id != "default"

    print(params)

    # For GPT models, we fine-tune with the OpenAI API
    if params.model_type == "gpt4mini":

        client = OpenAI()
        client.fine_tuning.jobs.create(training_file=params.id, model="gpt-4o-mini-2024-07-18", seed=params.random_seed)
 


