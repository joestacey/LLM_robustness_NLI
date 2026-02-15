import os
import json
import argparse
import random
import numpy as np
import pickle

from datasets import load_dataset

def get_args():
    """
    Parse command line arguments
    :return params:
    """
    parser = argparse.ArgumentParser(description="Training model parameters")

    # Data locations
    parser.add_argument(
        "--load_id", type=str, default="default", help="Predictions to load"
    )

    params, _ = parser.parse_known_args()

    return params

if __name__ == "__main__":

    params = get_args()

    print("Params:", params)

    
    for json_file in os.listdir("inference_outputs/" + params.load_id):

        with open("inference_outputs/" + params.load_id + '/' + json_file) as f:
            dataset = json.load(f)

            total = 0
            correct = 0

            pred_counts = {'0': 0, '1': 0, '2': 0}
            tp_counts = {'0': 0, '1': 0, '2': 0}
            fp_counts = {'0': 0, '1': 0, '2': 0}
            fn_counts = {'0': 0, '1': 0, '2': 0}
            
            for idx in dataset.keys():

                if 'full_response' in  dataset[idx].keys():
                    if dataset[idx]['full_response'] == 'contradiction':
                        pred = 2
                    elif dataset[idx]['full_response'] == 'entailment':
                        pred = 0
                    elif dataset[idx]['full_response'] == 'neutral':
                        pred = 1

                assert pred in [0, 1, 2]

                if 'scitail' in json_file or 'copa' in json_file:
                    if pred == 2:
                        pred = 1

                total += 1

                label = dataset[idx]['label']

                if pred == label:
                    correct += 1

                assert dataset[idx]['label'] in [0, 1, 2]
                
            print("\n\n\nDataset:", json_file)
            if total > 0:
                print("Total:", total, " || Correct:", correct, " || Percentage:", correct/total)
            else:
                print("No observations found")
