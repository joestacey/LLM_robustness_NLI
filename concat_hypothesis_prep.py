from datasets import load_dataset, load_from_disk, Dataset, ClassLabel
import os
import math
from datasets import Dataset
import random
import argparse
import string

path = os.path.abspath(os.getcwd())

from transformers import set_seed
import torch
import argparse
import json
from datasets import load_dataset

from openai import OpenAI
from utils import load_hf_dataset
from huggingface_hub import login

def get_args():
    '''
    Parse command line arguments
    :return params:
    '''
    parser = argparse.ArgumentParser(
            description="Training model parameters")

    parser.add_argument("--filter", type=int, default=1,
                        help="Do we filter the dataset to numsamples")

    parser.add_argument("--num_samples", type=int, default=10000,
                        help="description of data for huggingface") # Size of the baseline training examples

    parser.add_argument("--num_samples_up", type=int, default=1000000000,
                        help="description of data for huggingface") # We use the full remaining training set when looking for concat hyp

    parser.add_argument("--description", type=str, default="default",
            help="Description for hf dataset")

    parser.add_argument("--split_name", type=str, default="default",
            help="Split for hf dataset")

    parser.add_argument("--name_id", type=str, default="default",
            help="Name to save data")

    params, _ = parser.parse_known_args()

    return params

def get_duplicated_premises(data):

    dict_duplicate_prems = {}

    for obs in data:
        lab = obs['label']
        if lab == 0:
            lab = 'entailment'
        elif lab == 1:
            lab = 'neutral'
        elif lab == 2:
            lab = 'contradiction'

        if lab in ['entailment', 'neutral', 'contradiction']:

            if obs['premise'] not in dict_duplicate_prems:
                dict_duplicate_prems[obs['premise']] = {'entailment': [], 'contradiction': [], 'neutral': [], 'total': 0}
            dict_duplicate_prems[obs['premise']][lab].append(obs['hypothesis'])
            dict_duplicate_prems[obs['premise']]['total'] += 1


    return dict_duplicate_prems

def combine_sentences(sent1, sent2, sent3):

    all_three_sentences = [sent1, sent2, sent3]

    random.shuffle(all_three_sentences)

    sent1, sent2, sent3 = all_three_sentences[0], all_three_sentences[1], all_three_sentences[2]

    sent1 = sent1.strip()
    sent2 = sent2.strip()
    sent3 = sent3.strip()

    valid_example = False

    if sent1[-1].lower() in list(string.ascii_lowercase):
        sent1 = sent1 + '.'
        valid_example = True

    elif sent1[-1] == '.':
        valid_example = True

    if sent2[-1].lower() in list(string.ascii_lowercase):
        sent2 = sent2 + '.'
        valid_example = True

    elif sent2[-1] == '.':
        valid_example = True

    combined_sent = sent1 + ' ' + sent2 + ' ' + sent3

    return combined_sent, valid_example


def assemble_into_contradiction_example(dict_duplicate_prems, prem):

    # First do all possible ent examples
    non_cont_endings = dict_duplicate_prems[prem]['entailment'] + dict_duplicate_prems[prem]['neutral']
    cont_endings = dict_duplicate_prems[prem]['contradiction']

    random.shuffle(non_cont_endings)
    random.shuffle(cont_endings)

    non_cont_ending1 = non_cont_endings[0]
    non_cont_ending2 = non_cont_endings[1]
    cont_ending = cont_endings[0]

    cont_hypothesis, valid_example = combine_sentences(
            cont_ending,
            non_cont_ending1,
            non_cont_ending2)

    return {'premise': prem, 'hypothesis': cont_hypothesis, 'label': 2}, valid_example



def assemble_into_neutral_example(dict_duplicate_prems, prem):

    # First do all possible ent examples
    ent_endings = dict_duplicate_prems[prem]['entailment']
    neut_endings = dict_duplicate_prems[prem]['neutral']

    random.shuffle(ent_endings)
    random.shuffle(neut_endings)

    ent_ending1 = ent_endings[0]
    ent_ending2 = ent_endings[1]
    neut_ending = neut_endings[0]

    neut_hypothesis, valid_example = combine_sentences(
            ent_ending1, 
            ent_ending2,
            neut_ending)

    return {'premise': prem, 'hypothesis': neut_hypothesis, 'label': 1}, valid_example

def assemble_into_entailment_example(dict_duplicate_prems, prem):
    
    # First do all possible ent examples
    ent_endings = dict_duplicate_prems[prem]['entailment']
    random.shuffle(ent_endings)
    ent_ending1 = ent_endings[0]
    ent_ending2 = ent_endings[1]
    ent_ending3 = ent_endings[2]

    ent_hypothesis, valid_example = combine_sentences(ent_ending1, ent_ending2, ent_ending3)

    return {'premise': prem, 'hypothesis': ent_hypothesis, 'label': 0}, valid_example

def get_hard_pairs(params):

    # We load the data
    train_data_new = load_hf_dataset(params.description, params.split_name, shuffle_dataset=True)

    # Filter the data (for valid labels, and for size)
    train_data_new = train_data_new.filter(lambda example: example['label'] in [0, 1, 2])

    assert params.filter

    train_data_new = train_data_new.filter(
            lambda example, index: index >= params.num_samples and index < params.num_samples + params.num_samples_up, with_indices=True
            )

    dict_duplicate_prems = get_duplicated_premises(train_data_new)

    print("Number of duplicated premises:", len(dict_duplicate_prems))

    entailment_examples = []
    neutral_examples = []
    contradiction_examples = []

    for prem in dict_duplicate_prems:
        if len(dict_duplicate_prems[prem]['entailment']) >= 3:
            new_example, valid_example = assemble_into_entailment_example(
                        dict_duplicate_prems,
                        prem
                        )
            if valid_example:

                entailment_examples.append(
                        new_example)

        if len(dict_duplicate_prems[prem]['entailment']) >= 2 and len(dict_duplicate_prems[prem]['neutral']) >= 1:
            new_example, valid_example = assemble_into_neutral_example(
                        dict_duplicate_prems,
                        prem
                        )
            if valid_example:

                neutral_examples.append(
                        new_example)


        if len(dict_duplicate_prems[prem]['contradiction']) >= 1 and len(dict_duplicate_prems[prem]['neutral']) + len(dict_duplicate_prems[prem]['entailment']) >= 2:

            new_example, valid_example = assemble_into_contradiction_example(
                        dict_duplicate_prems,
                        prem
                        )
            if valid_example:

                contradiction_examples.append(
                        new_example)

    all_examples = contradiction_examples + neutral_examples + entailment_examples

    all_examples_df = Dataset.from_list(all_examples)

    return all_examples_df

if __name__ == '__main__':

    params = get_args()
    
    random.seed(13)
    new_data = get_hard_pairs(params)
    new_data = new_data.shuffle()

    print("Total training obs before augmentation:", len(new_data))
    print("Num. class 0:", len(new_data.filter(lambda example: example['label'] == 0)))
    print("Num. class 1:", len(new_data.filter(lambda example: example['label'] == 1)))
    print("Num. class 2:", len(new_data.filter(lambda example: example['label'] == 2)))
    print("Total new obs:", len(new_data))

    file_name = params.name_id

    print(file_name)

    assert not os.path.isfile(
                "concat_data/" + file_name + '.hf'), \
                        "Output filename exists, try changing name_id"

    new_data.save_to_disk("concat_data/" + file_name + ".hf")
