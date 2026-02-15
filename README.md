# Overview

This is the repo for the paper _**Improving the OOD Performance of Closed-Source LLMs on NLI Through Strategic Data Selection**_, published at EACL Findings 2026.

[You can find the paper here](https://arxiv.org/pdf/2505.20209)

<img width="663" height="214" alt="Screenshot 2026-02-15 at 20 06 35" src="https://github.com/user-attachments/assets/aa671f7c-caaa-42c1-8822-8a9e0a93c9af" />

For any questions, please contact j.stacey@sheffield.ac.uk

### Instructions for using this repo:

- The datasets created for each method (for gpt-4o-mini) are saved in saved_data_for_training
- To reproduce the gpt-4o-mini results in the paper, first fine-tune a model with these datasets (follow Section 1), then perform inference using this model (follow Section 5), and finally follow Section 6 for evaluation.
- If you are recreating each training dataset from scratch, please follow Sections 1-6 in full. Please note that openai responses are not fully deterministic, and the datasets created will differ from those in saved_data_for_training.

### Section 1. Fine-tuning models

- The data created for each method (using gpt-4o-mini) has already been saved in saved_data_for_training/ 

- To fine-tune a model with one of these datasets, e.g. the baseline model (snli_10k_baseline.jsonl), upload this dataset to OpenAI and then use:
python run_fine_tuning.py --model_type gpt4mini --id id_of_data_on_openai

### Section 2. Creating the data used for fine-tuning

This section explains how to create the datasets used in section 1 (these are already created in saved_data_for_training for gpt-4o-mini).

- To create the baseline training data:
  
python create_train_data.py --description snli --split_name train --do_replacement 0 --upload_to_openai no --name_id snli_10k_baseline

- To get the training data for the difficulty score sampling method:
  
python create_train_data.py --description snli --split_name train --save 1 --upload_to_openai no --sample_file_nextsplit training_preds_baseline --do_replacement 1 --method_name difficulty_score --difficulty_score_location saved_baseline_preds/questions_on_first_20k_examples_for_difficulty_score_sampling.json --name_id snli_difficulty_score_sampling

- To get the training data for the uncertainty sampling method:
  
python create_train_data.py --description snli --split_name train --upload_to_openai no --sample_file_nextsplit training_preds_baseline --do_replacement 1 --method_name uncertainty_sampling --name_id snli_uncertainty_sampling

- To get the training data for the misclassified sampling method:
  
python create_train_data.py --description snli --split_name train --upload_to_openai no --sample_file_nextsplit training_preds_baseline --do_replacement 1 --method_name misclassified_sampling --name_id snli_misclassified_sampling

- To get the training data for the concatenative hypothesis method:
  
python create_train_data.py --description snli --split_name train --upload_to_openai no --sample_file_nextsplit training_preds_baseline --do_replacement 1 --method_name concat --name_id snli_hyp_concat --concat_data_location snli_concat_data

### Section 3a. Creating the questions_on_first_20k_examples_for_difficulty_score_sampling.json file required in section 2:

questions_on_first_20k_examples_for_difficulty_score_sampling.json contains the model responses with difficulty scores for each training instance (using gpt-4o-mini). This file has already been provided for you (created using gpt-4o-mini).

- To recreate this:

python get_difficulty_scores.py --model_type gpt4mini --num_samples 20000 --filter 1 --name_id questions_on_first_20k_examples_for_difficulty_score_sampling

### Section 3b. Creating the training_preds_baseline file required in section 2:

The training_preds_baseline file described above contains predictions from the baseline fine-tuned model on the next 10k training examples (D_potential in the paper). This file has already been provided for you (created using gpt-4o-mini).

- To create this file, first you need to train a baseline model. You can follow the steps in 'Section 1. Fine-tuning models' above to do this using the baseline training data.

- Then you need to make predictions on the training data using this fine-tuned model:

python run_inference.py --model_type gpt4mini --name_id training_preds_baseline --model_id gpt-model-id_from_trained_openai_model --make_predictions_on_training_data 1 --training_data_to_make_preds_on snli --filter_type nextsplit --num_runs 1 --gpt_soft_probs 1

### Section 3c. Creating the snli_concat_data file required in section 2:

To be added...

### Section 4. Generating the synthetic data

To be added...

### Section 5. Running inference from a fine-tuned model

- First, you need to download the inli_test.csv dataset from https://github.com/google-deepmind/inli/blob/main/INLI%20Data/test.csv and put this within data/

- Then, you can run inference across each of the different test sets:

python run_inference.py --model_type gpt4mini --model_id id-of-fine-tuned-model-on-openai --name_id file-name-to-save-results --num_runs 1 --random_seed 42

### Section 6. Check performance after running inference

- Finally, to evaluate the results of the saved predictions from Step 5, use:
  
get_results.py --load_id file-name-where-inference-results-are-stored-from-step-5
