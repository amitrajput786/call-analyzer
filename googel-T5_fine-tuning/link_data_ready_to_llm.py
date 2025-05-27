#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:36:27 2025

@author: amit
"""
import pandas as pd 
# Load your train/validation/test data
train_df = pd.read_csv('/home/amit/audio_text/train.csv')
test_df = pd.read_csv('/home/amit/audio_text/test.csv')

# Set up model and tokenizer
model_id = "google/flan-t5-xl"
prompt_instruction = "Input: "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create datasets
from datasets import Dataset, DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# Calculate input/output lengths for padding
from datasets import concatenate_datasets
import numpy as np

# Assuming 'question' and 'answer' columns in your CSVs
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["transcript"], truncation=True), 
    batched=True, 
    remove_columns=["transcript", "summary"]
)
input_lengths = [len(x) for x in tokenized_inputs["input_ids"]]
max_source_length = int(np.percentile(input_lengths, 85))
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["summary"], truncation=True), 
    batched=True, 
    remove_columns=["transcript", "summary"]
)
target_lengths = [len(x) for x in tokenized_targets["input_ids"]]
max_target_length = int(np.percentile(target_lengths, 90))
print(f"Max target length: {max_target_length}")

# Preprocess function
def preprocess_function(sample, padding="max_length"):
    # Add prompt instruction to questions
    inputs = [prompt_instruction + item for item in sample["transcript"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Save processed datasets
tokenized_dataset["train"].save_to_disk("/home/amit/T5/train")
tokenized_dataset["test"].save_to_disk("/home/amit/T5/eval")

# Preview a sample
print(tokenized_dataset["train"][0])










