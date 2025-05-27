#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:01:09 2025

@author: amit
"""

# Install required packages

import torch
from transformers import AutoModel, AutoTokenizer

# Download the indicBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModel.from_pretrained("ai4bharat/indic-bert")

# Save the model and tokenizer locally
model_save_path = "./indicbert_model"
tokenizer_save_path = "./indicbert_tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f"Model saved to {model_save_path}")
print(f"Tokenizer saved to {tokenizer_save_path}")

# Verify we can load from the local paths
local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
local_model = AutoModel.from_pretrained(model_save_path)
print("Successfully loaded model and tokenizer from local paths")




"work-well to download indic bert "

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
login(token="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")

# Download the indicBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModel.from_pretrained("ai4bharat/indic-bert")

# Save the model and tokenizer locally
model_save_path = "./indicbert_model"
tokenizer_save_path = "./indicbert_tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f"Model saved to {model_save_path}")
print(f"Tokenizer saved to {tokenizer_save_path}")

# Example of how to load the locally saved model and tokenizer
local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
local_model = AutoModel.from_pretrained(model_save_path)

































# Import necessary libraries
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Download the indicBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModel.from_pretrained("ai4bharat/indic-bert")

# Save the model and tokenizer locally
model_save_path = "./indicbert_model"
tokenizer_save_path = "./indicbert_tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f"Model saved to {model_save_path}")
print(f"Tokenizer saved to {tokenizer_save_path}")

# Example of how to load the locally saved model and tokenizer
local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
local_model = AutoModel.from_pretrained(model_save_path)

# Example usage for sentiment analysis (you would need to fine-tune the model)
# For a complete sentiment analysis solution, you would need to:
# 1. Fine-tune the model on sentiment data
# 2. Use the fine-tuned model for predictions

# Example for loading a fine-tuned sentiment analysis model (if available)
# sentiment_model = AutoModelForSequenceClassification.from_pretrained("path/to/finetuned/model")

# Example of tokenizing and getting embeddings
text = "यह एक अच्छा दिन है" # "This is a good day" in Hindi
inputs = local_tokenizer(text, return_tensors="pt")
outputs = local_model(**inputs)

# The last hidden states can be used as embeddings
embeddings = outputs.last_hidden_state

print("Successfully loaded and tested the model!")





import torch
from transformers import BertTokenizer, BertModel

# Download the indicBERT model and tokenizer with specific configuration
# Instead of using AutoTokenizer, we'll use BertTokenizer directly
tokenizer = BertTokenizer.from_pretrained("ai4bharat/indic-bert", do_lower_case=True)
model = BertModel.from_pretrained("ai4bharat/indic-bert")

# Save the model and tokenizer locally
model_save_path = "./indicbert_model"
tokenizer_save_path = "./indicbert_tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f"Model saved to {model_save_path}")
print(f"Tokenizer saved to {tokenizer_save_path}")



""" individual code of indic bert """

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def analyze_sentiment(text):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert-sentiment")
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    # Process outputs
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_scores = {
        "positive": predictions[0][2].item(),
        "neutral": predictions[0][1].item(),
        "negative": predictions[0][0].item()
    }
    
    return sentiment_scores














