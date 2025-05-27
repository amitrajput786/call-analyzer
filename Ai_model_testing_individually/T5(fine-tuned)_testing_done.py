#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 01:42:10 2025

@author: amit
"""
'we have used our T5 testing dependencies , to summarize our transcript.txt file from our fine-tuned model '

from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM
import torch
import os

# Load tokenizer from the fine-tuned model directory
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./T5-fine-tuned-modelC")

# Load base model
print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", load_in_8bit=True, device_map="auto")

# Load PEFT adapter on top of the base model
print("Loading PEFT adapter...")
peft_model = PeftModel.from_pretrained(base_model, "./T5-fine-tuned-modelC")
print("PEFT adapter model loaded successfully!")

# Path for input and output files
input_file = '/home/amit/april_cohort/speech_brain/whisper_transcript.txt'
output_file = '/home/amit/april_cohort/summary_output.txt'

# Create output directory if it doesn't exist
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Load the prepared input as plain text
print(f"Reading text from: {input_file}")
try:
    with open(input_file, 'r') as f:
        full_text = f.read().strip()
    
    print(f"Successfully loaded text file. Length: {len(full_text)} characters")
    
    # Prepare input for T5 (add prefix as needed based on your fine-tuning)
    input_text = "summarize: " + full_text  # Adjust prefix based on how you fine-tuned T5
    
    # Tokenize
    print("Tokenizing text...")
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Move inputs to the correct device
    input_ids = inputs.input_ids.to(peft_model.device)
    attention_mask = inputs.attention_mask.to(peft_model.device) if 'attention_mask' in inputs else None
    
    # Generate summary
    print("Generating summary...")
    with torch.no_grad():
        summary_ids = peft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("\nGenerated Summary:")
    print(summary)
    
    # Save summary
    with open(output_file, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to {output_file}")

except FileNotFoundError:
    print(f"Error: Input file {input_file} not found.")
except Exception as e:
    print(f"Error during summarization: {str(e)}")

























