#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 01:16:52 2025

@author: amit
"""

import torch
from transformers import AutoModel, AutoTokenizer
import json

# Load the IndicBERT model and tokenizer from your local paths
tokenizer = AutoTokenizer.from_pretrained("/home/amit/indicbert_tokenizer")
model = AutoModel.from_pretrained("/home/amit/indicbert_model")

# Load your prepared data
with open('/home/amit/april_cohort/indicbert_input.json', 'r') as f:
    transcript_data = json.load(f)
    



"what is inside your json file , inspection inside the json file "
import json

with open('/home/amit/april_cohort/indicbert_input.json', 'r') as f:
    transcript_data = json.load(f)

# Print the type and a sample of the data
print(f"Type of transcript_data: {type(transcript_data)}")
if isinstance(transcript_data, str):
    print("First 100 characters:")
    print(transcript_data[:100])
elif isinstance(transcript_data, list):
    print(f"Number of items: {len(transcript_data)}")
    print("First item:")
    print(transcript_data[0])
else:
    print("Keys in transcript_data:")
    print(list(transcript_data.keys()))

"after the modification based on the input "




def process_with_indicbert(transcript_data):
    # Initialize lists to store results
    speaker_ids = []
    timestamps = []
    embeddings = []
    texts = []
    
    # Process utterances from your specific data structure
    utterances = transcript_data.get('utterances', [])
    
    for utterance in utterances:
        # Extract data from each utterance
        speaker = utterance.get('speaker', 'Unknown')
        start_time = utterance.get('start', 0)
        end_time = utterance.get('end', 0)
        text = utterance.get('text', '')
        
        # Skip empty segments
        if not text.strip():
            continue
        
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get model output (embeddings)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding as sentence representation (first token's last hidden state)
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Store results
        speaker_ids.append(speaker)
        timestamps.append((start_time, end_time))
        embeddings.append(sentence_embedding)
        texts.append(text)
    
    return {
        "speakers": speaker_ids,
        "timestamps": timestamps,
        "embeddings": embeddings,
        "texts": texts
    }


import torch
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np

# Load the IndicBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/amit/indicbert_tokenizer")
model = AutoModel.from_pretrained("/home/amit/indicbert_model")

# Load your prepared data
with open('/home/amit/april_cohort/indicbert_input.json', 'r') as f:
    transcript_data = json.load(f)

# Process the transcript with the updated function
results = process_with_indicbert(transcript_data)

# Display sample results
print(f"Processed {len(results['texts'])} segments")
print("\nSample processed segments:")
for i in range(min(5, len(results['texts']))):
    print(f"{results['speakers'][i]} ({results['timestamps'][i][0]}s-{results['timestamps'][i][1]}s): {results['texts'][i][:50]}...")
    print(f"Embedding shape: {results['embeddings'][i].shape}")
    print()

# Save the processed data for use with T5
processed_data = {
    "conversation_id": transcript_data.get("conversation_id", "unknown"),
    "full_text": " ".join(results['texts']),  # Concatenate all text for summarization
    "segments": [
        {
            "speaker": spk,
            "start_time": ts[0],
            "end_time": ts[1],
            "text": txt
        }
        for spk, ts, txt in zip(results['speakers'], results['timestamps'], results['texts'])
    ]
}

# Save to file for T5 input
with open('/home/amit/april_cohort/t5_input.json', 'w') as f:
    json.dump(processed_data, f, indent=2)

print("Saved T5 input file to /home/amit/april_cohort/t5_input.json")











# Process the transcript
results = process_with_indicbert(transcript_data)

# Display sample results
print(f"Processed {len(results['texts'])} segments")
print("\nSample processed segments:")
for i in range(min(5, len(results['texts']))):
    print(f"{results['speakers'][i]} ({results['timestamps'][i][0]}s-{results['timestamps'][i][1]}s): {results['texts'][i][:50]}...")
    print(f"Embedding shape: {results['embeddings'][i].shape}")
    print()

# Save the processed data for use with T5
processed_data = {
    "full_text": " ".join(results['texts']),  # Concatenate all text for summarization
    "segments": [
        {
            "speaker": spk,
            "start_time": ts[0],
            "end_time": ts[1],
            "text": txt
        }
        for spk, ts, txt in zip(results['speakers'], results['timestamps'], results['texts'])
    ]
}

# Save to file for T5 input
with open('/home/amit/april_cohort/t5_input.json', 'w') as f:
    json.dump(processed_data, f, indent=2)

print("Saved T5 input file to /home/amit/april_cohort/t5_input.json")