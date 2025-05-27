#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 00:26:18 2025

@author: amit
"""

# Install required packages
!pip install speechbrain torch torchaudio librosa scikit-learn

import os
import torch
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
import json

from pathlib import Path

# Define paths
audio_path = "/home/amit/april_cohort/audio_file/english_taking_4_people_5mins_set3.mp3"
output_dir = "/home/amit/april_cohort/speech_brain"
output_path = os.path.join(output_dir, "diarization_result.json")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define parameters for segmentation
segment_length = 3.0  # in seconds
step_size = 1.0  # in seconds
sampling_rate = 16000  # standard for most speech models

# Load the speaker embedding model
print("Loading speaker embedding model...")
speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(output_dir, "pretrained_models/spkrec-ecapa-voxceleb")
)
print("Speaker embedding model loaded successfully.")

# Load and preprocess audio
print(f"Loading audio file: {audio_path}")
audio, file_sr = librosa.load(audio_path, sr=None)
# Resample if needed
if file_sr != sampling_rate:
    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sampling_rate)

# Audio length in seconds
audio_length = len(audio) / sampling_rate
print(f"Audio length: {audio_length:.2f} seconds")

# Function to extract embeddings from audio segments
def extract_embeddings(audio, sampling_rate, segment_length, step_size):
    embeddings = []
    segments = []
    
    # Calculate number of segments
    num_segments = int((len(audio) / sampling_rate - segment_length) / step_size) + 1
    
    for i in range(num_segments):
        start_sample = int(i * step_size * sampling_rate)
        end_sample = int(start_sample + segment_length * sampling_rate)
        
        if end_sample > len(audio):
            break
        
        segment = audio[start_sample:end_sample]
        
        # Convert to torch tensor with proper dimensions for the model
        with torch.no_grad():
            segment_tensor = torch.FloatTensor(segment).unsqueeze(0)
            embedding = speaker_model.encode_batch(segment_tensor)
            embeddings.append(embedding.squeeze().cpu().numpy())
        
        # Store segment timing
        start_time = start_sample / sampling_rate
        end_time = end_sample / sampling_rate
        segments.append((start_time, end_time))
    
    return np.array(embeddings), segments

# Extract embeddings and segment information
print("Extracting speaker embeddings from audio segments...")
embeddings, segments = extract_embeddings(audio, sampling_rate, segment_length, step_size)
print(f"Extracted {len(embeddings)} segment embeddings")

# Determine optimal number of speakers using silhouette analysis (or pre-specify if known)
from sklearn.metrics import silhouette_score

# Estimate number of speakers (between 2 and 6)
if len(embeddings) > 10:  # Need enough segments for clustering
    max_speakers = min(6, len(embeddings) - 1)  # Don't try more clusters than we have samples
    best_score = -1
    best_n_speakers = 2  # Default
    
    for n_speakers in range(2, max_speakers + 1):
        clustering = AgglomerativeClustering(n_clusters=n_speakers)
        labels = clustering.fit_predict(embeddings)
        
        if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
            score = silhouette_score(embeddings, labels)
            print(f"Silhouette score for {n_speakers} speakers: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_n_speakers = n_speakers
    
    n_speakers = best_n_speakers
    print(f"Estimated optimal number of speakers: {n_speakers}")
else:
    n_speakers = 2  # Default to 2 speakers if we don't have enough segments
    print("Too few segments for reliable automatic speaker count estimation. Using default of 2 speakers.")

# Perform clustering to determine speaker for each segment
print(f"Performing speaker clustering with {n_speakers} speakers...")
clustering = AgglomerativeClustering(n_clusters=n_speakers)
labels = clustering.fit_predict(embeddings)

# Create diarization result
diarization_result = []
for i, ((start_time, end_time), label) in enumerate(zip(segments, labels)):
    diarization_result.append({
        "speaker": f"Speaker_{label+1}",
        "start": round(start_time, 2),
        "end": round(end_time, 2),
    })

# Post-process: merge consecutive segments from the same speaker
merged_result = []
current_segment = None

for segment in diarization_result:
    if current_segment is None:
        current_segment = segment.copy()
    elif current_segment["speaker"] == segment["speaker"]:
        # Same speaker, extend end time
        current_segment["end"] = segment["end"]
    else:
        # Different speaker, add the current segment and start a new one
        merged_result.append(current_segment)
        current_segment = segment.copy()

# Add the last segment
if current_segment is not None:
    merged_result.append(current_segment)

# Save diarization results to a JSON file
with open(output_path, 'w') as f:
    json.dump(merged_result, f, indent=2)

print(f"Diarization completed. Results saved to {output_path}")



# Print sample of the diarization result
print("\nSample of diarization result:")
for segment in merged_result[:10]:  # Show first 10 segments
    print(f"{segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s")

print(f"\nTotal segments: {len(merged_result)}")
print("Done!")