#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 00:54:37 2025

@author: amit
"""

import os
import json
import re
from pydub import AudioSegment
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import nltk
nltk.download('punkt')
# Download NLTK resources if not already downloaded


# Define file paths
whisper_output_path = "/home/amit/april_cohort/speech_brain/whisper_transcript.txt"
diarization_path = "/home/amit/april_cohort/speech_brain/diarization_result.json"
output_dir = "/home/amit/april_cohort/speech_brain"
combined_output_path = os.path.join(output_dir, "aligned_transcript.json")


from nltk.tokenize.punkt import PunktSentenceTokenizer
punkt_tokenizer = PunktSentenceTokenizer()
sentences = punkt_tokenizer.tokenize(whisper_transcript)
# Load the diarization results
print("Loading speaker diarization results...")
with open(diarization_path, 'r') as f:
    diarization_data = json.load(f)

# Load the whisper transcript
print("Loading whisper transcript...")
with open(whisper_output_path, 'r') as f:
    whisper_transcript = f.read()

# Function to extract timestamps from Whisper output if available
def extract_whisper_timestamps(transcript):
    # Check if the transcript has timestamps in [HH:MM:SS] format
    timestamp_pattern = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})\]')
    matches = timestamp_pattern.finditer(transcript)
    
    segments = []
    prev_end = 0
    
    for match in matches:
        # Convert timestamp to seconds
        h, m, s = map(int, match.groups())
        timestamp_sec = h * 3600 + m * 60 + s
        
        # Extract text from previous timestamp to this one
        text = transcript[prev_end:match.start()].strip()
        if text:  # Only add if there's actual text
            segments.append({
                "text": text,
                "timestamp": timestamp_sec
            })
        
        prev_end = match.end()
    
    # Add the last segment
    text = transcript[prev_end:].strip()
    if text:
        segments.append({
            "text": text,
            "timestamp": None  # No timestamp available for the end
        })
    
    return segments

# First check if Whisper output has timestamps embedded in it
whisper_segments = extract_whisper_timestamps(whisper_transcript)

from nltk.tokenize.punkt import PunktSentenceTokenizer

if not whisper_segments or len(whisper_segments) <= 1:
    print("No embedded timestamps found in Whisper output. Using sentence tokenization...")
    
    # Tokenize the transcript into sentences - USE THIS DIRECTLY
    punkt_tokenizer = PunktSentenceTokenizer()
    sentences = punkt_tokenizer.tokenize(whisper_transcript)
    
    # Estimate total duration from diarization data
    total_duration = max(segment["end"] for segment in diarization_data)
    
    # Distribute sentences across the timeline proportionally
    total_chars = sum(len(sentence) for sentence in sentences)
    char_duration = total_duration / total_chars
    
    whisper_segments = []
    current_time = 0
    
    for sentence in sentences:
        sentence_duration = len(sentence) * char_duration
        whisper_segments.append({
            "text": sentence,
            "start": current_time,
            "end": current_time + sentence_duration
        })
        current_time += sentence_duration

# Now assign speakers to each segment based on time overlap
print("Aligning transcript with speaker information...")
aligned_transcript = []

# Function to find the dominant speaker for a given time segment
def get_speaker_for_segment(start, end, diarization_data):
    # Find all speakers who speak during this segment
    speakers = {}
    for entry in diarization_data:
        # Check for overlap between the segment and this diarization entry
        if not (end <= entry["start"] or start >= entry["end"]):
            # Calculate the overlap duration
            overlap_start = max(start, entry["start"])
            overlap_end = min(end, entry["end"])
            overlap_duration = overlap_end - overlap_start
            
            # Add to speaker's total speaking time
            speaker = entry["speaker"]
            if speaker in speakers:
                speakers[speaker] += overlap_duration
            else:
                speakers[speaker] = overlap_duration
    
    # Return the speaker with the longest speaking time
    if speakers:
        return max(speakers, key=speakers.get)
    else:
        return "Unknown"

# Process each whisper segment
for i, segment in enumerate(whisper_segments):
    if "start" in segment and "end" in segment:
        # Already has timing info
        start_time = segment["start"]
        end_time = segment["end"]
    elif "timestamp" in segment and segment["timestamp"] is not None:
        # Has a timestamp - estimate duration based on text length
        start_time = segment["timestamp"]
        chars_per_second = 15  # Approximate speaking rate
        duration = len(segment["text"]) / chars_per_second
        end_time = start_time + duration
    else:
        # No timing info - assign based on position in transcript
        # This is very approximate
        segment_fraction = i / len(whisper_segments)
        start_time = segment_fraction * total_duration
        end_time = (i + 1) / len(whisper_segments) * total_duration
    
    # Find the most likely speaker for this segment
    speaker = get_speaker_for_segment(start_time, end_time, diarization_data)
    
    # Add to aligned transcript
    aligned_transcript.append({
        "speaker": speaker,
        "text": segment["text"],
        "start": round(start_time, 2),
        "end": round(end_time, 2)
    })

# Save the aligned transcript
with open(combined_output_path, 'w') as f:
    json.dump(aligned_transcript, f, indent=2)

print(f"Aligned transcript saved to {combined_output_path}")

# Create a cleaner version for model input
# Format each entry as "Speaker: Text"
formatted_output_path = os.path.join(output_dir, "formatted_transcript.txt")
formatted_output = []

for entry in aligned_transcript:
    formatted_output.append(f"{entry['speaker']}: {entry['text']}")

with open(formatted_output_path, 'w') as f:
    f.write("\n".join(formatted_output))

print(f"Formatted transcript saved to {formatted_output_path}")

# Create a more structured format for indicBERT input
# This format includes proper conversation structure with speakers and utterances
indicbert_input_path = os.path.join(output_dir, "indicbert_input.json")

# Group consecutive segments from the same speaker
condensed_transcript = []
current_speaker = None
current_text = []
current_start = None
current_end = None

for entry in aligned_transcript:
    if entry["speaker"] == current_speaker:
        # Same speaker, append text
        current_text.append(entry["text"])
        current_end = entry["end"]
    else:
        # New speaker, save previous entry if exists
        if current_speaker:
            condensed_transcript.append({
                "speaker": current_speaker,
                "text": " ".join(current_text),
                "start": current_start, 
                "end": current_end
            })
        # Start new entry
        current_speaker = entry["speaker"]
        current_text = [entry["text"]]
        current_start = entry["start"]
        current_end = entry["end"]

# Add the last entry
if current_speaker:
    condensed_transcript.append({
        "speaker": current_speaker,
        "text": " ".join(current_text),
        "start": current_start,
        "end": current_end
    })

# Format for indicBERT
indicbert_input = {
    "conversation_id": "transcript_1",
    "utterances": [
        {
            "speaker": entry["speaker"],
            "text": entry["text"],
            "start_time": entry["start"],
            "end_time": entry["end"]
        } for entry in condensed_transcript
    ]
}

with open(indicbert_input_path, 'w') as f:
    json.dump(indicbert_input, f, indent=2)

print(f"IndicBERT input file saved to {indicbert_input_path}")

# Print sample of the aligned transcript
print("\nSample of aligned transcript:")
for entry in aligned_transcript[:5]:  # Show first 5 entries
    print(f"{entry['speaker']} ({entry['start']:.1f}s-{entry['end']:.1f}s): {entry['text'][:50]}...")

print("\nDone!")