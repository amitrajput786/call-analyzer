#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 18:22:39 2025
@author: amit
"""
import whisper
import time
import os

def test_whisper(audio_path, model_path, output_dir):
    """Test Whisper transcription with local model"""
    # Verify both files exist
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None
    
    if not os.path.isfile(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        return None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading Whisper model from: {model_path}")
    start_time = time.time()
    model = whisper.load_model(model_path)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Transcribe audio
    print(f"Transcribing audio: {audio_path}")
    start_time = time.time()
    result = model.transcribe(audio_path)
    transcribe_time = time.time() - start_time
    print(f"Transcription completed in {transcribe_time:.2f} seconds")
    
    # Show results
    print("\n=== Transcription Result ===")
    print(result["text"])
    
    # Save transcript to file with full path
    output_file = os.path.join(output_dir, "whisper_transcript.txt")
    with open(output_file, "w") as f:
        f.write(result["text"])
    print(f"\nTranscript saved to {output_file}")
    
    return result["text"]

if __name__ == "__main__":
    # Use the correct paths that we verified
    audio_path = "/home/amit/april_cohort/audio_file/english_taking_4_people_5mins_set3.mp3"
    model_path = "/home/amit/.cache/whisper/base.pt"
    output_dir = "/home/amit/april_cohort/speech_brain"
    
    transcript = test_whisper(audio_path, model_path, output_dir)