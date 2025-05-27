#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:38:01 2025

@author: amit
"""

import torch
import whisper
from pathlib import Path
import os

def download_whisper_base():
    """
    Downloads the Whisper base model and returns the model object.
    Model size is approximately 142MB.
    """
    print("Downloading Whisper base model...")
    # This will download the model if it's not already cached
    model = whisper.load_model("base")
    print("Model downloaded successfully!")
    
    # Print where the model is cached
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    print(f"Model cached at: {cache_dir}")
    
    return model

def transcribe_audio(model, audio_file_path, language=None):
    """
    Transcribes audio using the whisper model.
    
    Args:
        model: The whisper model object
        audio_file_path: Path to the audio file to transcribe
        language: Optional language code (e.g., "hi" for Hindi). If None, language will be detected.
    
    Returns:
        The transcription result dictionary
    """
    print(f"Transcribing: {audio_file_path}")
    
    # Transcribe audio
    transcription_options = {}
    if language:
        transcription_options["language"] = language
    
    result = model.transcribe(audio_file_path, **transcription_options)
    
    return result

def main():
    # Download the model
    model = download_whisper_base()
    
    # Example usage
    audio_path = "/home/amit/april_cohort/audio_file/WhatsApp Audio 2025-04-12 at 11.41.25.mpga.mp3"  # Replace with your audio file path
    
    # For Hindi audio
    result = transcribe_audio(model, audio_path, language="hi")
    
    # Print results
    print("\nTranscription Result:")
    print(f"Detected language: {result['language']}")
    print(f"Transcription: {result['text']}")
    
    # If you need segments with timestamps
    print("\nSegments with timestamps:")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

if __name__ == "__main__":
    main()
    
    
""" the  path save dir:/home/amit/.cache/whisper"""








"code after the modification  of it "


import torch
import whisper
from pathlib import Path
import os
import shutil

def download_whisper_base(custom_path=None):
    """
    Downloads the Whisper base model and optionally saves it to a custom location.
    Model size is approximately 142MB.
    
    Args:
        custom_path: Optional path to save the model to. If None, the model is cached at the default location.
    """
    print("Downloading Whisper base model...")
    # This will download the model if it's not already cached
    model = whisper.load_model("base")
    print("Model downloaded successfully!")
    
    # Print where the model is cached by default
    default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    print(f"Model cached at: {default_cache_dir}")
    
    # If custom path is provided, copy the model files there
    if custom_path:
        custom_path = Path(custom_path)
        custom_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the model files from cache to custom location
        try:
            # Copy the relevant model files from the default cache
            model_files = [f for f in os.listdir(default_cache_dir) if f.startswith('base')]
            for file in model_files:
                source_path = os.path.join(default_cache_dir, file)
                dest_path = os.path.join(custom_path, file)
                shutil.copy2(source_path, dest_path)
            print(f"Model copied to custom path: {custom_path}")
        except Exception as e:
            print(f"Error copying model files: {e}")
    
    return model

def load_model_from_path(model_path):
    """
    Load a Whisper model from a specific path instead of downloading it.
    
    Args:
        model_path: Path where the model files are stored
    
    Returns:
        The loaded Whisper model
    """
    # This is a simplified approach - in practice, you might need to adjust this
    # based on how whisper actually loads models from disk
    try:
        # Set environment variable to instruct whisper where to look for models
        # This is an approximation and might need adjustment based on whisper's implementation
        os.environ["WHISPER_HOME"] = str(model_path)
        model = whisper.load_model("base", download_root=model_path)
        print(f"Model loaded from custom path: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from path: {e}")
        print("Falling back to default cached model...")
        return whisper.load_model("base")

def transcribe_audio(model, audio_file_path, language=None):
    """
    Transcribes audio using the whisper model.
    
    Args:
        model: The whisper model object
        audio_file_path: Path to the audio file to transcribe
        language: Optional language code (e.g., "hi" for Hindi). If None, language will be detected.
    
    Returns:
        The transcription result dictionary
    """
    print(f"Transcribing: {audio_file_path}")
    
    # Transcribe audio
    transcription_options = {}
    if language:
        transcription_options["language"] = language
    
    result = model.transcribe(audio_file_path, **transcription_options)
    
    return result

def main():
    # Define custom path to save model
    custom_model_path = "/home/amit/april_cohort/pretrained_models"  # Replace with your desired path
    
    # Download the model and save to custom path
    model = download_whisper_base(custom_model_path)










import torch
import whisper
from pathlib import Path
import os

def download_whisper_base(custom_path=None):
    """
    Downloads the Whisper base model to a custom location.
    Model size is approximately 142MB.
    
    Args:
        custom_path: Path to save the model to. If None, uses default cache.
        
    Returns:
        The loaded whisper model
    """
    print("Downloading Whisper base model...")
    
    if custom_path:
        # Ensure the directory exists
        os.makedirs(custom_path, exist_ok=True)
        
        # Use download_root parameter to specify where to save the model
        model = whisper.load_model("base", download_root=custom_path)
        print(f"Model downloaded and saved to: {custom_path}")
    else:
        # Use default cache location
        model = whisper.load_model("base")
        default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        print(f"Model cached at default location: {default_cache_dir}")
    
    print("Model downloaded successfully!")
    return model

def load_model_from_custom_path(custom_path):
    """
    Load a Whisper model from a custom path.
    
    Args:
        custom_path: Path where the model files are stored
    
    Returns:
        The loaded Whisper model
    """
    try:
        # Check if path exists
        if not os.path.exists(custom_path):
            print(f"Custom path {custom_path} does not exist.")
            print("Falling back to default cached model...")
            return whisper.load_model("base")
            
        model = whisper.load_model("base", download_root=custom_path)
        print(f"Model loaded from custom path: {custom_path}")
        return model
    except Exception as e:
        print(f"Error loading model from custom path: {e}")
        print("Falling back to default cached model...")
        return whisper.load_model("base")

def transcribe_audio(model, audio_file_path, language=None):
    """
    Transcribes audio using the whisper model.
    
    Args:
        model: The whisper model object
        audio_file_path: Path to the audio file to transcribe
        language: Optional language code (e.g., "hi" for Hindi). If None, language will be detected.
    
    Returns:
        The transcription result dictionary
    """
    print(f"Transcribing: {audio_file_path}")
    
    # Transcribe audio
    transcription_options = {}
    if language:
        transcription_options["language"] = language
    
    result = model.transcribe(audio_file_path, **transcription_options)
    
    return result

def print_transcription_results(result):
    """
    Print the transcription results in a readable format
    
    Args:
        result: The transcription result dictionary from whisper
    """
    print("\nTranscription Result:")
    print(f"Detected language: {result['language']}")
    print(f"Transcription: {result['text']}")
    
    # Print segments with timestamps
    print("\nSegments with timestamps:")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

def main():
    # Define custom path to save model
    custom_model_path = "/home/amit/april_cohort/pretrained_models"
    
    # Option 1: Download model and save to custom path
    model = download_whisper_base(custom_model_path)
    
    # Option 2: Load model from custom path (for future runs)
    # Uncomment the line below after first run (when model is already downloaded)
    # model = load_model_from_custom_path(custom_model_path)
    
    # Example usage with audio file
    audio_path = "/home/amit/april_cohort/audio_file/WhatsApp Audio 2025-04-12 at 11.41.25.mpga.mp3"  # Replace with your audio file path
    
    # For Hindi audio
    result = transcribe_audio(model, audio_path, language="hi")
    
    # Print results
    print_transcription_results(result)

if __name__ == "__main__":
    main()





# First, let's check which whisper module you have
import sys
import whisper
print(f"Whisper module path: {whisper.__file__}")
print(f"Whisper version: {whisper.__version__ if hasattr(whisper, '__version__') else 'Unknown'}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper Speech Recognition Model Handler
- Downloads and saves Whisper model to custom path
- Loads model from custom path
- Transcribes audio files with language detection or specification
"""

import os
import sys
from pathlib import Path

# First try to import openai-whisper correctly
try:
    import whisper
    print(f"Using whisper from: {whisper.__file__}")
except AttributeError:
    print("Error: The whisper module doesn't seem to be OpenAI's whisper.")
    print("Please install the correct package with: pip install openai-whisper")
    sys.exit(1)

def download_whisper_base(custom_path=None):
    """
    Downloads the Whisper base model to a custom location.
    Model size is approximately 142MB.
    """
    print("Downloading Whisper base model...")
    
    try:
        if custom_path:
            # Ensure the directory exists
            os.makedirs(custom_path, exist_ok=True)
            
            # Check if download_root is a valid parameter
            if 'download_root' in whisper.load_model.__code__.co_varnames:
                model = whisper.load_model("base", download_root=custom_path)
                print(f"Model downloaded and saved to: {custom_path}")
            else:
                # Alternative approach if download_root is not supported
                # Set environment variable that might affect where the model is downloaded
                original_cache_dir = os.environ.get("XDG_CACHE_HOME")
                os.environ["XDG_CACHE_HOME"] = custom_path
                model = whisper.load_model("base")
                if original_cache_dir:
                    os.environ["XDG_CACHE_HOME"] = original_cache_dir
                else:
                    os.environ.pop("XDG_CACHE_HOME", None)
                print(f"Model downloaded and saved to: {custom_path}")
        else:
            # Use default cache location
            model = whisper.load_model("base")
            default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            print(f"Model cached at default location: {default_cache_dir}")
        
        print("Model downloaded successfully!")
        return model
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Make sure you have the OpenAI Whisper package installed:")
        print("pip install openai-whisper")
        sys.exit(1)

# Rest of your functions remain the same

import whisper
print(f"Whisper module path: {whisper.__file__}")
print(dir(whisper))  # This will list all available functions/attributes
































