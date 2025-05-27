#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 08:06:24 2025

@author: amit
"""

#!/usr/bin/env python3
"""
Script to download SpeechBrain models locally for offline use.
"""

#!/usr/bin/env python3
"""
Script to download SpeechBrain models locally for offline use.
"""

import os
import torch
import requests
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('speechbrain_downloader')

def download_file(url, dest):
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return dest

def download_speechbrain_model(source_path, save_dir):
    """
    Download a SpeechBrain model for offline use.
    
    Args:
        source_path (str): Path to the source model on Hugging Face Hub (e.g., 'speechbrain/spkrec-ecapa-voxceleb')
        save_dir (str): Directory where the model should be saved locally
    """
    logger.info(f"Downloading SpeechBrain model from {source_path} to {save_dir}")
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine hub/organization and model name from source path
    parts = source_path.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid source path: {source_path}. Expected format: 'organization/model-name'")
    
    hub, model_name = parts
    
    # Path to the hyperparameter file on HuggingFace
    hparams_file = f"https://huggingface.co/{source_path}/resolve/main/hyperparams.yaml"
    
    # Download hyperparams.yaml
    logger.info("Downloading hyperparams.yaml")
    local_hparams_path = os.path.join(save_dir, "hyperparams.yaml")
    download_file(hparams_file, local_hparams_path)
    
    # Create the metadata file required by SpeechBrain
    metadata = {
        "source": source_path,
        "date": "2025-04-14"
    }
    
    with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
        json.dump(metadata, f)
    
    # Download the model files
    logger.info("Downloading model files - this may take a while...")
    
    # Common model files to download
    files_to_download = [
        "classifier.ckpt",  # Main model checkpoint
        "embedding_model.ckpt",  # Embedding model checkpoint
        "mean_var_norm_emb.ckpt",  # Normalization parameters
        "label_encoder.txt",  # Label encoder
    ]
    
    for file in files_to_download:
        file_url = f"https://huggingface.co/{source_path}/resolve/main/{file}"
        try:
            local_file_path = os.path.join(save_dir, file)
            logger.info(f"Downloading {file}")
            download_file(file_url, local_file_path)
        except Exception as e:
            logger.warning(f"Failed to download {file}: {e}. This file might not be required.")
    
    logger.info(f"Model successfully downloaded to {save_dir}")
    return save_dir

if __name__ == "__main__":
    # The model you want to download (speaker recognition model)
    SOURCE_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    
    # The directory where you want to save the model
    SAVE_DIR = "/home/amit/april_cohort/pretrained_models/spkrec-ecapa-voxceleb"
    
    downloaded_path = download_speechbrain_model(SOURCE_MODEL, SAVE_DIR)
    print(f"Model downloaded to: {downloaded_path}")
    print("You can now use this local model in your application.")