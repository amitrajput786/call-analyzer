#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 11:46:27 2025

@author: amit
"""

# app.py - Flask backend for audio pipeline
from flask import Flask, request, jsonify, render_template, send_from_directory
from pipeline_api import PipelineAPI
import os
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('flask_app.log')
    ]
)
logger = logging.getLogger('flask_app')

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize Pipeline API
pipeline_api = PipelineAPI()
pipeline_api.update_config({"output_dir": RESULTS_FOLDER})

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    # Check if file is in request
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['audioFile']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file extension
    if not file.filename.lower().endswith('.mp3'):
        return jsonify({"error": "Only MP3 files are allowed"}), 400
    
    try:
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save file to upload folder with job ID as prefix
        filename = f"{job_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Start processing the audio file
        result = pipeline_api.process_audio(
            audio_path=file_path,
            job_id=job_id,
            output_dir=os.path.join(RESULTS_FOLDER, job_id)
        )
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "File uploaded and processing started",
            "status": result["status"]
        })
        
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the status of a processing job"""
    try:
        result = pipeline_api.get_job_status(job_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get the results of a processing job"""
    try:
        result = pipeline_api.get_job_results(job_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<job_id>/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    """Download result files"""
    job_dir = os.path.join(RESULTS_FOLDER, job_id)
    
    # Map file_type to actual filename
    file_mapping = {
        "transcript": "whisper_transcript.txt",
        "diarization": "diarization_result.json",
        "aligned": "aligned_transcript.json",
        "summary": "summary_output.txt",
        "sentiment": "sentiment_results.json",
        "formatted": "formatted_transcript.txt"
    }
    
    if file_type not in file_mapping:
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = file_mapping[file_type]
    file_path = os.path.join(job_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_from_directory(job_dir, filename, as_attachment=True)

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all processing jobs"""
    try:
        result = pipeline_api.list_jobs()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)