#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 07:19:06 2025

@author: amit
"""
"create a simplified API version that you can use to integrate the pipeline with your frontend application:"
"Audio processing pipeline API wrapper code"



import os
import json
import time
import logging
from typing import Dict, Any, Optional

# Import the main pipeline
from audio_preprocessing_coordinator import AudioPipeline, get_default_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_api.log')
    ]
)
logger = logging.getLogger('audio_pipeline_api')

class PipelineAPI:
    def __init__(self, base_dir: str = None):
        """Initialize the pipeline API"""
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.config = get_default_config()
        
        # Make sure output directory exists
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Track job status
        self.jobs = {}
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with provided values"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        return self.config
    
    def process_audio(self, 
                      audio_path: str, 
                      job_id: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      force_rerun: bool = False) -> Dict[str, Any]:
        """Process an audio file and return a job ID"""
        # Generate a job ID if not provided
        if not job_id:
            job_id = f"job_{int(time.time())}"
        
        # Set up job-specific output directory
        if output_dir:
            job_output_dir = output_dir
        else:
            job_output_dir = os.path.join(self.config["output_dir"], job_id)
        
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Create job config
        job_config = self.config.copy()
        job_config["audio_path"] = audio_path
        job_config["output_dir"] = job_output_dir
        
        # Update intermediate file paths to include job ID
        for key, value in job_config["intermediate_files"].items():
            job_config["intermediate_files"][key] = os.path.join(job_output_dir, os.path.basename(value))
        
        # Initialize job status
        self.jobs[job_id] = {
            "status": "started",
            "start_time": time.time(),
            "config": job_config,
            "results": None
        }
        
        try:
            # Initialize and run pipeline
            pipeline = AudioPipeline(job_config, force_rerun=force_rerun)
            results = pipeline.run_pipeline()
            
            # Update job status
            self.jobs[job_id]["status"] = results["pipeline_status"]
            self.jobs[job_id]["end_time"] = time.time()
            self.jobs[job_id]["duration"] = self.jobs[job_id]["end_time"] - self.jobs[job_id]["start_time"]
            self.jobs[job_id]["results"] = results
            
            return {
                "job_id": job_id,
                "status": results["pipeline_status"],
                "output_dir": job_output_dir,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["end_time"] = time.time()
            self.jobs[job_id]["duration"] = self.jobs[job_id]["end_time"] - self.jobs[job_id]["start_time"]
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a specific job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        return {
            "job_id": job_id,
            "status": self.jobs[job_id]["status"],
            "start_time": self.jobs[job_id].get("start_time"),
            "end_time": self.jobs[job_id].get("end_time"),
            "duration": self.jobs[job_id].get("duration"),
            "output_dir": self.jobs[job_id]["config"]["output_dir"]
        }
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get the full results of a specific job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        if self.jobs[job_id]["status"] not in ["completed", "partially_completed"]:
            return {
                "job_id": job_id,
                "status": self.jobs[job_id]["status"],
                "error": "Results not available yet or job failed"
            }
        
        try:
            # Initialize pipeline to use its collect_results method
            pipeline = AudioPipeline(self.jobs[job_id]["config"])
            detailed_results = pipeline.collect_results()
            
            return {
                "job_id": job_id,
                "status": self.jobs[job_id]["status"],
                "results": detailed_results
            }
            
        except Exception as e:
            logger.error(f"Error collecting results for job {job_id}: {str(e)}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": f"Error collecting results: {str(e)}"
            }
    
    def list_jobs(self) -> Dict[str, Any]:
        """List all jobs and their statuses"""
        job_list = {}
        for job_id, job_info in self.jobs.items():
            job_list[job_id] = {
                "status": job_info["status"],
                "start_time": job_info.get("start_time"),
                "end_time": job_info.get("end_time"),
                "duration": job_info.get("duration"),
                "output_dir": job_info["config"]["output_dir"]
            }
        
        return {"jobs": job_list}


# Example of how to use with Flask
if __name__ == "__main__":
    # This is just an example of how you might integrate with Flask
    # You would actually implement this in your backend
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    pipeline_api = PipelineAPI()
    
    @app.route('/process', methods=['POST'])
    def process_audio():
        data = request.json
        audio_path = data.get('audio_path')
        
        if not audio_path:
            return jsonify({"error": "No audio path provided"}), 400
        
        result = pipeline_api.process_audio(
            audio_path=audio_path,
            job_id=data.get('job_id'),
            output_dir=data.get('output_dir'),
            force_rerun=data.get('force_rerun', False)
        )
        
        return jsonify(result)
    
    @app.route('/status/<job_id>', methods=['GET'])
    def get_status(job_id):
        result = pipeline_api.get_job_status(job_id)
        return jsonify(result)
    
    @app.route('/results/<job_id>', methods=['GET'])
    def get_results(job_id):
        result = pipeline_api.get_job_results(job_id)
        return jsonify(result)
    
    @app.route('/jobs', methods=['GET'])
    def list_jobs():
        result = pipeline_api.list_jobs()
        return jsonify(result)
    
    # Example of direct use without Flask
    def example_direct_usage():
        api = PipelineAPI()
        
        # Process an audio file
        result = api.process_audio(
            audio_path="/path/to/audio.mp3",
            job_id="test_job_1"
        )
        
        print(f"Job submitted: {result}")
        
        # Check status
        while True:
            status = api.get_job_status("test_job_1")
            print(f"Job status: {status['status']}")
            
            if status['status'] in ["completed", "partially_completed", "failed"]:
                break
                
            time.sleep(5)
        
        # Get results
        if status['status'] in ["completed", "partially_completed"]:
            results = api.get_job_results("test_job_1")
            print(f"Job results: {results}")