#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 07:21:52 2025

@author: amit
"""
"For backend integration:"
from pipeline_api import PipelineAPI

# Initialize API
api = PipelineAPI()

# Process an audio file
result = api.process_audio(audio_path="/path/to/uploaded/audio.mp3")
job_id = result["job_id"]

# Get results when complete
status = api.get_job_status(job_id)
if status["status"] == "completed":
    results = api.get_job_results(job_id)