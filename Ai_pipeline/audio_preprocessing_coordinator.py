#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 08:20:26 2025

@author: amit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 07:17:20 2025

@author: amit
"""
"audio processing pipeline coordinator code "
import os
import time
import json
import torch
import subprocess
import logging
import numpy as np
import librosa
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger('audio_pipeline')

class AudioPipeline:
    def __init__(self, config: Dict[str, Any], force_rerun: bool = False):
        self.config = config
        self.force_rerun = force_rerun
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Prepare paths
        self._setup_paths()
        
        # Initialize results dictionary
        self.results = {
            "pipeline_status": "initialized",
            "steps_completed": [],
            "errors": [],
            "output_files": {}
        }

    def _setup_paths(self):
        """Setup all file paths for inputs and outputs"""
        # Ensure all intermediate file paths are absolute
        for key, file_name in self.config["intermediate_files"].items():
            self.config["intermediate_files"][key] = os.path.join(
                self.config["output_dir"], file_name
            )
    
    def _file_exists(self, file_path: str) -> bool:
        """Check if a file exists and is not empty"""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    def run_with_progress(self, func_name: str, *args, **kwargs):
        """Run a function with progress tracking"""
        logger.info(f"Starting {func_name}...")
        start_time = time.time()
        
        try:
            result = getattr(self, func_name)(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Completed {func_name} in {duration:.2f} seconds")
            self.results["steps_completed"].append(func_name)
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error in {func_name} after {duration:.2f} seconds: {str(e)}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return False
    
    def run_diarization(self) -> bool:
        """Run speech diarization using SpeechBrain"""
        output_path = self.config["intermediate_files"]["diarization"]
        
        # Skip if file exists and force_rerun is False
        if self._file_exists(output_path) and not self.force_rerun:
            logger.info(f"Diarization output already exists at {output_path}")
            return True
        
        try:
            # Import here to avoid loading models if not needed
            import librosa
            import numpy as np
            import torch
            from speechbrain.pretrained import EncoderClassifier
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            
            # Define parameters for segmentation
            segment_length = 3.0  # in seconds
            step_size = 1.0  # in seconds
            sampling_rate = 16000  # standard for most speech models
            
            # Load the speaker embedding model
            logger.info("Loading speaker embedding model...")
            speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.config["model_paths"]["speechbrain"]
            )
            logger.info("Speaker embedding model loaded successfully.")
            
            # Load and preprocess audio
            logger.info(f"Loading audio file: {self.config['audio_path']}")
            audio, file_sr = librosa.load(self.config['audio_path'], sr=None)
            # Resample if needed
            if file_sr != sampling_rate:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sampling_rate)
                
            # Audio length in seconds
            audio_length = len(audio) / sampling_rate
            logger.info(f"Audio length: {audio_length:.2f} seconds")
            
            # Extract embeddings from audio segments
            embeddings = []
            segments = []
            
            # Calculate number of segments
            num_segments = int((len(audio) / sampling_rate - segment_length) / step_size) + 1
            
            logger.info("Extracting speaker embeddings from audio segments...")
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
            
            embeddings = np.array(embeddings)
            logger.info(f"Extracted {len(embeddings)} segment embeddings")
            
            # Determine optimal number of speakers using silhouette analysis
            if len(embeddings) > 10:  # Need enough segments for clustering
                max_speakers = min(6, len(embeddings) - 1)  # Don't try more clusters than we have samples
                best_score = -1
                best_n_speakers = 2  # Default
                
                for n_speakers in range(2, max_speakers + 1):
                    clustering = AgglomerativeClustering(n_clusters=n_speakers)
                    labels = clustering.fit_predict(embeddings)
                    
                    if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
                        score = silhouette_score(embeddings, labels)
                        logger.info(f"Silhouette score for {n_speakers} speakers: {score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_n_speakers = n_speakers
                
                n_speakers = best_n_speakers
                logger.info(f"Estimated optimal number of speakers: {n_speakers}")
            else:
                n_speakers = 2  # Default to 2 speakers if we don't have enough segments
                logger.info("Too few segments for reliable speaker count estimation. Using default of 2 speakers.")
            
            # Perform clustering to determine speaker for each segment
            logger.info(f"Performing speaker clustering with {n_speakers} speakers...")
            clustering = AgglomerativeClustering(n_clusters=n_speakers)
            labels = clustering.fit_predict(embeddings)
            
            # Create diarization result
            diarization_result = []
            for i, ((start_time, end_time), label) in enumerate(zip(segments, labels)):
                diarization_result.append({
                    "speaker": f"SPEAKER_{label+1}",
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
            
            logger.info(f"Diarization completed. Results saved to {output_path}")
            self.results["output_files"]["diarization"] = output_path
            return True
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise
    
    def run_whisper(self) -> bool:
        """Run Whisper transcription"""
        output_path = self.config["intermediate_files"]["transcript"]
        
        # Skip if file exists and force_rerun is False
        if self._file_exists(output_path) and not self.force_rerun:
            logger.info(f"Transcription output already exists at {output_path}")
            return True
        
        try:
            # Import here to avoid loading models if not needed
            import whisper
            
            # Load the Whisper model
            model = whisper.load_model("base", download_root=os.path.dirname(self.config["model_paths"]["whisper"]))
            
            # Transcribe the audio
            result = model.transcribe(self.config["audio_path"])
            
            # Save the transcript
            with open(output_path, 'w') as f:
                f.write(result["text"])
            
            self.results["output_files"]["transcript"] = output_path
            return True
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            raise
    
    def run_alignment(self) -> bool:
        """Run alignment between diarization and transcript"""
        diarization_path = self.config["intermediate_files"]["diarization"]
        transcript_path = self.config["intermediate_files"]["transcript"]
        aligned_output = self.config["intermediate_files"]["aligned"]
        formatted_output = os.path.join(self.config["output_dir"], "formatted_transcript.txt")
        indicbert_output = self.config["intermediate_files"]["indicbert_input"]
        
        # Skip if files exist and force_rerun is False
        if (self._file_exists(aligned_output) and 
            self._file_exists(formatted_output) and 
            self._file_exists(indicbert_output) and 
            not self.force_rerun):
            logger.info(f"Alignment outputs already exist")
            return True
        
        try:
            # Load diarization results
            with open(diarization_path, 'r') as f:
                diarization_data = json.load(f)
            
            # Load transcript
            with open(transcript_path, 'r') as f:
                transcript_text = f.read().strip()
            
            # Simple alignment algorithm
            # In a real implementation, you would need a more sophisticated 
            # algorithm that aligns the transcript with the speaker segments
            
            # Placeholder for a proper alignment algorithm
            # This is a simplified version - replace with your actual alignment logic
            words = transcript_text.split()
            total_duration = diarization_data[-1]["end"]
            words_per_second = len(words) / total_duration
            
            aligned_transcript = []
            formatted_lines = []
            indicbert_input = {"utterances": []}
            
            
            for segment in diarization_data:
                speaker = segment["speaker"]
                start = segment["start"]
                end = segment["end"]
                duration = end - start
                
                # Estimate words in this segment
                word_count = int(duration * words_per_second)
                if words:
                    segment_words = words[:word_count]
                    words = words[word_count:]  # Remove used words
                    segment_text = " ".join(segment_words)
                else:
                    segment_text = ""
                
                aligned_transcript.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": segment_text
                })
                
                formatted_lines.append(f"{speaker}: {segment_text}")
                
                indicbert_input["utterances"].append({
               "speaker": speaker,
               "text": segment_text,
               "start_time": start,
               "end_time": end
               })
            
            # Save aligned transcript
            with open(aligned_output, 'w') as f:
                json.dump(aligned_transcript, f, indent=2)
            
            # Save formatted transcript
            with open(formatted_output, 'w') as f:
                f.write("\n".join(formatted_lines))
            
            # Save IndicBERT input
            with open(indicbert_output, 'w') as f:
                json.dump(indicbert_input, f, indent=2)
            
            self.results["output_files"]["aligned"] = aligned_output
            self.results["output_files"]["formatted"] = formatted_output
            self.results["output_files"]["indicbert_input"] = indicbert_output
            return True
            
        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
            raise
    
    def run_sentiment_analysis(self) -> bool:
        """Run IndicBERT sentiment analysis"""
        input_path = self.config["intermediate_files"]["indicbert_input"]
        output_path = os.path.join(self.config["output_dir"], "sentiment_results.json")
        
        # Skip if file exists and force_rerun is False
        if self._file_exists(output_path) and not self.force_rerun:
            logger.info(f"Sentiment analysis output already exists at {output_path}")
            return True
        
        try:
            # Import here to avoid loading models if not needed
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Load IndicBERT model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_paths"]["indicbert_tokenizer"]
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_paths"]["indicbert_model"]
            )
            
            # Load input data
            with open(input_path, 'r') as f:
                input_data = json.load(f)
            
            sentiment_results = []
            utterances=input_data.get("utterances",[])
            for item in utterances:
                
                # Skip empty text
                if not item["text"].strip():
                    sentiment_results.append({
                        "speaker": item["speaker"],
                        "timestamp": item["start_time"],  # Fixed: use start_time instead of timestamp
                        "text": item["text"],
                        "sentiment": "neutral",
                        "sentiment_score": 0.5
                    })
                    continue
                
                # Tokenize and get sentiment
                inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Process outputs
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                sentiment_score = float(scores[0][1].item())  # Assuming binary classification
                
                # Map score to sentiment label
                if sentiment_score > 0.7:
                    sentiment = "positive"
                elif sentiment_score < 0.3:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                sentiment_results.append({
                    "speaker": item["speaker"],
                    "timestamp": item["start_time"],  # Fixed: use start_time instead of timestamp
                    "text": item["text"],
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score
                })
            
            # Save sentiment results
            with open(output_path, 'w') as f:
                json.dump(sentiment_results, f, indent=2)
            
            self.results["output_files"]["sentiment"] = output_path
            return True
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise
    
    def run_summarization(self) -> bool:
        """Run T5 summarization"""
        input_path = self.config["intermediate_files"]["transcript"]
        output_path = self.config["intermediate_files"]["summary"]
        
        # Skip if file exists and force_rerun is False
        if self._file_exists(output_path) and not self.force_rerun:
            logger.info(f"Summarization output already exists at {output_path}")
            return True
        
        try:
            # Import here to avoid loading models if not needed
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            # Load T5 model and tokenizer
            model = T5ForConditionalGeneration.from_pretrained(self.config["model_paths"]["t5_model"])
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            
            # Load transcript
            with open(input_path, 'r') as f:
                transcript = f.read().strip()
            
            # Prepare input
            # T5 requires a specific prefix for summarization
            input_text = "summarize: " + transcript
            
            # Handle long transcripts by chunking
            max_token_length = 512
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in input_text.split():
                tokens = tokenizer.tokenize(word)
                if current_length + len(tokens) > max_token_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(tokens)
                else:
                    current_chunk.append(word)
                    current_length += len(tokens)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Process each chunk
            summary_parts = []
            
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", max_length=max_token_length, truncation=True)
                
                # Generate summary for this chunk
                with torch.no_grad():
                    output = model.generate(
                        inputs.input_ids,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                
                # Decode summary
                summary = tokenizer.decode(output[0], skip_special_tokens=True)
                summary_parts.append(summary)
            
            # Combine summaries
            final_summary = " ".join(summary_parts)
            
            # Save summary
            with open(output_path, 'w') as f:
                f.write(final_summary)
            
            self.results["output_files"]["summary"] = output_path
            return True
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        try:
            # Step 1: Run independent processes (can be parallelized in the future)
            diarization_success = self.run_with_progress("run_diarization")
            transcription_success = self.run_with_progress("run_whisper")
            
            # Check if both succeeded before alignment
            if not (diarization_success and transcription_success):
                logger.error("Error: Diarization or transcription failed")
                self.results["pipeline_status"] = "failed"
                return self.results
            
            # Step 2: Alignment (depends on step 1)
            alignment_success = self.run_with_progress("run_alignment")
            if not alignment_success:
                logger.error("Error: Alignment failed")
                self.results["pipeline_status"] = "failed"
                return self.results
            
            # Step 3: Run parallel processes (sentiment and summarization)
            sentiment_success = self.run_with_progress("run_sentiment_analysis")
            summary_success = self.run_with_progress("run_summarization")
            
            # Final status
            if sentiment_success and summary_success:
                self.results["pipeline_status"] = "completed"
            else:
                self.results["pipeline_status"] = "partially_completed"
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            self.results["pipeline_status"] = "failed"
            self.results["errors"].append(str(e))
            return self.results
    
    def collect_results(self) -> Dict[str, Any]:
        """Collect and load all result files"""
        results = {
            "pipeline_info": self.results,
            "outputs": {}
        }
        
        # Try to load each output file
        for key, path in self.results.get("output_files", {}).items():
            if self._file_exists(path):
                try:
                    if path.endswith(".json"):
                        with open(path, 'r') as f:
                            results["outputs"][key] = json.load(f)
                    else:
                        with open(path, 'r') as f:
                            results["outputs"][key] = f.read()
                except Exception as e:
                    logger.error(f"Could not load {key} from {path}: {str(e)}")
                    results["outputs"][key] = f"Error loading file: {str(e)}"
        
        return results


# Example configuration
def get_default_config():
    return {
        "audio_path": "/home/amit/april_cohort/audio_file/Narendra Modi on Mahatma Gandhi Lex Fridman Podcast Clips.mp3",
        "output_dir": "/home/amit/april_cohort/testing",
        "model_paths": {
            "whisper": "/home/amit/.cache/whisper/base.pt",
            "indicbert_model": "/home/amit/indicbert_model",
            "indicbert_tokenizer": "/home/amit/indicbert_tokenizer", 
            "t5_model": "/home/amit/T5-fine-tuned-modelC",
            "speechbrain":"/home/amit/april_cohort/pretrained_models/spkrec-ecapa-voxceleb"
        },
        "intermediate_files": {
            "diarization": "diarization_result.json",
            "transcript": "whisper_transcript.txt",
            "aligned": "aligned_transcript.json",
            "indicbert_input": "indicbert_input.json",
            "summary": "summary_output.txt"
        }
    }


def main():
    # Get configuration (can be loaded from a file in production)
    config = get_default_config()
    
    # Initialize pipeline
    pipeline = AudioPipeline(config, force_rerun=False)
    
    # Run pipeline
    results = pipeline.run_pipeline()
    
    # Display results
    logger.info(f"Pipeline status: {results['pipeline_status']}")
    
    if results["pipeline_status"] == "completed":
        logger.info("Pipeline completed successfully!")
        
        # Collect detailed results
        detailed_results = pipeline.collect_results()
        
        # Save final results to a file
        output_path = os.path.join(config["output_dir"], "pipeline_results.json")
        with open(output_path, 'w') as f:
            json.dump(detailed_results["pipeline_info"], f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    else:
        logger.error("Pipeline did not complete successfully")
        if results.get("errors"):
            logger.error("Errors:")
            for error in results["errors"]:
                logger.error(f"  - {error}")


if __name__ == "__main__":
    main()


































