# 📞 Call Analyzer

Call Analyzer is a comprehensive pipeline designed to help companies that offer calling services better understand and evaluate conversations between their agents and customers.

This tool takes in audio or recorded calls between two people and performs a detailed analysis including transcription, speaker diarization, sentiment analysis, and conversation summarization.

---

## 📂 Local Deployment: AI Audio Processing Web App

I have done the local hosting of my AI audio processing web application using **Flask**, in which the backend connects the complete AI audio processing architecture through **REST API** to Flask, and a simple HTML file is used for the frontend.

The application is hosted locally on the server: [http://127.0.0.1:5000](http://127.0.0.1:5000)

My machine setup includes:

* **GPU:** 4GB RTX 3050

This setup is capable of handling the complete AI audio pipeline and conducting deep analysis as described.

I will include architecture diagrams and code images for clarity.

---

## 🧠 System Architecture

![Call Analyzer Flow](Results/final%20map.png)

---

## 🚀 Features & Workflow

### 1. 🎧 Audio to Text Transcription

* **Model Used:** Fine-tuned [Whisper-Base](https://github.com/openai/whisper)
* **Testing Code:** [`whisper_testingFinal.py`](Ai_model_testing_individually/whisper_testingFinal.py)
* **Input:** 16kHz audio call file [`english_taking_4_people_5mins_set3.mp3`](Results/english_taking_4_people_5mins_set3.mp3)
* **Output:** Raw text transcript [`whisper_transcript(4).txt`](Results/whisper_transcript%284%29.txt)
* **Purpose:** Converts audio into readable text.

### 2. 📝 Formatted Transcript

* Converts the raw transcript into a more human-readable and structured format.
* **Formatted Transcript:** [`formatted_transcript(6).txt`](Results/formatted_transcript%286%29.txt)

### 3. 🗣️ Speaker Diarization

* **Tool Used:** [pyannote-audio](https://github.com/pyannote/pyannote-audio)
* **Testing Code:** [`speech_brain_tested.py`](Ai_model_testing_individually/speech_brain_tested.py)
* **Input:** 16kHz audio file [`english_taking_4_people_5mins_set3.mp3`](Results/english_taking_4_people_5mins_set3.mp3)
* **Output:** Speaker Diarization JSON [`diarization_result(5).json`](Results/diarization_result%285%29.json)
* **Purpose:** Identifies "who spoke when" in the conversation.

### 4. 🔗 Aligned Transcript

* Combines:

  * Formatted transcript [`formatted_transcript(6).txt`](Results/formatted_transcript%286%29.txt)
  * Speaker diarization results [`diarization_result(5).json`](Results/diarization_result%285%29.json)
* **Output:** Aligned Transcript JSON [`aligned_transcript(4).json`](Results/aligned_transcript%284%29.json)
* **Purpose:** Creates a full transcript with speaker attribution.

### 5. 💬 Sentiment Analysis

* **Model Used:** [Indic-BERT](https://huggingface.co/ai4bharat/indic-bert)
* **Testing Code:** [`indicBERT_tested.py`](Ai_model_testing_individually/indicBERT_tested.py)
* **Input:** Aligned transcript [`aligned_transcript(4).json`](Results/aligned_transcript%284%29.json)
* **Output:** Sentiment Results JSON [`sentiment_results(4).json`](Results/sentiment_results%284%29.json)

  * Sentiment: Positive, Negative, Neutral
  * NPS: Net Promoter Score

### 6. 🧾 Summary Generation

* **Model Used:** Fine-tuned [T5 (Text-to-Text Transfer Transformer)](https://huggingface.co/models)
* **Testing Code:** [`T5(fine-tuned)_testing_done.py`](Ai_model_testing_individually/T5%28fine-tuned%29_testing_done.py)
* **Fine-Tuning Script:** [`fine_tune_t5C.py`](googel-T5_fine-tuning/fine_tune_t5C.py)
* **Input:** Sentiment-analyzed transcript
* **Output:** Summary Output [`summary_output(3).txt`](Results/summary_output%283%29.txt)
* **Purpose:** Quickly helps companies understand the nature and outcome of the call.

---

## 🗃️ Input Format

* `.wav` or `.mp3` audio file (mono/stereo, resampled to 16kHz)

---

## 📄 Output Files

* [`whisper_transcript(4).txt`](Results/whisper_transcript%284%29.txt) – Raw transcript from Whisper
* [`formatted_transcript(6).txt`](Results/formatted_transcript%286%29.txt) – Cleaned and structured text
* [`diarization_result(5).json`](Results/diarization_result%285%29.json) – Speaker segments
* [`aligned_transcript(4).json`](Results/aligned_transcript%284%29.json) – Speaker-labeled full transcript
* [`sentiment_results(4).json`](Results/sentiment_results%284%29.json) – Sentiment analysis with NPS
* [`summary_output(3).txt`](Results/summary_output%283%29.txt) – T5-generated conversation summary

---

## 🔬 Testing Code

All individual modules are tested and organized under:

**Folder:** [`Ai_model_testing_individually/`](Ai_model_testing_individually/)

Includes:

* Whisper: [`whisper_testingFinal.py`](Ai_model_testing_individually/whisper_testingFinal.py)
* Diarization: [`speech_brain_tested.py`](Ai_model_testing_individually/speech_brain_tested.py)
* Sentiment Analysis: [`indicBERT_tested.py`](Ai_model_testing_individually/indicBERT_tested.py)
* Summary: [`T5(fine-tuned)_testing_done.py`](Ai_model_testing_individually/T5%28fine-tuned%29_testing_done.py)

---

## 📊 Results

All outputs, including test audio and result files, are available in:

**Folder:** [`Results/`](Results/)

* [`english_taking_4_people_5mins_set3.mp3`](Results/english_taking_4_people_5mins_set3.mp3) – Test Audio File
* [`whisper_transcript(4).txt`](Results/whisper_transcript%284%29.txt), [`formatted_transcript(6).txt`](Results/formatted_transcript%286%29.txt)
* [`diarization_result(5).json`](Results/diarization_result%285%29.json), [`aligned_transcript(4).json`](Results/aligned_transcript%284%29.json)
* [`sentiment_results(4).json`](Results/sentiment_results%284%29.json), [`summary_output(3).txt`](Results/summary_output%283%29.txt)

### 📸 Sample Result Preview

![Result Preview](Results/result-screenshot.png)

> This image illustrates the complete flow from speaker diarization to sentiment analysis and summary.

---

## 👨‍💼 Contributing

We welcome PRs and issue submissions. Please create an issue before submitting major changes.

---

## 📜 License

MIT License

---

## 🧠 Authors

* \[Your Name or Team Name]
* \[Email or Contact Info]
