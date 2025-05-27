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
* **Testing Code:** `Ai_model_testing_individually/whisper_testingFinal.py`
* **Input:** 16kHz audio call file \[`Results/english_taking_4_people_5mins_set3.mp3`]
* **Output:** Raw text transcript \[`Results/whisper_transcript(4).txt`]
* **Purpose:** Converts audio into readable text.

### 2. 📝 Formatted Transcript

* Converts the raw transcript into a more human-readable and structured format.
* **Formatted Transcript:** \[`Results/formatted_transcript(6).txt`]

### 3. 🚩 Speaker Diarization

* **Tool Used:** [pyannote-audio](https://github.com/pyannote/pyannote-audio)
* **Testing Code:** `Ai_model_testing_individually/speech_brain_tested.py`
* **Input:** 16kHz audio file \[`Results/english_taking_4_people_5mins_set3.mp3`]
* **Output:** Speaker Diarization JSON \[`Results/diarization_result(5).json`]
* **Purpose:** Identifies "who spoke when" in the conversation.

### 4. 🔗 Aligned Transcript

* Combines:

  * Formatted transcript \[`Results/formatted_transcript(6).txt`]
  * Speaker diarization results \[`Results/diarization_result(5).json`]
* **Output:** Aligned Transcript JSON \[`Results/aligned_transcript(4).json`]
* **Purpose:** Creates a full transcript with speaker attribution.

### 5. 💬 Sentiment Analysis

* **Model Used:** [Indic-BERT](https://huggingface.co/ai4bharat/indic-bert)
* **Testing Code:** `Ai_model_testing_individually/indicBERT_tested.py`
* **Input:** Aligned transcript \[`Results/aligned_transcript(4).json`]
* **Output:** Sentiment Results JSON \[`Results/sentiment_results(4).json`]

  * Sentiment: Positive, Negative, Neutral
  * NPS: Net Promoter Score

### 6. 📟 Summary Generation

* **Model Used:** Fine-tuned [T5 (Text-to-Text Transfer Transformer)](https://huggingface.co/models)
* **Testing Code:** `Ai_model_testing_individually/T5(fine-tuned)_testing_done.py`
* **Fine-Tuning Script:** `googel-T5_fine-tuning/fine_tune_t5C.py`
* **Input:** Sentiment-analyzed transcript
* **Output:** Summary Output \[`Results/summary_output(3).txt`]
* **Purpose:** Quickly helps companies understand the nature and outcome of the call.

---

## 📂 Input Format

* `.wav` or `.mp3` audio file (mono/stereo, resampled to 16kHz)

---

## 📄 Output Files

* `whisper_transcript(4).txt` – Raw transcript from Whisper
* `formatted_transcript(6).txt` – Cleaned and structured text
* `diarization_result(5).json` – Speaker segments
* `aligned_transcript(4).json` – Speaker-labeled full transcript
* `sentiment_results(4).json` – Sentiment analysis with NPS
* `summary_output(3).txt` – T5-generated conversation summary

---

## 🔮 Testing Code

All individual modules are tested and organized under:

**Folder:** `Ai_model_testing_individually/`

Includes:

* Whisper: `whisper_testingFinal.py`
* Diarization: `speech_brain_tested.py`
* Sentiment Analysis: `indicBERT_tested.py`
* Summary: `T5(fine-tuned)_testing_done.py`

---

## 📊 Results

All outputs, including test audio and result files, are available in:

**Folder:** `Results/`

* `english_taking_4_people_5mins_set3.mp3` – Test Audio File
* `whisper_transcript(4).txt`, `formatted_transcript(6).txt`
* `diarization_result(5).json`, `aligned_transcript(4).json`
* `sentiment_results(4).json`, `summary_output(3).txt`

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
