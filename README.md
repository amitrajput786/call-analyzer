# call-analyzer
Our solution processes call audio through an AI pipeline that transcribes English and Hindi conversations, identifies speakers, and analyzes key metrics like sentiment, hold time, and overtalk. The modular architecture uses fine-tuned open-source models for accuracy and easy scalability to other Indian languages, 

# 📞 Call Analyzer

Call Analyzer is a comprehensive pipeline designed to help companies that offer calling services better understand and evaluate conversations between their agents and customers.

This tool takes in audio or recorded calls between two people, and performs a detailed analysis including transcription, speaker diarization, sentiment analysis, and conversation summarization.

---

## 🧠 System Architecture

![Call Analyzer Flow](./final%20map.png)

---

## 🚀 Features & Workflow

### 1. 🎧 Audio to Text Transcription
- **Model Used:** Fine-tuned [Whisper-Base](https://github.com/openai/whisper)
- **Input:** 16kHz audio call file
- **Output:** Raw text transcript
- **Purpose:** Converts audio into readable text.

### 2. 📝 Formatted Transcript
- Converts the raw transcript into a more human-readable and structured format.

### 3. 🗣️ Speaker Diarization
- **Tool Used:** [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Input:** Same 16kHz audio file
- **Output:** JSON file with speaker segments and timestamps
- **Purpose:** Identifies “who spoke when” in the conversation.

### 4. 🔗 Aligned Transcript
- Combines:
  - Formatted transcript
  - Speaker diarization results
- **Output:** Aligned JSON file
- **Purpose:** Creates a full transcript with speaker attribution.

### 5. 💬 Sentiment Analysis
- **Model Used:** [Indic-BERT](https://huggingface.co/ai4bharat/indic-bert)
- **Input:** Aligned speaker-labeled transcript
- **Output:** JSON file including:
  - Sentiment (Positive, Negative, Neutral)
  - NPS (Net Promoter Score)
- **Purpose:** Measures speaker sentiments throughout the call.

### 6. 🧾 Summary Generation
- **Model Used:** Fine-tuned [T5 (Text-to-Text Transfer Transformer)](https://huggingface.co/models)
- **Input:** Sentiment-analyzed transcript
- **Output:** Short summary of the full call
- **Purpose:** Quickly helps companies understand the nature and outcome of the call.

---

## 📂 Input Format

- `.wav` or `.mp3` audio file (mono/stereo, resampled to 16kHz)

---

## 📤 Output Files

- `raw_transcript.txt` – Raw text from Whisper
- `formatted_transcript.txt` – Human-readable format
- `speaker_segments.json` – Speaker diarization output
- `aligned_transcript.json` – Merged transcript with speaker labels
- `sentiment_analysis.json` – Sentiment result with scores
- `call_summary.txt` – T5-generated call summary

---

## 🧪 Use Case

This tool is ideal for:

- Call center quality monitoring
- Customer support analytics
- Sales call evaluation
- Market research on consumer-agent interactions

---

## 🛠️ Setup Instructions

> Note: Please ensure you have Python 3.8+ and necessary GPU/CPU dependencies installed.

```bash
# Clone the repository
git clone https://github.com/your-org/call-analyzer.git
cd call-analyzer

# Install requirements
pip install -r requirements.txt

# Download models as per instruction in scripts/
