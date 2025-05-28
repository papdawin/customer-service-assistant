# 🗣️ LLM Assistant for AI-Powered Customer Service Phone System

This project is an intelligent **AI Customer Service Phone Assistant**, tailored for **Hungarian-language** telephonic interactions. It integrates:

* 🎙️ **Speech-to-Text (STT)** – Fine-tuned Whisper Large
* 🧠 **Large Language Model (LLM)** – currently via OpenAI API
* 🔊 **Text-to-Speech (TTS)** – Fine-tuned XTTS from Coqui

---

## 📁 Project Structure

```
LLM-assistant/
├── app/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py          # Handles OpenAI LLM interactions
│   │   ├── stt.py          # Speech-to-Text processing
│   │   ├── tts.py          # Text-to-Speech processing
│   ├── static/
│   │   ├── script.js       # Web client logic
│   ├── templates/
│   │   ├── index.html      # Web UI
│   │── __init__.py
│   │── config.py           # Configuration setup
│   │── routes.py           # Flask app routes
├── uploads/                # Uploaded audio files (user input)
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile
├── environment.yml
├── main.py                 # Flask app entry
├── requirements.txt
```

---

## 🚀 Getting Started

### 📦 Local installation

```bash
# Using pip
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
conda activate llm-assistant
```

### 🏁 Run the Application

```bash
python main.py
```

Open your browser at [http://localhost:5000](http://localhost:5000)

---

## 🧠 Component Overview

### 🔊 Text-to-Speech (TTS)

Converts AI-generated text into **natural Hungarian speech**.

* ✅ **Currently using:** [Coqui XTTS](https://github.com/coqui-ai/TTS) with a **fine-tuned Hungarian model**
* 📦 Supports expressive multilingual synthesis
* 💡 XTTS enables cross-lingual transfer with high audio quality

**Alternatives explored:**

* F5 TTS – Promising, with a story-telling style, but was abandoned due to slow procesing times
* google-tts – Calls an online API
* facebook/mms-tts-hun - Doesn't sound that good

---

### 🗣️ Speech-to-Text (STT)

Transcribes **Hungarian spoken input** to text.

* ✅ **Currently using:** **Fine-tuned [Whisper Large](https://github.com/openai/whisper)** model for Hungarian
* 📍 Local deployment for real-time performance
* 🎯 Optimized for domain-specific (Hun/Eng, one at a time) customer service vocabulary

**Alternatives explored:**

* Multilanguage whisper – Fast, lightweight, but picks up words from other languages

---

### 🧠 Large Language Model (LLM)

Provides the intelligence and conversation flow.

* ✅ **Currently using: OpenAI API**
* 💬 Handles natural language understanding and response generation
* 🧩 Summarizes answers as to nat take too long on the phone

**Alternatives considered:**

*none - as of yet*

---

## 🔍 Experimentation Summary
| Component | Methodology Explored                                         | Final Approach                      | Rationale                                                                                |
| --------- |--------------------------------------------------------------| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| **TTS**   | Coqui XTTS, F5 TTS, google-tts, facebook/mms-tts-hun         | ✅ **Fine-tuned Coqui XTTS (HU)**    | Best-in-class Hungarian voice quality, expressive, multilingual support                  |
| **STT**   | Whisper Large (multilingual)                                 | ✅ **Fine-tuned Whisper Large (HU)** | High accuracy and real-time performance for Hungarian customer service use               |
| **LLM**   | OpenAI GPT-4  | ✅ **OpenAI GPT-3.5 API**            | Reliable generation, robust dialogue management; future-proof for fallback to local LLMs |

---

## 💡 Potential Future Enhancements

* 📞 Integrate **SIP/VoIP** for real phone calls (e.g. Twilio)
* 🧠 Add **RAG** integration for retrieval-augmented responses
* 📈 Add calendar and other system integration tools, maybe Agent-to-Agent protocol
* 🔐 Move LLM inference to **on-premise or private cloud** and multi-containerize the application
* 🗣️ Support multilingual switching in real-time

---

## 🛡️ Environment & Deployment

### Environment Variables (`.env`)

```
OPENAI_API_KEY=your_openai_api_key
UPLOAD_FOLDER=uploads_folder_location
STATIC_FOLDER=static_folder_location
DEBUG=whether_to_enable_debug_mode
```

### Docker Support

```bash
docker build -t llm-assistant .
docker run -p 5000:5000 llm-assistant
```

or you may pull it from dockerhub, it's publicly available at [pdwn/assistant:latest](https://hub.docker.com/r/pdwn/assistant)

```bash
docker pull pdwn/assistant:latest
```
---

## 📬 Contact

Have questions or suggestions? Open an issue or reach out directly.
