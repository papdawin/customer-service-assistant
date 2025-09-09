# 🎙️ Hungarian Voice Assistant (RAG + STT + TTS + LLM + Web)

An AI-powered **voice assistant application** designed for Hungarian-language customer service.
It leverages **Speech-to-Text (STT)**, **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLM)**, and **Text-to-Speech (TTS)** to deliver real-time, interactive voice conversations through a browser-based interface.


## ⚙️ Microservices

* **LLM:** [vLLM](https://github.com/vllm-project/vllm) serving Google Gemma-3-27B-IT
* **STT:** `sarpba/faster-base-hungarian_int8_V2` (Whisper-finetune)
* **TTS:** [Piper](https://github.com/rhasspy/piper) with Hungarian voice (`hu_HU-berta-medium`)
* **RAG:** Custom FastAPI app with vector indices per tenant
* **Frontend:** Lightweight web interface (JS/HTML)

![Architecture diagram](https://raw.githubusercontent.com/papdawin/customer-service-assistant/refs/heads/master/pictures/diagram.png)

Other info

* **Containerization:** Docker + Docker Compose
* **GPU Acceleration:** CUDA-enabled inference for LLM/STT/TTS



## 📁 Project Structure

```bash
.
├── rag/                       # Retrieval-Augmented Generation service
│   ├── app.py 
│   ├── requirements.txt
│   └── Dockerfile
├── stt/                       # Speech-to-Text microservice
│   ├── app.py 
│   ├── requirements.txt
│   └── Dockerfile
├── tts/                       # Text-to-Speech microservice
│   ├── app.py
│   └── Dockerfile
├── web/                       # Web UI for voice assistant
│   ├── app.py 
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── microsoft/             # Microsoft tenant documents
│   └── sonrisa/               # Sonrisa tenant documents
├── Dockerfile.vllm            # vLLM inference container
├── docker-compose.yml         # Main compose file
└── README.md                  # This documentation
```

## 🚀 Getting Started

### 1. Add your documents

The RAG system expects txt files to be present in the company's directory

### 2. Start with Docker Compose

```bash
docker compose up --build -d
```

This will spin up:

* **vLLM** on `localhost:8000`
* **RAG Services** (Example Company: `8101`, Test Company: `8102`)
* **STT** on `5001`
* **TTS** on `5002`
* **Web Interfaces:**

  * Example Company: [http://localhost:8090](http://localhost:8090)
  * Test Company: [http://localhost:8091](http://localhost:8091)

## ⚙️ Extension

You may extend the services and provide it to another company, by copying the template (RAG and web containers) and creating other document holder files in the data folder.

## 🧠 Component Overview

### 🔊 STT (Speech-to-Text)

* **Model:** `sarpba/faster-base-hungarian_int8_V2`
* **Reasoning:** Lightweight quantized model for fast, accurate Hungarian transcription.
* **Alternatives Considered:** OpenAI Whisper (slower), Coqui STT (lower accuracy)

### 🧾 RAG (Retrieval-Augmented Generation)

* **Approach:** FAISS-based document index per tenant
* **Strength:** Ensures company-specific answers (Example company / Test company)

### 🧠 LLM

* **Model:** `google/gemma-3-27b-it` served with vLLM
* **Precision:** bfloat16
* **Parallelism:** 4-way tensor parallelism for multi-GPU scaling, the machine I was testing the solution on had 4 GPUs

### 🗣️ TTS (Text-to-Speech)

* **Engine:** Piper
* **Voice:** `hu_HU-berta-medium.onnx`

## 🔍 Experimentation Summary

| Component | Options Explored                     | Final Choice          | Reason                               |
| --------- | ------------------------------------ | --------------------- | ------------------------------------ |
| STT       | Whisper large, faster-whisper, Coqui | sarpba's Whisper fine-tune | Best speed      |
| LLM       | GPT-4, Mistral, Gemma                | Gemma-3-27B-IT        | Best Hungarian support + self-hosted |
| TTS       | gTTS, Piper, Coqui                   | Piper (Berta voice)   | Good quality and speed      |


## 🔐 Security & Privacy Considerations

* **API Keys:** Loaded via `.env` or Docker Compose `environment`.
* **Audio Data:** Processed in-memory, not persisted.
* **Multi-Tenant Isolation:** Separate RAG indices for The companies, which adds a layer of separation, and caompanies can use the same LLM securely