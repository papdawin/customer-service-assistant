FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libsndfile1 espeak-ng-data libespeak-ng1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir fastapi uvicorn piper-tts piper-phonemize

# Hungarian voice (Piper)
RUN mkdir -p /voices \
 && curl -L -o /voices/hu_HU-berta-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx \
 && curl -L -o /voices/hu_HU-berta-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx.json

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 5002
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5002"]
