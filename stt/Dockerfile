FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System deps (ffmpeg for decoding webm/ogg/wav)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
EXPOSE 5001

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5001"]
