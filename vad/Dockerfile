FROM python:3.11-slim

WORKDIR /app

# webrtcvad needs a compiler toolchain
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001", "--no-access-log"]
