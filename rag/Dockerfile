FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# App deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py /app/app.py

# Create non-root user, create dirs as root, then chown, THEN drop privileges
RUN useradd -m appuser \
 && mkdir -p /app /app/data /app/indices \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
