FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

# EXPOSE is only documentation and can't be dynamic; you can omit it or leave a default.
# EXPOSE 8090

# Use shell so ${PORT} from docker-compose is honored at runtime.
CMD sh -c 'uvicorn app:app --host 0.0.0.0 --port ${PORT:-8090}'