FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend

WORKDIR /app/backend
CMD sh -c 'uvicorn app:app --host 0.0.0.0 --port ${PORT:-8090}'
