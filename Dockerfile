# Render Free Web Service (Docker runtime) — this is how ffmpeg gets installed.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORK_DIR=/tmp/voice-content-bot

# ffmpeg + ffprobe. ffprobe gives exact durations; ffmpeg does the
# downsampling and chunking of long recordings.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY prompts ./prompts
COPY scripts ./scripts

# Render injects PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Single worker on purpose: the background job queue lives in the process, so a
# second worker would not see jobs submitted to the first.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]
