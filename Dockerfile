FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        UVICORN_WORKERS=2

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY pytest.ini .

# Create a non-root runtime user for better container security.
# UID 1000 explicitly, matching Hugging Face Spaces' documented convention
# for Docker-SDK Spaces (also harmless for every other deployment target).
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
