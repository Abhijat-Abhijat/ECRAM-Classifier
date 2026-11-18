# Production image for the Cell Classifier backend (FastAPI + ONNX Runtime, no TensorFlow).
# Works on Render (free web service), Hugging Face Docker Spaces, and any Docker host.
#
#   docker build -t cell-classifier .
#   docker run --rm -p 7860:7860 cell-classifier      -> http://localhost:7860
#
# The host may set PORT (Render sets 10000); the server listens on whatever PORT is.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/cell_classifier.onnx \
    PORT=7860

# Non-root user (uid 1000 is also what Hugging Face Spaces expects)
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Dependencies first -> cached layer, rebuilt only when requirements change
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user src/ src/
# Same file the browser app uses (scripts/export_onnx.py verifies it against the Keras model)
COPY --chown=user web/model/cell_classifier.onnx models/cell_classifier.onnx

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health', timeout=4)"

# One worker: the model is loaded once (~150 MB RAM); requests run in a thread pool.
CMD ["sh", "-c", "exec uvicorn --app-dir src cell_classifier.api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
