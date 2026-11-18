"""
Central configuration.

Every value that differs between laptop / Docker / cloud can be overridden with
an environment variable, so the same code runs everywhere unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels above this file (src/cell_classifier/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# .h5/.keras -> TensorFlow backend (development); .onnx -> ONNX Runtime backend (the
# production Docker image sets MODEL_PATH to the ONNX file - no TensorFlow needed).
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "cell_classifier.h5"))
ONNX_MODEL_PATH = PROJECT_ROOT / "web" / "model" / "cell_classifier.onnx"  # written by scripts/export_onnx.py
IMG_SIZE: tuple[int, int] = (224, 224)

# Keras flow_from_directory assigns indices alphabetically by folder name:
#   {'cancerous': 0, 'non_cancerous': 1}
# The sigmoid output is therefore P(non_cancerous). Do NOT change unless you
# retrain with different folder names.
CLASS_NAMES: dict[int, str] = {0: "cancerous", 1: "non_cancerous"}

# Decision threshold on P(cancerous). 0.5 = the model's natural boundary.
# Lowering it (e.g. 0.3) flags more slides -> higher sensitivity, more false alarms.
CANCER_THRESHOLD = float(os.getenv("CANCER_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"              # original slides (cancerous/, non_cancerous/)
PROCESSED_DIR = DATA_DIR / "processed"  # 224x224 cache produced by scripts/prepare_data.py
REPORTS_DIR = PROJECT_ROOT / "reports"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# API limits (protect the server from huge uploads / decompression bombs)
# ---------------------------------------------------------------------------
# Some slides in this dataset are whole-slide TIFFs of ~1.1 gigapixels, which
# need ~3.3 GB RAM to decode. Defaults below accept every image in the dataset
# on a 16 GB host (Hugging Face Spaces free tier). Lower them on small hosts.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(1_200_000_000)))

# Comma-separated list of allowed browser origins. "*" is fine because the UI
# is served from the same origin; restrict it if you host the UI elsewhere.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
