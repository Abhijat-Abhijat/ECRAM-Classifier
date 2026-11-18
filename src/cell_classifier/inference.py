"""Model loading and prediction.

Two interchangeable backends, chosen by the model file's extension:
    .onnx       -> ONNX Runtime (production server: ~60 MB RAM, no TensorFlow needed)
    .h5/.keras  -> TensorFlow/Keras (training, evaluation, local development)
scripts/export_onnx.py guarantees the two agree (max |diff| < 1e-3, no label flips).
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config
from .preprocessing import ImageSource, preprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF info spam


@dataclass
class Prediction:
    label: str                 # "cancerous" | "non_cancerous"
    is_cancerous: bool
    confidence: float          # % confidence in the predicted label (50-100)
    prob_cancerous: float      # % probability of the cancerous class (0-100)
    threshold: float           # % threshold on prob_cancerous used for the decision

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """Thread-safe model wrapper. Load once, reuse for every request."""

    def __init__(self, model_path: str | Path = config.MODEL_PATH,
                 threshold: float = config.CANCER_THRESHOLD):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train one with `python scripts/train.py` "
                f"or set the MODEL_PATH environment variable."
            )
        self.model_path = model_path
        self.threshold = threshold
        self.backend = "onnxruntime" if model_path.suffix.lower() == ".onnx" else "tensorflow"
        self._lock = threading.Lock()
        if self.backend == "onnxruntime":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0"))  # 0 = all cores
            self.model = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
            self._input = self.model.get_inputs()[0].name    # "image"
            self._output = self.model.get_outputs()[0].name  # "prob_cancerous" (already P(cancer))
        else:
            import tensorflow as tf  # imported lazily so unit tests can stub the predictor
            self.model = tf.keras.models.load_model(model_path, compile=False)
        # Warm-up: the first call is slow (graph build / allocation); do it at startup, not on a user.
        self.predict_array(np.zeros((1, *config.IMG_SIZE, 3), dtype=np.float32))

    def predict_array(self, batch: np.ndarray) -> np.ndarray:
        """Return P(cancerous) for a preprocessed batch of shape (N, 224, 224, 3)."""
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        with self._lock:
            if self.backend == "onnxruntime":
                # The exported graph has a fixed batch size of 1 (what the browser uses).
                return np.concatenate([self.model.run([self._output], {self._input: x[None]})[0].reshape(-1)
                                       for x in batch])
            p_non_cancerous = self.model(batch, training=False).numpy().reshape(-1)
        return 1.0 - p_non_cancerous

    def predict(self, source: ImageSource) -> Prediction:
        p_cancer = float(self.predict_array(preprocess(source))[0])
        return self.from_probability(p_cancer)

    def from_probability(self, p_cancer: float) -> Prediction:
        is_cancer = p_cancer >= self.threshold
        confidence = p_cancer if is_cancer else 1.0 - p_cancer
        return Prediction(
            label="cancerous" if is_cancer else "non_cancerous",
            is_cancerous=is_cancer,
            confidence=round(confidence * 100, 1),
            prob_cancerous=round(p_cancer * 100, 1),
            threshold=round(self.threshold * 100, 1),
        )
