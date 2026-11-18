"""Integration tests against the real model files (each backend skips if its file/library is missing).

    tensorflow  : models/cell_classifier.h5            (training / evaluation / local dev)
    onnxruntime : web/model/cell_classifier.onnx       (production server + browser)
"""

import importlib.util

import numpy as np
import pytest

from cell_classifier import config

pytestmark = pytest.mark.slow

H5 = config.PROJECT_ROOT / "models" / "cell_classifier.h5"
BACKENDS = {
    "tensorflow": (H5, "tensorflow"),
    "onnxruntime": (config.ONNX_MODEL_PATH, "onnxruntime"),
}


def _require(backend):
    path, module = BACKENDS[backend]
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    return path


_cache = {}


@pytest.fixture(params=list(BACKENDS), scope="module")
def predictor(request):
    path = _require(request.param)
    if request.param not in _cache:
        from cell_classifier.inference import Predictor
        _cache[request.param] = Predictor(path)
    return _cache[request.param]


def test_backend_selected_by_extension(predictor):
    assert predictor.backend == ("onnxruntime" if predictor.model_path.suffix == ".onnx" else "tensorflow")


def test_tensorflow_model_contract():
    from cell_classifier.inference import Predictor
    p = Predictor(_require("tensorflow"))
    assert p.model.input_shape == (None, 224, 224, 3)
    assert p.model.output_shape == (None, 1)


def test_outputs_are_probabilities_and_deterministic(predictor):
    x = np.random.default_rng(0).random((4, 224, 224, 3), dtype=np.float32)
    p1, p2 = predictor.predict_array(x), predictor.predict_array(x)
    assert p1.shape == (4,)
    assert ((0 <= p1) & (p1 <= 1)).all()
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_backends_agree():
    """The server (ONNX) must give the same answers as the evaluated Keras model."""
    from cell_classifier.inference import Predictor
    tf_p, ort_p = Predictor(_require("tensorflow")), Predictor(_require("onnxruntime"))
    x = np.random.default_rng(1).random((6, 224, 224, 3), dtype=np.float32)
    np.testing.assert_allclose(tf_p.predict_array(x), ort_p.predict_array(x), atol=1e-3)


def test_prediction_fields_consistent(predictor, png_bytes):
    r = predictor.predict(png_bytes)
    assert r.label in ("cancerous", "non_cancerous")
    assert r.is_cancerous == (r.prob_cancerous >= r.threshold)
    assert 50.0 <= r.confidence <= 100.0


def test_end_to_end_api_with_real_model(predictor, png_bytes):
    from fastapi.testclient import TestClient
    from cell_classifier.api.main import create_app
    with TestClient(create_app(lambda: predictor)) as c:
        r = c.post("/api/predict", files={"file": ("x.png", png_bytes, "image/png")})
        info = c.get("/api/info").json()
    assert r.status_code == 200 and r.json()["label"] in ("cancerous", "non_cancerous")
    assert info["model"] == predictor.model_path.name


def test_reproduces_recorded_predictions(predictor):
    """Golden test: predictions must match reports/predictions.csv (written by scripts/evaluate.py).

    Catches preprocessing changes, TensorFlow/Keras version drift and a stale ONNX export - the
    main ways a deployed model silently starts giving different answers than the one you evaluated.
    """
    import csv
    from PIL import Image
    golden = config.REPORTS_DIR / "predictions.csv"
    if not golden.exists() or not (config.PROCESSED_DIR / "cancerous").is_dir():
        pytest.skip("run scripts/prepare_data.py and scripts/evaluate.py first")
    rows = list(csv.DictReader(open(golden, encoding="utf-8")))
    rows = rows[:: max(1, len(rows) // 40)]  # ~40 images spread over both classes
    x = np.stack([np.asarray(Image.open(config.PROCESSED_DIR / f"{r['file']}.png").convert("RGB"),
                             dtype=np.float32) / 255.0 for r in rows])
    p = predictor.predict_array(x)
    np.testing.assert_allclose(p, [float(r["prob_cancerous"]) for r in rows], atol=2e-3)
    acc = np.mean([(pi >= config.CANCER_THRESHOLD) == (r["true_label"] == "cancerous") for pi, r in zip(p, rows)])
    assert acc >= 0.70, f"accuracy on sampled (mostly training) images dropped to {acc:.2f}"
