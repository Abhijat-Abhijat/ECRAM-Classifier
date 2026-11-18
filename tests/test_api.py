"""API tests with a stub predictor (fast, no TensorFlow needed)."""

import pytest
from fastapi.testclient import TestClient

from cell_classifier import config
from cell_classifier.api.main import create_app
from cell_classifier.inference import Prediction


class StubPredictor:
    model_path = "stub.h5"
    threshold = 0.5

    def __init__(self, p_cancer=0.8):
        self.p = p_cancer

    def predict(self, source):
        from cell_classifier.preprocessing import preprocess
        preprocess(source)  # real decoding/validation path
        c = self.p >= self.threshold
        return Prediction("cancerous" if c else "non_cancerous", c,
                          round((self.p if c else 1 - self.p) * 100, 1), round(self.p * 100, 1), 50.0)


@pytest.fixture
def client():
    with TestClient(create_app(lambda: StubPredictor())) as c:
        yield c


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Cell Classifier" in r.text
    assert r.headers["x-content-type-options"] == "nosniff"


def test_static_assets(client):
    assert client.get("/manifest.webmanifest").status_code == 200
    for f in ("icon.svg", "icon-192.png", "icon-512.png"):
        assert client.get(f"/static/{f}").status_code == 200


def test_health_and_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/api/info").json()
    assert info["classes"] == ["cancerous", "non_cancerous"] and info["max_upload_mb"] == config.MAX_UPLOAD_MB


@pytest.mark.parametrize("path", ["/api/predict", "/predict"])
def test_predict_ok(client, png_bytes, path):
    r = client.post(path, files={"file": ("slide.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "cancerous" and body["is_cancerous"] is True
    assert body["prob_cancerous"] == 80.0 and body["confidence"] == 80.0
    assert "processing_ms" in body


def test_tif_with_octet_stream_content_type_accepted(client):
    from conftest import make_image_bytes
    r = client.post("/api/predict", files={"file": ("s.tif", make_image_bytes(fmt="TIFF"), "application/octet-stream")})
    assert r.status_code == 200


def test_non_image_rejected(client):
    r = client.post("/api/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "error" in r.json()


def test_corrupt_image_rejected(client):
    r = client.post("/api/predict", files={"file": ("x.png", b"\x89PNG broken", "image/png")})
    assert r.status_code == 400 and "Couldn't read" in r.json()["error"]


def test_empty_file_rejected(client):
    assert client.post("/api/predict", files={"file": ("x.png", b"", "image/png")}).status_code == 400


def test_upload_size_limit(client, png_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    assert client.post("/api/predict", files={"file": ("x.png", png_bytes, "image/png")}).status_code == 413


def test_pixel_limit(client, png_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 10)
    r = client.post("/api/predict", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 413 and "MP" in r.json()["error"]


def test_missing_file_field(client):
    assert client.post("/api/predict").status_code == 422
