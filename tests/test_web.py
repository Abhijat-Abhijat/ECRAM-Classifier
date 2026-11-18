"""Checks for the static, in-browser web app (web/)."""

import hashlib
import json
import random
import re

import numpy as np
import pytest
from PIL import Image

from cell_classifier import config

WEB = config.PROJECT_ROOT / "web"


def accumulated_nearest(n: int, out: int = 224) -> np.ndarray:
    """Python mirror of nearestIndices() in web/imaging.js."""
    a, xo, idx = n / out, n / out * 0.5, []
    for _ in range(out):
        idx.append(min(n - 1, int(xo)))
        xo += a
    return np.array(idx)


@pytest.mark.parametrize("n", [224, 300, 581, 1350, 2048, 2880, 11090, 25875, 42242] +
                         random.Random(0).sample(range(100, 60000), 40))
def test_js_nearest_formula_matches_pillow(n):
    line = Image.fromarray(np.arange(n, dtype=np.int32).reshape(1, -1), "I")
    got = np.asarray(line.resize((224, 1), Image.Resampling.NEAREST)).reshape(-1)
    np.testing.assert_array_equal(got, accumulated_nearest(n))


def test_imaging_js_uses_the_same_formula():
    js = (WEB / "imaging.js").read_text(encoding="utf-8")
    assert re.search(r"let xo = a \* 0\.5;", js) and "xo += a" in js and "Math.trunc(xo)" in js


def test_static_bundle_complete_and_consistent():
    for f in ("index.html", "app.js", "imaging.js", "styles.css", "manifest.webmanifest",
              "icon.svg", "icon-192.png", "icon-512.png",
              "vendor/ort/ort.wasm.min.js", "vendor/ort/ort-wasm-simd-threaded.wasm",
              "vendor/ort/ort-wasm-simd-threaded.mjs", "model/model.json"):
        assert (WEB / f).is_file(), f"missing web/{f}"
    meta = json.loads((WEB / "model" / "model.json").read_text(encoding="utf-8"))
    onnx = WEB / "model" / meta["onnx_file"]
    assert hashlib.sha256(onnx.read_bytes()).hexdigest() == meta["sha256"], "model.json sha256 out of date"
    assert meta["output"]["meaning"] == "P(cancerous)"
    h5 = config.PROJECT_ROOT / "models" / meta["source_model"]
    if h5.exists():  # stale export = browser and server would run an older model than the .h5
        assert hashlib.sha256(h5.read_bytes()).hexdigest() == meta["source_sha256"], \
            "models/*.h5 changed - re-run: python scripts/export_onnx.py"
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for src in re.findall(r'(?:src|href)="([^":]+?)"', html):   # every local reference resolves
        assert (WEB / src).exists(), f"index.html references missing {src}"


@pytest.mark.slow
def test_onnx_matches_keras():
    ort = pytest.importorskip("onnxruntime")
    import tensorflow as tf
    meta = json.loads((WEB / "model" / "model.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(WEB / "model" / meta["onnx_file"]), providers=["CPUExecutionProvider"])
    keras = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    x = np.random.default_rng(3).random((8, 224, 224, 3), dtype=np.float32)
    p_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
    p_keras = 1.0 - keras.predict(x, verbose=0).reshape(-1)
    np.testing.assert_allclose(p_onnx, p_keras, atol=1e-4)
