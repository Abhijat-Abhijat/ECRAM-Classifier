"""
Export the Keras model to ONNX for in-browser inference (static website) and
verify that ONNX output matches Keras on the cached dataset.

    python scripts/export_onnx.py
    # -> web/model/cell_classifier.onnx  (+ web/model/model.json metadata)

Output of the ONNX graph is P(cancerous) directly (1 - keras sigmoid), so the
browser does not need to know the Keras class-index convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from cell_classifier import __version__, config  # noqa: E402
from cell_classifier.data import list_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "web" / "model" / "cell_classifier.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import onnxruntime as ort
    import tf2onnx

    keras_model = tf.keras.models.load_model(args.model, compile=False)

    @tf.function(input_signature=[tf.TensorSpec([None, *config.IMG_SIZE, 3], tf.float32, name="image")])
    def prob_cancerous(image):
        # Keras sigmoid = P(non_cancerous); export P(cancerous) so the web app is convention-free.
        return {"prob_cancerous": 1.0 - keras_model(image, training=False)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(prob_cancerous, input_signature=prob_cancerous.input_signature,
                                  opset=args.opset, output_path=str(args.out))
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")

    # ---- parity check on every cached image ---------------------------------
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    if not (config.PROCESSED_DIR / "cancerous").is_dir():
        print("data/processed missing - skipped parity check (run scripts/prepare_data.py)")
        max_diff = None
    else:
        samples = list_samples(config.RAW_DIR)
        x = np.stack([np.asarray(Image.open(config.PROCESSED_DIR / s.label / f"{s.path.name}.png").convert("RGB"),
                                 dtype=np.float32) / 255.0 for s in samples])
        p_keras = 1.0 - keras_model.predict(x, verbose=0).reshape(-1)
        p_onnx = np.concatenate([sess.run([out_name], {in_name: x[i:i + 32]})[0].reshape(-1)
                                 for i in range(0, len(x), 32)])
        max_diff = float(np.max(np.abs(p_keras - p_onnx)))
        flips = int(np.sum((p_keras >= 0.5) != (p_onnx >= 0.5)))
        print(f"Parity on {len(x)} images: max |P_keras - P_onnx| = {max_diff:.2e}, label flips = {flips}")
        if max_diff > 1e-3 or flips:
            sys.exit("ONNX output does not match Keras - not publishing.")

    meta = {
        "app_version": __version__,
        "source_model": args.model.name,
        # Lets CI / deploy_hf.py detect a stale export (file mtimes are meaningless after git clone)
        "source_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "onnx_file": args.out.name,
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "input": {"name": in_name, "shape": [1, *config.IMG_SIZE, 3], "dtype": "float32",
                  "preprocessing": "RGB, nearest-neighbour resize (Pillow NEAREST sampling), divide by 255"},
        "output": {"name": out_name, "meaning": "P(cancerous)"},
        "threshold": config.CANCER_THRESHOLD,
        "max_abs_diff_vs_keras": max_diff,
    }
    (args.out.parent / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.parent / 'model.json'}")


if __name__ == "__main__":
    main()
