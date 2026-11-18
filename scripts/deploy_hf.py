"""
Deploy to Hugging Face Spaces - no git needed.

DEFAULT (free for every account): a *Static* Space serving web/ - the model runs in
the visitor's browser (ONNX Runtime Web), images never leave their device.

    pip install huggingface_hub
    hf auth login                                   # paste a token with WRITE access
    python scripts/export_onnx.py                   # only after changing models/cell_classifier.h5
    python scripts/deploy_hf.py --space cell-classifier
    #   -> https://<username>-cell-classifier.static.hf.space

OPTIONAL: --docker deploys the FastAPI server instead. Hugging Face requires a paid
plan (PRO / Team / Enterprise) to create Docker Spaces.

Re-run the same command to redeploy; only changed files are uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

STATIC_README = """---
title: Cell Classifier
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Cancerous vs non-cancerous slide classifier, runs in-browser
---

# Cell Classifier (research prototype)

Classifies a microscopy slide image as cancerous or non-cancerous. The MobileNetV2 model
runs **entirely in your browser** with ONNX Runtime Web, so images are never uploaded.

**Not a medical device. Not a diagnosis.** On unseen cases the current model performs
close to chance; see the project's `reports/evaluation.md`.
"""

DOCKER_README = (STATIC_README
                 .replace("sdk: static\napp_file: index.html", "sdk: docker\napp_port: 7860")
                 .replace("slide classifier, runs in-browser", "slide classifier, FastAPI server")
                 .replace("runs **entirely in your browser** with ONNX Runtime Web, so images are never uploaded.",
                          "runs on the server (FastAPI + ONNX Runtime). Uploaded images are processed in memory and not stored."))

DOCKER_ALLOW = ["Dockerfile", ".dockerignore", ".gitattributes", "requirements.txt",
                "src/**", "web/model/cell_classifier.onnx"]
STATIC_ALLOW = ["index.html", "app.js", "imaging.js", "styles.css", "manifest.webmanifest",
                "icon.svg", "icon-192.png", "icon-512.png", "model/**", "vendor/**"]
IGNORE = ["**/__pycache__/**", "**/*.pyc"]


def check_static() -> None:
    meta_path = WEB / "model" / "model.json"
    if not meta_path.exists():
        sys.exit("web/model/model.json missing - run: python scripts/export_onnx.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    onnx = WEB / "model" / meta["onnx_file"]
    if not onnx.exists():
        sys.exit(f"{onnx} missing - run: python scripts/export_onnx.py")
    h5 = ROOT / "models" / meta["source_model"]
    if h5.exists() and meta.get("source_sha256") and \
            hashlib.sha256(h5.read_bytes()).hexdigest() != meta["source_sha256"]:
        sys.exit(f"{h5.name} changed since {onnx.name} was exported - re-run: python scripts/export_onnx.py")
    for f in ("ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"):
        if not (WEB / "vendor" / "ort" / f).exists():
            sys.exit(f"web/vendor/ort/{f} missing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="Space name, e.g. cell-classifier (or user/name)")
    ap.add_argument("--docker", action="store_true", help="deploy the FastAPI Docker image (paid HF plan required)")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--token", default=None, help="HF token (default: from `hf auth login` or HF_TOKEN)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install it first:  pip install huggingface_hub")

    api = HfApi(token=args.token)
    user = api.whoami()["name"]
    repo_id = args.space if "/" in args.space else f"{user}/{args.space}"
    sub = repo_id.replace("/", "-").replace("_", "-").replace(".", "-").lower()

    if args.docker:
        check_static()  # the Docker image serves web/model/cell_classifier.onnx
        print(f"Creating/updating Docker Space {repo_id} (requires a paid HF plan) ...")
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        api.upload_file(path_or_fileobj=DOCKER_README.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space", commit_message="Space card")
        api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space",
                          allow_patterns=DOCKER_ALLOW, ignore_patterns=IGNORE,
                          commit_message="Deploy Cell Classifier (Docker)")
        url = f"https://{sub}.hf.space"
    else:
        check_static()
        print(f"Creating/updating Static Space {repo_id} (free) ...")
        api.create_repo(repo_id, repo_type="space", space_sdk="static", private=args.private, exist_ok=True)
        api.upload_file(path_or_fileobj=STATIC_README.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space", commit_message="Space card")
        api.upload_folder(folder_path=str(WEB), repo_id=repo_id, repo_type="space",
                          allow_patterns=STATIC_ALLOW, ignore_patterns=IGNORE,
                          commit_message="Deploy Cell Classifier (static, in-browser)")
        url = f"https://{sub}.static.hf.space"

    print(f"\nSpace page : https://huggingface.co/spaces/{repo_id}")
    print(f"App URL    : {url}")


if __name__ == "__main__":
    main()
