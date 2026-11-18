# Cell Classifier

Classifies microscopy slide images as **cancerous** or **non-cancerous** with a MobileNetV2 transfer-learning model. The production app is a **static website**: the model runs **inside the visitor's browser** (ONNX Runtime Web), works on phones and laptops, costs nothing to host, and never uploads images. A FastAPI server is included for API and self-hosted use.

> ⚠️ **Research prototype, not a diagnosis.** The current model **does not generalise to unseen cases** (see [Model status](#model-status)). Do not use it for any clinical decision.

* **GitHub + automatic free deploys (UI → GitHub Pages, API → Render):** [docs/GITHUB_DEPLOY.md](docs/GITHUB_DEPLOY.md)
* **Other free hosts:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) (Hugging Face Static Space / Netlify / Cloudflare Pages)
* **Model test results:** [reports/evaluation.md](reports/evaluation.md), [reports/cv_results.json](reports/cv_results.json)
* **Browser vs Python parity:** [reports/browser_parity.json](reports/browser_parity.json)

---

## Project structure

```
cancer_project/
├── README.md
├── pyproject.toml            ← project metadata + pytest config
├── requirements.txt          ← pinned API-server deps (ONNX Runtime, no TensorFlow) for the Docker image
├── requirements-dev.txt      ← + TensorFlow, training, testing, ONNX export, deploy tools
├── Dockerfile  .dockerignore ← API server image (~130 MB RAM, fits Render's free tier)
├── render.yaml               ← Render Blueprint for the free backend
├── .gitignore  .gitattributes (line endings; binaries are plain git objects, no LFS)
│
├── web/                      ← ★ PRODUCTION APP: static site, deploy this folder
│   ├── index.html  styles.css  app.js     ← responsive UI (mobile + desktop, camera, PWA)
│   ├── imaging.js            ← in-browser image reader: Pillow-identical NEAREST sampling,
│   │                           streaming TIFF reader (JPEG/LZW/Deflate/PackBits, tiles, BigTIFF)
│   ├── model/                ← cell_classifier.onnx (9.6 MB) + model.json (made by export_onnx.py)
│   ├── vendor/ort/           ← ONNX Runtime Web 1.31.1 (pinned, hash-verified)
│   └── manifest.webmanifest, icons
│
├── src/cell_classifier/      ← Python package (training/evaluation + optional API server)
│   ├── config.py             ← every setting; env-var overridable
│   ├── preprocessing.py      ← reference preprocessing (the web app reproduces it exactly)
│   ├── inference.py          ← Predictor (.onnx → ONNX Runtime, .h5 → TensorFlow; thread-safe, warm-up)
│   ├── data.py               ← dataset indexing, case-level grouping, data-quality checks
│   └── api/                  ← FastAPI app (/, /health, /api/info, /api/predict) + its UI
│
├── scripts/
│   ├── prepare_data.py       ← decode raw slides once → 224 px cache (data/processed)
│   ├── train.py              ← leakage-safe training; --cv for 5-fold case-grouped CV
│   ├── evaluate.py           ← metrics, CIs, threshold sweep → reports/
│   ├── export_onnx.py        ← .h5 → web/model/*.onnx, with Keras parity check
│   └── deploy_hf.py          ← one-command deploy to a free Hugging Face Static Space
│
├── models/cell_classifier.h5 ← source model (23.6 MB)
├── data/                     ← NOT committed (git-ignored)
│   ├── raw/{cancerous,non_cancerous}/       ← original slides (4.2 GB)
│   └── processed/{cancerous,non_cancerous}/ ← 224×224 cache made by prepare_data.py
├── reports/                  ← evaluation.md/.json, predictions.csv (golden), cv_results.json, browser_parity.json
├── tests/                    ← pytest (preprocessing, data, API, model, web bundle)
│   └── web/                  ← real-browser parity test (headless Edge/Chrome)
├── docs/GITHUB_DEPLOY.md     ← GitHub → Pages (UI) + Render (API) auto-deploy, step by step
├── docs/DEPLOYMENT.md        ← end-to-end free deployment guide (other hosts)
└── .github/workflows/ci.yml  ← tests + browser parity + Docker smoke test, then deploys on push to main
```

---

## Quick start (Windows, PowerShell)

Requires **Python 3.12** for the Python tooling (TensorFlow has no stable wheels for 3.14 yet). The web app itself needs no Python. Full details: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

```powershell
# Run the production web app locally (any Python works for this)
python -m http.server 8080 --directory web
# laptop: http://127.0.0.1:8080     phone (same Wi-Fi): http://<laptop-IPv4>:8080

# Developer setup
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -m "not slow"                 # fast tests
python tests/web/browser_parity.py             # real-browser parity test

# Deploy (free): see docs/DEPLOYMENT.md
hf auth login
python scripts/deploy_hf.py --space cell-classifier
```

macOS/Linux: use `python3.12 -m venv .venv && source .venv/bin/activate`. Everything else is the same.

### Model workflow

```powershell
python scripts/prepare_data.py                 # once, ~10 min: caches 224px copies of data/raw
python scripts/evaluate.py                     # test the production model → reports/
python scripts/train.py --cv                   # estimate how well the training recipe generalises
python scripts/train.py                        # train a candidate → models/cell_classifier_candidate.h5
python scripts/evaluate.py --model models/cell_classifier_candidate.h5
python scripts/export_onnx.py                  # after promoting a new .h5: refresh the web model
```

---

## API (optional server)

Run with `uvicorn --app-dir src cell_classifier.api.main:app --port 8000` or the Dockerfile.

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | `{"status":"ok","model_loaded":true,"version":"1.0.0","commit":"<git sha on Render>"}` |
| GET | `/api/info` | Model name, classes, threshold, upload limits |
| POST | `/api/predict` | multipart field `file` (JPG/PNG/TIF/BMP) → prediction |
| POST | `/predict` | Alias of `/api/predict` (backward compatible) |
| GET | `/docs` | Interactive OpenAPI docs |

```bash
curl -F "file=@slide.tif" http://127.0.0.1:8000/api/predict
```
```json
{"label":"cancerous","is_cancerous":true,"confidence":79.1,"prob_cancerous":79.1,"threshold":50.0,"processing_ms":1834}
```

* `prob_cancerous`: model probability (%) that the slide is cancerous.
* `confidence`: probability of the predicted label (always ≥ 50%).
* Errors return `{"error": "..."}` with status 400 (not an image / unreadable / empty), 413 (too large) or 422 (missing `file`).

---

## Model status

**Architecture:** MobileNetV2 (ImageNet) + GAP + Dense(128) + Dropout + sigmoid. Input is RGB, **nearest-neighbour** resized to 224×224, scaled to [0, 1]. Output = P(`non_cancerous`), and the API reports P(cancerous) = 1 − output.

### Test results for `models/cell_classifier.h5` (full details: [reports/evaluation.md](reports/evaluation.md))

| Test set | Images | Accuracy | ROC-AUC |
|---|---|---|---|
| **Held-out, unseen cases only** (honest) | 23 | **43.5%** (95% CI 21.7–65.2%) | 0.667 (CI 0.41–0.89) |
| Original Keras validation split | 47 | 51.1% | 0.565 |
| All images (~80% seen in training) | 237 | 78.1% | 0.863 |

**Verdict:** on cases it hasn't seen, the model performs **no better than chance**. The 78% figure comes from memorising training images.

### Why: data findings from testing

1. **237 files are only 180 slides from 85 cases.** 56 slides exist twice (`.png` + `.tif`), and most cases have 2–4 fields of view (`EC 90 10X`, `EC 90 10X 2`, …).
2. **30 case numbers have slides in both classes** (e.g. `EC 90` and `EN 90`, likely tumour and normal tissue of the same patient). A random file split puts the same patient in train and validation.
3. **Label problems:** `EN 257 23 10X 2.tif` is byte-identical in *both* class folders. `EN 133 10X.tif` (normal code) sits in `cancerous/`, and `EC254`, `EC329` and `ec92` (cancer codes) sit in `non_cancerous/`. These need confirming with the data owner.
4. **Train/serve skew in the original app:** training resized with *nearest*, but the old API used Pillow's default *bicubic*. On the same images that flips **49 of 237 predictions**. Fixed: the API now uses the exact training preprocessing, and a test asserts it is bit-identical to the Keras loader.
5. The original training applied augmentation to the validation set too, so the accuracy printed during training was unreliable.

### Retrained pipeline (`scripts/train.py`)

Splits by case, drops the conflicting file, doesn't augment validation, uses class weights, rescales MobileNet input correctly, and selects on AUC. 5-fold case-grouped cross-validation result:

| Fold | Held-out images (cases) | Accuracy | ROC-AUC |
|---|---|---|---|
| 1 | 44 (15) | 63.6% | 0.654 |
| 2 | 48 (17) | 56.3% | 0.760 |
| 3 | 49 (18) | 71.4% | 0.827 |
| 4 | 48 (19) | 77.1% | 0.836 |
| 5 | 46 (16) | 78.3% | 0.845 |
| **Mean ± SD** | 235 (85) | **69.3% ± 8.3%** | **0.784 ± 0.072** |

Every case is held out exactly once, so the whole dataset is evaluated on unseen cases. The fixed pipeline clearly learns *some* signal (AUC 0.78 vs. 0.57–0.67 for the original model), but it is not clinically usable: roughly 1 in 3 slides is still misclassified. These figures are slightly optimistic, because the best epoch was chosen on the held-out fold (there is no separate test set at this data size). Raw numbers: [reports/cv_results.json](reports/cv_results.json).

To produce a deployable model from this recipe: `python scripts/train.py`, then `python scripts/evaluate.py --model models/cell_classifier_candidate.h5`, then promote it (see [docs/DEPLOYMENT.md §9](docs/DEPLOYMENT.md#9-day-2-operations)).

### What would actually make it useful

* **More cases.** 85 cases is far too few for a reliable slide classifier. Aim for hundreds, ideally from more than one site and scanner.
* **Verified labels and real patient IDs**, so splits can be done by patient instead of inferred from filenames.
* **Tile-based training** for the gigapixel TIFFs: downsizing a 42,000 × 26,000 px slide to 224 px discards almost all cellular detail. Train on 224 px tiles at 10× and aggregate per slide.
* Report sensitivity and specificity at a clinically chosen threshold (see the threshold sweep in the report), not just accuracy.

---

## Configuration

* **Web app:** the decision threshold is `"threshold"` in [web/model/model.json](web/model/model.json).
* **Python / API server:** [src/cell_classifier/config.py](src/cell_classifier/config.py), overridable with environment variables `MODEL_PATH`, `CANCER_THRESHOLD`, `MAX_UPLOAD_MB`, `MAX_IMAGE_PIXELS`, `CORS_ORIGINS`, `DATA_DIR`.

## Licence and disclaimer

Research and educational use only. Not a medical device. Outputs must not be used for diagnosis or treatment decisions.
