# End-to-end free deployment guide

This guide takes the project from a Windows laptop to a public HTTPS link that works on phones and laptops, **for free, with no credit card and no server to run**.

## How it works: the model runs in the browser

Hugging Face now needs a **paid plan** to create Docker or Gradio Spaces. Its docs say: *"Gradio and Docker Spaces run on compute and require a paid plan to create… Static Spaces are free for everyone."* So the production deployment is a **static website**:

```
Phone / laptop browser  ──HTTPS──►  any free static host (Hugging Face Static Space, Netlify, GitHub Pages, Cloudflare Pages)
        │                                   serves only files: web/index.html, app.js, imaging.js, model/*.onnx, vendor/ort/*
        ▼
  imaging.js   reads the image ON THE DEVICE and samples exactly the 224×224 pixels the model needs
               (Pillow-identical NEAREST; whole-slide TIFFs are streamed tile by tile, never fully decoded)
  ONNX Runtime Web (WebAssembly) runs cell_classifier.onnx → P(cancerous)
```

| Property | Value |
|---|---|
| Cost | **Free** on every host below |
| Privacy | Images **never leave the device**. There is no backend. |
| Accuracy parity | Verified in headless Edge on **all 237 dataset images + 16 synthetic TIFF/PNG/BMP variants**: the 224×224 pixels fed to the model are **identical** to the Python pipeline, and probabilities differ by ≤ 0.00005 (`reports/browser_parity.json`) |
| Download | ~25 MB on first visit (9.6 MB model + 14.8 MB WebAssembly runtime), then browser-cached |
| Speed (measured, laptop) | PNG/normal TIFF: ~0.02–0.15 s to read. Whole-slide TIFF (up to ~770 MP): median 5.9 s. Inference: ~85 ms. Expect phones to be several times slower. |
| No cold starts | Static files are always on. Nothing sleeps. |

The FastAPI/Docker server (`src/`, `Dockerfile`) still exists for local use, integrations and self-hosting. See [§8](#8-optional-the-api-server-docker).

---

## 1. Prerequisites (one time)

| What | Where | Needed for |
|---|---|---|
| **Python 3.12** (64-bit) | <https://www.python.org/downloads/windows/>, tick **"Add python.exe to PATH"** | Tests, ONNX export, deploy script. TensorFlow has no stable wheels for Python 3.14 yet. 3.12 installs alongside it. |
| **Hugging Face account** | <https://huggingface.co/join> | Option A/B |
| **HF access token, type Write** | <https://huggingface.co/settings/tokens> → *Create new token* → **Write** | Option A. Keep it secret. |
| Edge or Chrome | already on Windows | Browser parity test |

```powershell
py -3.12 --version          # → Python 3.12.x
```

---

## 2. Set up and test locally (~10 min)

PowerShell in the project folder (`E:\3_aug_2026_d_drive_data\cancer_project`):

```powershell
# 2.1 Fresh environment on Python 3.12 (delete an old 3.14 .venv first: Remove-Item -Recurse -Force .venv)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# 2.2 Unit + API + web-bundle tests (~2–3 min)
python -m pytest -m "not slow"

# 2.3 Full suite incl. real model, ONNX-vs-Keras and golden predictions
#     (needs data/processed from scripts/prepare_data.py and reports/predictions.csv)
python -m pytest

# 2.4 Real-browser parity test: runs the actual web app in headless Edge and compares
#     its pixels and probabilities with Python. Writes reports/browser_parity.json.
python tests/web/browser_parity.py          # synthetic cases + a dataset sample, ~1 min
python tests/web/browser_parity.py --all    # all 237 dataset images
```

### 2b. Run the static site locally

```powershell
python -m http.server 8080 --directory web
```

* **Laptop:** <http://127.0.0.1:8080>. The header dot turns green: "Model ready (on device)".
* **Phone on the same Wi-Fi:** `ipconfig` → IPv4 address (e.g. `192.168.1.23`) → open `http://192.168.1.23:8080` on the phone. If it doesn't load, allow Python through Windows Firewall (Private network) when prompted.

Try `data/raw/non_cancerous/EN 90 10X.tif`. The result must be **Non-cancerous, P(cancer) 27.8%**, the same as `reports/predictions.csv`.

---

## 3. When the model changes: re-export the browser model

The site uses `web/model/cell_classifier.onnx`, exported from `models/cell_classifier.h5`. Whenever you replace the `.h5`:

```powershell
python scripts/export_onnx.py      # writes web/model/cell_classifier.onnx + model.json
                                   # and refuses to publish unless ONNX == Keras on all 237 images
python scripts/evaluate.py         # refresh reports/ (golden predictions)
python -m pytest
python tests/web/browser_parity.py
```

The deploy script refuses to upload if the `.h5` is newer than the `.onnx`.

---

## 4. Deploy (pick one; all free)

### Option A: Hugging Face Static Space, one command (recommended)

```powershell
.\.venv\Scripts\Activate.ps1
hf auth login                        # paste the WRITE token; "n" to git credential
python scripts/deploy_hf.py --space cell-classifier
```

The script creates the Space with the **static** SDK (free), uploads **only** `web/` (never `data/`), and adds a Space card README. It prints:

```
Space page : https://huggingface.co/spaces/<username>/cell-classifier
App URL    : https://<username>-cell-classifier.static.hf.space
```

The site is live as soon as the upload finishes, with no build step. Use the **App URL** on phones: full screen, without the Hugging Face frame. Re-run the same command to redeploy (only changed files upload). Add `--private` to restrict it to your HF account (visitors must be logged in to huggingface.co).

### Option B: Hugging Face, browser only

1. <https://huggingface.co/new-space> → name `cell-classifier` → **SDK: Static** → **Blank** → Public → *Create Space*.
2. *Files* → *+ Contribute* → *Upload files* → drag in **the contents of `web/`** (not the folder itself): `index.html`, `app.js`, `imaging.js`, `styles.css`, `manifest.webmanifest`, the icons, and the folders `model/` and `vendor/`.
3. *Commit changes to main*. Large files (`.onnx`, `.wasm`) are stored with LFS/Xet automatically.
4. Keep the Space's own `README.md`; its header must contain `sdk: static`.

### Option C: Netlify Drop (no account settings, 1 minute)

1. Open <https://app.netlify.com/drop> and log in (free; GitHub/email).
2. Drag the **`web` folder** onto the page. You get `https://<random-name>.netlify.app` in seconds.
3. *Site configuration → Change site name* for a nicer URL. To update, drag the folder again on the site's *Deploys* page.

### Option D: GitHub Pages / Cloudflare Pages

* **GitHub Pages (automated, with the backend on Render):** follow **[GITHUB_DEPLOY.md](GITHUB_DEPLOY.md)**. Every push to `main` is tested, then deployed. URL: `https://<user>.github.io/<repo>/`.
* **Cloudflare Pages:** *Workers & Pages → Create → Pages → Upload assets* → drag `web/`. URL: `https://<project>.pages.dev`. Every file is under its 25 MB per-file limit.

---

## 5. Verify the deployment

1. Open the App URL on a laptop. The status shows **"Model ready (on device)"** within a few seconds.
2. Analyze `data/raw/non_cancerous/EN 90 10X.tif` → **27.8%**, and `data/raw/cancerous/1744174588706eu7h-EC181.tif` (73 MB whole-slide TIFF) → **79.1%**. Same numbers as `reports/predictions.csv` means the deployment is faithful.
3. Optional: DevTools → *Network*. After the page loads, analyzing an image makes **no network requests**, which confirms the image stays on the device.
4. **Phone:** open the App URL → *Take photo* or tap the slide area to pick a file. To install it as an app: Android Chrome ⋮ → *Add to Home screen* / *Install app*; iOS Safari Share → *Add to Home Screen*.

---

## 6. Limits and supported files

| Input | Support |
|---|---|
| JPG, PNG, WebP, BMP, GIF | Yes, via the browser's own decoder (PNG, BMP and TIFF are pixel-exact vs Python; JPEG photos may differ by decoder rounding). iPhone **HEIC** isn't reliably decodable in browsers: set *Camera → Formats → Most Compatible* to shoot JPEG. |
| TIFF (8-bit RGB/grayscale) | Uncompressed, JPEG, LZW, Deflate, PackBits. Strips or tiles. Predictor 2. Little/big-endian. BigTIFF. |
| Whole-slide (pyramidal) TIFF | Yes. Full-resolution level only, as the model was trained. Only the needed tiles are decoded (~11,500 tiles for a 750 MP slide). |
| Not supported | 16-bit or planar TIFF, CMYK, other TIFF compressions (JPEG 2000, ZSTD…), vendor WSI formats (.svs/.ndpi/.mrxs). Export these as PNG/JPEG. |
| File size | No server limit. A phone's free memory is the practical limit (the reader streams, so large TIFFs are fine). |

---

## 7. Day-2 operations

| Task | How |
|---|---|
| Redeploy after UI changes | Option A: re-run `deploy_hf.py`. Netlify/Cloudflare: drag `web/` again. |
| Ship a new model | `scripts/train.py` → `scripts/evaluate.py --model models/cell_classifier_candidate.h5` → if better, copy it over `models/cell_classifier.h5` → **§3** → redeploy |
| Roll back | HF: *Files → History* → revert the commit. Netlify: *Deploys* → pick an older deploy → *Publish deploy*. |
| Change the threshold | Edit `"threshold"` in `web/model/model.json` (0.5 = default; see the threshold sweep in `reports/evaluation.md`) and redeploy |
| CI/CD | `.github/workflows/ci.yml` runs pytest, the browser parity test and a Docker smoke test on every push. Pushes to `main` then deploy to GitHub Pages and Render, and to the HF Static Space if `HF_SPACE`/`HF_TOKEN` are set. See [GITHUB_DEPLOY.md](GITHUB_DEPLOY.md). |

**Privacy and safety:** there's no server, no logs and no storage, so images stay on the device. It's still a **research prototype, not a medical device**, and the page says so. Don't advertise it for clinical use.

---

## 8. Optional: the API server (Docker)

Use this when another program needs an HTTP API, or to run everything on a hospital machine.

```powershell
# local, no Docker
uvicorn --app-dir src cell_classifier.api.main:app --host 0.0.0.0 --port 8000     # http://127.0.0.1:8000/docs
# Docker
docker build -t cell-classifier .
docker run --rm -p 7860:7860 cell-classifier                                        # http://localhost:7860
# local, lightweight like production (only requirements.txt installed, no TensorFlow)
$env:MODEL_PATH="web/model/cell_classifier.onnx"; uvicorn --app-dir src cell_classifier.api.main:app --port 8000
```

| Host | Cost | Notes |
|---|---|---|
| Your own PC/server | Free | `docker run -d -p 80:7860 --restart unless-stopped cell-classifier` |
| Your PC + **Cloudflare Quick Tunnel** | Free | Temporary public HTTPS link: `winget install Cloudflare.cloudflared`, then `cloudflared tunnel --url http://localhost:8000`. The PC must stay on. |
| Hugging Face Docker Space | **Paid plan** (PRO/Team) | `python scripts/deploy_hf.py --space cell-classifier-api --docker` |
| Google Cloud Run | Free tier, but **needs a billing card** | `gcloud run deploy --source . --memory 4Gi --port 7860` |
| **Render free** (recommended) | Free | The image runs the ONNX model with ONNX Runtime (~130 MB RAM, no TensorFlow), so it fits the 512 MB free instance. Sleeps after 15 min idle. Auto-deployed from GitHub: [GITHUB_DEPLOY.md](GITHUB_DEPLOY.md). |

API settings (env vars): `MODEL_PATH` (`.onnx` → ONNX Runtime, `.h5` → TensorFlow; the image uses the ONNX file), `CANCER_THRESHOLD`, `MAX_UPLOAD_MB` (100), `MAX_IMAGE_PIXELS` (1.2e9; 4e7 on Render), `ORT_THREADS`, `CORS_ORIGINS`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` fails: *No matching distribution for tensorflow-cpu* | Python 3.14 | Use Python 3.12 (§1) |
| pip error mentioning *Windows Long Path* | 260-char path limit | Keep `.venv` in the project root, or enable long paths (admin): `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f`, then reboot |
| Status stays **"Loading model…"** | Slow connection (25 MB first load) or a blocked file | Wait. Check DevTools → *Network* that `model/cell_classifier.onnx` and `vendor/ort/*.wasm` return 200. |
| **"Model failed to load"** | Files missing from the upload (often `vendor/` or `model/`) | Re-upload the whole `web/` folder contents, keeping subfolders |
| Works on `http://127.0.0.1` but not when opening `index.html` by double-click | Browsers block `fetch` from `file://` | Serve it: `python -m http.server 8080 --directory web` |
| *"This TIFF uses compression … not supported"* | Exotic TIFF (JPEG 2000, ZSTD, 16-bit…) | Export the slide as PNG/JPEG |
| *"This browser can't decode that image"* | HEIC on Chrome/Android | Shoot or export as JPEG |
| Phone tab reloads/crashes on a huge TIFF | Phone out of memory | Use a laptop for whole-slide TIFFs, or export a PNG at 10× |
| Different probability than `reports/predictions.csv` | Stale ONNX after a model change | §3: `python scripts/export_onnx.py`, then redeploy |
| Results look "random" | **The current model doesn't generalise** (`reports/evaluation.md`) | Not a deployment bug. See README → *Model status*. |
