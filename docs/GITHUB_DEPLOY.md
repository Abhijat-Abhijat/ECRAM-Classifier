# GitHub + free auto-deploy (UI on GitHub Pages, backend on Render)

Once this is set up, every `git push` to `main` runs the tests and then deploys both parts automatically. Everything runs on free plans.

```
git push ──► GitHub Actions (.github/workflows/ci.yml)
               ├─ test      pytest (Keras + ONNX backends must agree) + headless-Chrome parity test
               ├─ docker    builds the backend image, runs it with 512 MB RAM, calls /health and /api/predict
               │            (both must pass before anything deploys)
               ├─ deploy-ui ──► GitHub Pages   https://<github-user>.github.io/<repo>/
               │                web/ = in-browser app: the model runs on the phone/laptop, images never leave it
               ├─ deploy-api ─► Render (free)  https://<service>.onrender.com
               │                FastAPI + ONNX Runtime: /api/predict, /docs, and its own upload UI at /
               └─ deploy-hf  ─► Hugging Face Static Space (optional mirror, only if configured)
```

| Part | Host | Cost | Limits worth knowing |
|---|---|---|---|
| UI (`web/`) | GitHub Pages | Free | Repo must be **public** on a free GitHub account. 1 GB site, 100 GB/month bandwidth (soft limits). |
| Backend (`Dockerfile`) | Render free web service | Free | 512 MB RAM, 0.1 CPU, 750 instance-hours/month. **Sleeps after 15 min without traffic, and the first request after that takes ~1 min.** Uploads are capped at 40 megapixels (see `render.yaml`); whole-slide TIFFs work in the Pages UI instead. |
| CI | GitHub Actions | Free | Unlimited minutes for public repos (2,000 min/month for private). |

The backend serves the **same ONNX file** as the browser (`web/model/cell_classifier.onnx`) through ONNX Runtime. It uses no TensorFlow and about 130 MB of RAM, which is why it fits Render's free tier. The tests check that its answers match the evaluated Keras model (`EN 90 10X.tif` → 27.8% on all three paths).

---

## 1. Put the code on GitHub (≈10 min, one time)

**What goes into the repo.** `.gitignore` excludes `data/` (4.2 GB of slides), `*.tif`, `.venv/` and model experiments. What is committed is the code, `models/cell_classifier.h5` (23.6 MB), `web/` (including the 9.6 MB ONNX model and the 14.8 MB ONNX Runtime wasm) and `reports/`. Large files are committed as normal git files, **not Git LFS**. All of them are under GitHub's 100 MB limit, and LFS's free 1 GB/month bandwidth would run out from CI checkouts.

> **Public repo?** GitHub Pages on a free account needs a public repo. Anyone could then download the code, the model, and `reports/predictions.csv`, which lists the slide **file names** and their predictions but no images. If that is not acceptable, make the repo private and host the UI on the Hugging Face Static Space you already have, or on Cloudflare Pages. Then set repo variable `DEPLOY_PAGES` = `false` (step 3).

1. Go to <https://github.com/new>. Pick a name (e.g. `cell-classifier`) and choose **Public**. Do **not** add a README, .gitignore or licence, because the repo must start empty. Click **Create repository**.
2. In PowerShell in the project folder. `git` is installed but not on PowerShell's PATH on this PC, so either use **Git Bash** or first run `$env:Path += ";C:\Program Files\Git\cmd"`.

   ```bash
   cd E:/3_aug_2026_d_drive_data/cancer_project
   git config --global user.name  "Your Name"            # once per PC, if not set yet
   git config --global user.email "you@example.com"      # use the email of your GitHub account
   git init -b main                                        # skip if already done
   git add .
   git status                                              # check: no data/, no .venv/, no *.tif
   git commit -m "Initial commit: cell classifier, web app, API, CI/CD"
   git remote add origin https://github.com/<github-user>/<repo>.git
   git push -u origin main
   ```

   The first push opens a browser window to sign in to GitHub (Git Credential Manager). The upload is about 50 MB.

---

## 2. Turn on GitHub Pages (UI) (1 min)

1. In the repo, open **Settings → Pages**.
2. Under *Build and deployment → Source*, choose **GitHub Actions**. Nothing else needs changing.
3. Open the **Actions** tab. If the first run already failed at `deploy-ui` (it ran before Pages was enabled), open the run and click **Re-run failed jobs**.
4. When `deploy-ui` turns green, its summary shows the URL: **`https://<github-user>.github.io/<repo>/`**. Open it. The status should say *Model ready (on device)*.

---

## 3. Create the backend on Render (≈10 min, one time)

1. Sign up at <https://render.com> with **"GitHub"** and allow Render to access the repo. You can grant access to only this repo.
   *Render may ask for a card to verify the account. The free instance type is not billed. If you don't want to enter a card, skip the backend; the UI on Pages doesn't need it.*
2. **New + → Blueprint**, select the repo, then **Apply**. Render reads [`render.yaml`](../render.yaml) and creates the free Docker web service `cell-classifier-api`. The first build takes ~5 min.
3. When it shows **Live**, copy its URL from the top of the service page, e.g. `https://cell-classifier-api.onrender.com`. Render adds a suffix if the name is taken.
   Check it: `https://…onrender.com/health` should show `{"status":"ok","model_loaded":true,...}`; `/docs` shows the interactive API; `/` shows the upload UI.
4. **Connect it to GitHub Actions.** Deploys then happen only after the tests pass. `autoDeploy` is off in `render.yaml`.
   * Render: service → **Settings → Deploy Hook** → copy the URL. Treat it like a password, because anyone holding it can trigger deploys.
   * GitHub: repo → **Settings → Secrets and variables → Actions**:
     * *Secrets* tab → **New repository secret**: `RENDER_DEPLOY_HOOK_URL` = the deploy-hook URL.
     * *Variables* tab → **New repository variable**: `API_URL` = `https://cell-classifier-api.onrender.com` (no trailing slash). With this set, CI waits until `/health` reports the new commit, so a failed deploy turns the run red.

Optional repository variables/secrets:

| Name | Kind | Effect |
|---|---|---|
| `DEPLOY_PAGES` = `false` | variable | Don't deploy to GitHub Pages (e.g. private repo). |
| `HF_SPACE` = `cell-classifier` + `HF_TOKEN` | variable + secret | Also update your Hugging Face Static Space on every push. Use a write token from <https://huggingface.co/settings/tokens>. |

---

## 4. Everyday use

| You want to… | Do this |
|---|---|
| Ship a change | `git add -A && git commit -m "…" && git push`. The Actions tab shows test → docker → deploy-ui / deploy-api. Nothing deploys if a test fails. |
| Try something without deploying | Push a branch and open a Pull Request. Tests and the Docker smoke test run, but nothing deploys. |
| Redeploy without changes | Actions → *ci* → **Run workflow** (on `main`). |
| Ship a new model | `scripts/train.py` → `scripts/evaluate.py` → copy over `models/cell_classifier.h5` → **`python scripts/export_onnx.py`** → commit **both** the `.h5` and `web/model/*` → push. CI fails if you forget the export, because `model.json` records the `.h5` hash. |
| Roll back | `git revert <bad-commit> && git push`. For Render only, you can also use service → *Events* → pick an older deploy → **Rollback**. |
| Change the threshold | UI: `"threshold"` in `web/model/model.json`. Backend: `CANCER_THRESHOLD` in `render.yaml` / Render → *Environment*. Keep them equal. |
| See backend logs | Render → service → **Logs** (each prediction logs size, label, probability and time, but never the image). |

## 5. Calling the backend

```bash
curl -F "file=@slide.png" https://cell-classifier-api.onrender.com/api/predict
# {"label":"non_cancerous","is_cancerous":false,"confidence":72.2,"prob_cancerous":27.8,"threshold":50.0,"processing_ms":1035}
```

The first call after 15 idle minutes waits ~1 min while the instance wakes up. To keep it awake during a demo, open `/health` beforehand. CORS is open (`CORS_ORIGINS="*"`), so web pages on other domains can call it. Set `CORS_ORIGINS` to your Pages origin (`https://<github-user>.github.io`) to restrict that.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `deploy-ui`: *"Get Pages site failed"* / *"Not Found"* | Pages isn't enabled: do step 2, then *Re-run failed jobs*. |
| Pages site loads but says *Model failed to load* | Open the browser console. A 404 on `model/…` or `vendor/ort/…` means those files weren't committed: `git ls-files web` must list `web/model/cell_classifier.onnx` and the three `web/vendor/ort/*` files. |
| `deploy-api` notice *"RENDER_DEPLOY_HOOK_URL secret not set"* | Step 3.4. The name must match exactly. |
| `deploy-api` red after 20 min | Render → service → *Events/Logs*. Common causes: the build is still queued (free builds can wait), or the instance ran out of memory, which shows as *"Ran out of memory"* in the logs. |
| Backend answers `413 … megapixels` | Expected for huge whole-slide images on 512 MB. Use the Pages UI for those (it streams TIFF tiles on the device), or raise `MAX_IMAGE_PIXELS` on a larger instance. |
| `git push` rejected: *file exceeds 100 MB* | Something unintended was added (e.g. a slide). Run `git rm --cached <file>`, add it to `.gitignore`, then `git commit --amend` and push again. |
| `test` job fails in `test_web.py` with *"models/*.h5 changed"* | Run `python scripts/export_onnx.py` and commit `web/model/`. |
