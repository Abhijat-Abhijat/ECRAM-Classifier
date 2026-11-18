"""
FastAPI service: serves the web UI and the prediction API.

Run locally (from the project root):
    uvicorn --app-dir src cell_classifier.api.main:app --reload --port 8000
Then open http://127.0.0.1:8000  (API docs: http://127.0.0.1:8000/docs)

Endpoints
    GET  /             web UI (mobile + desktop)
    GET  /health       liveness/readiness probe
    GET  /api/info     model + limits metadata
    POST /api/predict  multipart form field "file" -> prediction JSON
    POST /predict      alias kept for backward compatibility
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from .. import __version__, config
from ..inference import Predictor
from ..preprocessing import ImageTooLargeError

log = logging.getLogger("cell_classifier.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
CHUNK = 1024 * 1024


def create_app(predictor_factory: Callable[[], Predictor] = Predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        app.state.predictor = predictor_factory()
        log.info("Model loaded in %.1fs", time.perf_counter() - t0)
        yield

    app = FastAPI(title="Cell Classifier API", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=()")
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        # UI expects {"error": "..."}
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def manifest():
        return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": getattr(request.app.state, "predictor", None) is not None,
                "version": __version__,
                # Render sets RENDER_GIT_COMMIT; CI uses it to confirm the new deploy is live.
                "commit": os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT")}

    @app.get("/api/info")
    def info(request: Request):
        p = request.app.state.predictor
        return {"version": __version__, "model": Path(p.model_path).name,
                "backend": getattr(p, "backend", None),
                "classes": list(config.CLASS_NAMES.values()), "threshold": p.threshold,
                "input_size": list(config.IMG_SIZE), "max_upload_mb": config.MAX_UPLOAD_MB,
                "max_image_megapixels": config.MAX_IMAGE_PIXELS // 1_000_000,
                "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS)}

    @app.post("/api/predict")
    @app.post("/predict", include_in_schema=False)
    def predict(request: Request, file: UploadFile = File(...)):
        # Sync endpoint -> FastAPI runs it in a worker thread, so TF never blocks the event loop.
        name = file.filename or ""
        ext = Path(name).suffix.lower()
        if not (file.content_type or "").startswith("image/") and ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(400, "That file doesn't look like an image. Please upload a JPG, PNG or TIF.")

        limit = config.MAX_UPLOAD_MB * 1024 * 1024
        buf = bytearray()
        while chunk := file.file.read(CHUNK):
            buf += chunk
            if len(buf) > limit:
                raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
        if not buf:
            raise HTTPException(400, "The uploaded file is empty.")

        t0 = time.perf_counter()
        try:
            result = request.app.state.predictor.predict(bytes(buf))
        except ImageTooLargeError as exc:
            raise HTTPException(413, str(exc)) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("Unreadable image %r: %s", name, exc)
            raise HTTPException(400, "Couldn't read that image. Please try a different file.") from exc
        ms = (time.perf_counter() - t0) * 1000
        log.info("predict size=%.1fMB label=%s p_cancer=%.1f%% %.0fms",
                 len(buf) / 1e6, result.label, result.prob_cancerous, ms)
        return {**result.to_dict(), "processing_ms": round(ms)}

    return app


app = create_app()
