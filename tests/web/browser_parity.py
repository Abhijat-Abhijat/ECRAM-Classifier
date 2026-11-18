"""
Browser parity test for the static web app (web/).

Runs the real web app in headless Edge/Chrome and checks that, for every case, the
224x224 pixels the browser feeds the model are identical to the Python pipeline
(src/cell_classifier/preprocessing.py) and that the probability matches the golden
reports/predictions.csv.

    python tests/web/browser_parity.py                 # synthetic + a dataset sample
    python tests/web/browser_parity.py --all           # every image in data/raw (slow: gigapixel TIFFs)

Writes reports/browser_parity.json and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # trusted local dataset (gigapixel slides)

from cell_classifier import config  # noqa: E402
from cell_classifier.preprocessing import load_image  # noqa: E402

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def python_pixels(path: Path) -> np.ndarray:
    with load_image(path, max_pixels=10**12) as img:
        return np.asarray(img.resize(config.IMG_SIZE, Image.Resampling.NEAREST), dtype=np.uint8)


def make_synthetic(out: Path) -> list[Path]:
    """TIFF/PNG/JPEG variants covering every code path in web/imaging.js."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (1013, 1517, 3), dtype=np.uint8)
    arr[::7] = arr[::7] // 3               # some structure so predictors/LZW matter
    img = Image.fromarray(arr)
    gray = img.convert("L")
    cases = {
        "raw_strips.tif": (img, dict()),
        "lzw.tif": (img, dict(compression="tiff_lzw")),
        "lzw_predictor.tif": (img, dict(compression="tiff_lzw", tiffinfo={317: 2})),
        "deflate.tif": (img, dict(compression="tiff_adobe_deflate")),
        "deflate_predictor.tif": (img, dict(compression="tiff_adobe_deflate", tiffinfo={317: 2})),
        "packbits.tif": (img, dict(compression="packbits")),
        "gray.tif": (gray, dict()),
        "photo.png": (img, dict()),
        "rgba.png": (img.convert("RGBA"), dict()),
        "gray.png": (gray, dict()),
        "photo.bmp": (img, dict()),
    }
    paths = []
    try:  # Pillow cannot write tiled TIFFs; tifffile can (pip install tifffile)
        import tifffile
        tifffile.imwrite(out / "tiled_raw.tif", arr, tile=(256, 256), photometric="rgb")
        tifffile.imwrite(out / "tiled_deflate.tif", arr, tile=(128, 128), photometric="rgb", compression="zlib")
        tifffile.imwrite(out / "tiled_deflate_predictor.tif", arr, tile=(256, 256), photometric="rgb",
                         compression="zlib", predictor=True)
        tifffile.imwrite(out / "bigendian.tif", arr, byteorder=">", photometric="rgb")
        tifffile.imwrite(out / "bigtiff.tif", arr, bigtiff=True, photometric="rgb", tile=(256, 256))
        paths += [out / n for n in ("tiled_raw.tif", "tiled_deflate.tif", "tiled_deflate_predictor.tif",
                                    "bigendian.tif", "bigtiff.tif")]
    except ImportError:
        print("  (tifffile not installed: tiled/big-endian/BigTIFF synthetic cases skipped)")
    for name, (im, kw) in cases.items():
        p = out / name
        try:
            im.save(p, **kw)
        except Exception as exc:  # e.g. Pillow built without tiled-save support
            print(f"  (skip synthetic {name}: {exc})")
            continue
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="test every dataset image (slow)")
    ap.add_argument("--pyramidal", type=int, default=4, help="number of JPEG-tiled whole-slide TIFFs to test")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--match", default=None, help="only dataset files whose name contains this text (no synthetic cases)")
    args = ap.parse_args()

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        sys.exit("No Edge/Chrome found - cannot run the browser test.")

    tmp = Path(tempfile.mkdtemp(prefix="cc_browser_"))
    synth = make_synthetic(tmp)

    golden = {}
    gpath = config.REPORTS_DIR / "predictions.csv"
    if gpath.exists():
        golden = {r["file"]: float(r["prob_cancerous"]) for r in csv.DictReader(open(gpath, encoding="utf-8"))}

    cases = [{"id": f"synthetic/{p.name}", "name": p.name, "url": f"/__synthetic/{p.name}", "predict": False, "path": p}
             for p in synth]
    if config.RAW_DIR.is_dir():
        raw = sorted(config.RAW_DIR.glob("*/*"))
        pyramidal = [p for p in raw if p.suffix.lower() == ".tif" and getattr(Image.open(p), "n_frames", 1) > 1]
        simple_tif = [p for p in raw if p.suffix.lower() == ".tif" and p not in pyramidal]
        pngs = [p for p in raw if p.suffix.lower() == ".png"]
        if args.match:
            chosen = [p for p in raw if args.match in p.name]
            cases = []
        elif args.all:
            chosen = raw
        else:
            pyramidal.sort(key=lambda p: p.stat().st_size)
            pick = [pyramidal[int(i * (len(pyramidal) - 1) / max(1, args.pyramidal - 1))] for i in range(args.pyramidal)]
            chosen = simple_tif[::6] + pngs[::4] + pick   # every 6th / 4th + smallest..largest WSIs
        for p in chosen:
            rel = f"{p.parent.name}/{p.name}"
            cases.append({"id": rel, "name": p.name, "url": "/data/raw/" + rel.replace(" ", "%20"),
                          "predict": True, "path": p})

    results, done = {}, threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm",
                          ".mjs": "text/javascript", ".js": "text/javascript", ".onnx": "application/octet-stream",
                          ".webmanifest": "application/manifest+json"}

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def log_message(self, *a):
            pass

        def translate_path(self, path):
            if path.startswith("/__synthetic/"):
                return str(tmp / path.split("/")[-1])
            return super().translate_path(path)

        def do_GET(self):
            if self.path == "/__cases":
                body = json.dumps([{k: v for k, v in c.items() if k != "path"} for c in cases]).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            if self.path in ("/", "/selftest"):
                self.path = "/tests/web/selftest.html"
            return super().do_GET()

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if self.path == "/__result":
                results[data["id"]] = data
                print(f"  [{len(results):3d}/{len(cases)}] {data['id']}"
                      f"{'  ERROR ' + data['error'] if 'error' in data else ''}", flush=True)
            elif self.path == "/__done":
                if data.get("fatal"):
                    print("FATAL in browser:", data["fatal"])
                done.set()
            self.send_response(204); self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tmp / "profile"
    print(f"Serving on http://127.0.0.1:{port}/  browser: {Path(browser).name}  cases: {len(cases)}")
    proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                             "--no-first-run", "--disable-extensions", f"http://127.0.0.1:{port}/selftest"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=(os.name != "nt"))
    t0 = time.time()
    finished = done.wait(args.timeout)
    # Kill the whole browser process tree (the launcher's children otherwise keep running).
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)
    server.shutdown()
    if not finished:
        print(f"Timed out after {args.timeout}s with {len(results)}/{len(cases)} results")

    # ---- compare --------------------------------------------------------------
    report, failures = [], 0
    for c in cases:
        r = results.get(c["id"], {"error": "no result"})
        row = {"case": c["id"], "kind": r.get("kind"), "read_ms": r.get("read_ms"), "total_ms": r.get("total_ms")}
        if "error" in r:
            row["status"] = "ERROR: " + r["error"]; failures += 1; report.append(row); continue
        got = np.frombuffer(base64.b64decode(r["pixels"]), dtype=np.uint8).reshape(*config.IMG_SIZE, 3)
        cached = config.PROCESSED_DIR / Path(c["path"]).parent.name / f"{Path(c['path']).name}.png"
        if c["id"].startswith("synthetic/") or not cached.exists():
            exp = python_pixels(c["path"])
        else:  # same preprocessing, already computed by prepare_data.py (avoids re-decoding gigapixel TIFFs)
            exp = np.asarray(Image.open(cached).convert("RGB"), dtype=np.uint8)
        diff = np.abs(got.astype(int) - exp.astype(int))
        row["pixels_identical"] = bool(diff.max() == 0)
        row["pixel_max_diff"] = int(diff.max())
        row["pixel_mean_diff"] = round(float(diff.mean()), 4)
        is_jpeg = "JPEG" in (r.get("kind") or "")
        if "prob" in r and c["id"] in golden:
            row["prob_browser"] = round(r["prob"], 5)
            row["prob_python"] = golden[c["id"]]
            row["abs_diff"] = round(abs(r["prob"] - golden[c["id"]]), 5)
            row["label_flip"] = (r["prob"] >= 0.5) != (golden[c["id"]] >= 0.5)
        # Pass criteria: exact pixels for lossless formats; JPEG decoders may differ by a few levels.
        ok = row["pixels_identical"] if not is_jpeg else row["pixel_mean_diff"] < 2.0
        if "abs_diff" in row:
            ok = ok and (row["abs_diff"] < (1e-3 if not is_jpeg else 0.05)) and not row["label_flip"]
        row["status"] = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        report.append(row)

    summary = {
        "browser": Path(browser).name, "cases": len(cases), "failures": failures,
        "seconds": round(time.time() - t0),
        "lossless_cases_pixel_identical": sum(1 for r in report if r.get("pixels_identical")),
        "max_prob_abs_diff": max((r.get("abs_diff", 0) for r in report), default=0),
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "browser_parity.json").write_text(json.dumps({"summary": summary, "cases": report}, indent=2),
                                                           encoding="utf-8")
    for r in report:
        extra = f" dP={r['abs_diff']}" if "abs_diff" in r else ""
        print(f"{r['status']:5s} {r['case'][:60]:60s} px_max={r.get('pixel_max_diff')}{extra} "
              f"read={r.get('read_ms')}ms  {r.get('kind') or ''}")
    print("\nSUMMARY", json.dumps(summary))
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
