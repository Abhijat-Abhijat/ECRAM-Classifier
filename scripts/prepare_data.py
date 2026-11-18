"""
Cache every raw slide as a 224x224 PNG, using the exact serving preprocessing.

Why: some raw TIFFs are >1 gigapixel. Decoding them on every training epoch takes
hours; decoding once and caching takes minutes. Because Keras resizes *before*
augmenting, training on the cache is equivalent to training on the raw files.

    python scripts/prepare_data.py                # data/raw -> data/processed
    python scripts/prepare_data.py --workers 1    # if you run low on RAM

Output keeps the original filename: data/processed/cancerous/<name>.tif.png
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cell_classifier import config  # noqa: E402
from cell_classifier.data import list_samples  # noqa: E402
from cell_classifier.preprocessing import load_image  # noqa: E402

FILTERS = {"nearest": Image.Resampling.NEAREST, "bicubic": Image.Resampling.BICUBIC}


def _process(src: Path, dests: dict[str, Path]) -> str:
    todo = {k: d for k, d in dests.items() if not d.exists()}
    if not todo:
        return "cached"
    with load_image(src, max_pixels=10**12) as img:
        for interp, dest in todo.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.resize(config.IMG_SIZE, resample=FILTERS[interp]).save(dest)
    return "written"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--bicubic-out", type=Path, default=None,
                    help="optionally also write BICUBIC-resized copies here (for the skew study)")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel decoders; each gigapixel TIFF needs ~3.5 GB RAM")
    args = ap.parse_args()

    samples = list_samples(args.raw)
    print(f"{len(samples)} images in {args.raw}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for s in samples:
            dests = {"nearest": args.out / s.label / f"{s.path.name}.png"}
            if args.bicubic_out:
                dests["bicubic"] = args.bicubic_out / s.label / f"{s.path.name}.png"
            futures[pool.submit(_process, s.path, dests)] = s
        for i, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            try:
                status = fut.result()
            except Exception as exc:  # keep going; report at the end
                status = f"FAILED: {exc}"
            print(f"[{i:3d}/{len(samples)}] {status:8s} {s.label}/{s.path.name}", flush=True)
    print(f"Done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
