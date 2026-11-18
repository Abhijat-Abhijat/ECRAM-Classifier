"""
Evaluate the trained classifier and write reports/evaluation.{json,md} + predictions.csv.

    python scripts/prepare_data.py   # once
    python scripts/evaluate.py

Subsets reported
----------------
holdout_clean   Held-out images (Keras validation subset for the original model, or
                <model>.split.json for models from scripts/train.py) MINUS any image
                whose CASE (same case number, any slide/format/class) was in the
                training set, MINUS images with contradictory labels. This is the
                only honest estimate of accuracy on new patients.
holdout_all     The full held-out subset (may contain cases also seen in training).
all_images      Every image. ~80% were seen in training -> optimistic, sanity check only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

from cell_classifier import config  # noqa: E402
from cell_classifier.data import code_mismatches, keras_validation_split, label_conflicts, list_samples  # noqa: E402
from cell_classifier.inference import Predictor  # noqa: E402


def load_cached(paths: list[Path]) -> np.ndarray:
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in paths])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    prec = div(tp, tp + fp)
    return {
        "n": int(len(y)), "n_cancerous": int(y.sum()), "n_non_cancerous": int((1 - y).sum()),
        "threshold": threshold,
        "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity_recall": sens,          # of truly cancerous slides, % flagged
        "specificity": spec,                 # of truly normal slides, % cleared
        "precision_ppv": prec,               # of flagged slides, % truly cancerous
        "npv": div(tn, tn + fn),             # of cleared slides, % truly normal
        "f1": div(2 * prec * sens, prec + sens),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y, p, threshold, n_boot=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    acc, auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        acc.append(np.mean((pb >= threshold) == yb))
        if 0 < yb.sum() < len(yb):
            auc.append(roc_auc_score(yb, pb))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"accuracy_95ci": q(acc), "roc_auc_95ci": q(auc)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--raw", type=Path, default=config.RAW_DIR)
    ap.add_argument("--processed", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--compare", type=Path, default=None,
                    help="second cache (e.g. bicubic) to measure preprocessing skew against")
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    args = ap.parse_args()

    raw = list_samples(args.raw)
    split_file = args.model.with_suffix(".split.json")
    if split_file.exists():  # model trained by scripts/train.py: use its own held-out set
        names = set(json.loads(split_file.read_text())["val"])
        val = [s for s in raw if f"{s.label}/{s.path.name}" in names
               or f"{s.label}/{s.path.name}.png" in names]
        train = [s for s in raw if s not in val]
        print(f"Using held-out split from {split_file.name} ({len(val)} images)")
    else:  # original model: reproduce the Keras validation_split it was trained with
        train, val = keras_validation_split(raw)
    train_cases = {s.case for s in train}
    val_ids = {s.path for s in val}
    conflicts = label_conflicts(raw)

    cached = lambda root, s: root / s.label / f"{s.path.name}.png"  # noqa: E731
    missing = [s for s in raw if not cached(args.processed, s).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not cached. Run scripts/prepare_data.py first.")

    predictor = Predictor(args.model)
    thr = predictor.threshold
    y = np.array([s.y for s in raw])
    p = predictor.predict_array(load_cached([cached(args.processed, s) for s in raw]))

    in_val = np.array([s.path in val_ids for s in raw])
    leaked = np.array([s.path in val_ids and s.case in train_cases for s in raw])
    conflicted = np.array([s.group in conflicts for s in raw])
    clean = in_val & ~leaked & ~conflicted
    subsets = {"holdout_clean": clean, "holdout_all": in_val, "all_images": np.ones(len(raw), bool)}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model.name),
        "positive_class": "cancerous",
        "dataset": {
            "images": len(raw), "unique_slides": len({s.group for s in raw}),
            "unique_cases": len({s.case for s in raw}),
            "cases_in_both_classes": len({s.case for s in raw if s.label == "cancerous"}
                                         & {s.case for s in raw if s.label == "non_cancerous"}),
            "split_source": split_file.name if split_file.exists() else "keras validation_split=0.2",
            "train_images": len(train), "val_images": len(val),
            "val_images_case_leaked_from_train": int(leaked.sum()),
            "label_conflicts": sorted(conflicts),
        },
        "subsets": {},
    }
    for name, mask in subsets.items():
        m = metrics(y[mask], p[mask], thr)
        if name == "holdout_clean":
            m.update(bootstrap_ci(y[mask], p[mask], thr))
            m["threshold_sweep"] = {f"{t:.1f}": {k: metrics(y[mask], p[mask], t)[k]
                                                 for k in ("sensitivity_recall", "specificity", "accuracy")}
                                    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}
        report["subsets"][name] = m

    # Possible label noise: filename code (EC/EN) disagrees with folder
    report["label_code_mismatches"] = [f"{s.label}/{s.path.name}" for s in code_mismatches(raw)]

    if args.compare:
        p2 = predictor.predict_array(load_cached([cached(args.compare, s) for s in raw]))
        report["preprocessing_skew"] = {
            "compared_cache": args.compare.name,
            "mean_abs_prob_diff": float(np.mean(np.abs(p - p2))),
            "max_abs_prob_diff": float(np.max(np.abs(p - p2))),
            "label_flips": int(np.sum((p >= thr) != (p2 >= thr))),
            "holdout_clean_accuracy_with_compare": metrics(y[clean], p2[clean], thr)["accuracy"],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(args.out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "prob_cancerous", "predicted", "correct", "subset"])
        for s, pi, v, c in zip(raw, p, in_val, clean):
            pred = "cancerous" if pi >= thr else "non_cancerous"
            subset = "holdout_clean" if c else ("holdout_leaked_or_conflict" if v else "train")
            w.writerow([f"{s.label}/{s.path.name}", s.label, f"{pi:.4f}", pred, pred == s.label, subset])
    (args.out / "evaluation.md").write_text(render_markdown(report), encoding="utf-8")
    print((args.out / "evaluation.md").read_text(encoding="utf-8"))


def render_markdown(r: dict) -> str:
    pct = lambda v: "n/a" if v != v else f"{v * 100:.1f}%"  # noqa: E731
    d = r["dataset"]
    L = [f"# Model evaluation — `{r['model']}`", "",
         f"Generated {r['generated_utc']} by `scripts/evaluate.py`. Positive class = **cancerous**.", "",
         "## Dataset", "",
         f"- {d['images']} images = **{d['unique_slides']} unique slides** (56 slides exist as both .png and .tif) "
         f"= **{d['unique_cases']} cases** (several fields of view per case number).",
         f"- {d['cases_in_both_classes']} case numbers have slides in *both* classes (likely tumour + normal tissue "
         "of one patient), so splitting must be done by case.",
         f"- Held-out split: {d['split_source']} → {d['train_images']} train / {d['val_images']} held-out images.",
         f"- **{d['val_images_case_leaked_from_train']} held-out images belong to a case that was also in training** "
         "and are excluded from `holdout_clean`.",
         f"- Contradictory labels (same file in both folders), excluded: {', '.join(d['label_conflicts']) or 'none'}.",
         "",
         "## Results", "",
         "| Metric | holdout_clean (honest) | holdout_all | all_images (optimistic) |",
         "|---|---|---|---|"]
    S = r["subsets"]
    rows = [("Images", "n"), ("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"),
            ("Sensitivity (recall)", "sensitivity_recall"), ("Specificity", "specificity"),
            ("Precision (PPV)", "precision_ppv"), ("NPV", "npv"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]
    for title, k in rows:
        vals = [str(S[s][k]) if k == "n" else (f"{S[s][k]:.3f}" if k == "roc_auc" else pct(S[s][k]))
                for s in ("holdout_clean", "holdout_all", "all_images")]
        L.append(f"| {title} | " + " | ".join(vals) + " |")
    h = S["holdout_clean"]
    cm = h["confusion_matrix"]
    L += ["", f"95% bootstrap CI on holdout_clean: accuracy {pct(h['accuracy_95ci'][0])}–{pct(h['accuracy_95ci'][1])}, "
          f"ROC-AUC {h['roc_auc_95ci'][0]:.3f}–{h['roc_auc_95ci'][1]:.3f}.", "",
          "### Confusion matrix (holdout_clean, threshold "
          f"{h['threshold']})", "",
          "| | Predicted cancerous | Predicted non-cancerous |", "|---|---|---|",
          f"| **Actually cancerous** | {cm['tp']} (TP) | {cm['fn']} (FN — missed cancer) |",
          f"| **Actually non-cancerous** | {cm['fp']} (FP — false alarm) | {cm['tn']} (TN) |", "",
          "### Threshold sweep (holdout_clean)", "",
          "Set `CANCER_THRESHOLD` to trade false alarms for missed cancers.", "",
          "| Threshold on P(cancer) | Sensitivity | Specificity | Accuracy |", "|---|---|---|---|"]
    for t, v in h["threshold_sweep"].items():
        L.append(f"| {t} | {pct(v['sensitivity_recall'])} | {pct(v['specificity'])} | {pct(v['accuracy'])} |")
    if "preprocessing_skew" in r:
        sk = r["preprocessing_skew"]
        L += ["", "## Preprocessing skew check", "",
              f"Same images resized with **BICUBIC** instead of the training-time **NEAREST** filter: "
              f"mean |ΔP| = {sk['mean_abs_prob_diff']:.3f}, max |ΔP| = {sk['max_abs_prob_diff']:.3f}, "
              f"**{sk['label_flips']} of {d['images']} predictions flip**; holdout_clean accuracy becomes "
              f"{pct(sk['holdout_clean_accuracy_with_compare'])}. The API therefore uses NEAREST."]
    if r["label_code_mismatches"]:
        L += ["", "## Possible label noise", "",
              "Filename code (`EC` = cancerous, `EN` = non-cancerous) disagrees with the folder for:", ""]
        L += [f"- `{m}`" for m in r["label_code_mismatches"]]
        L += ["", "Ask the data owner to confirm these labels before retraining."]
    L += ["", "## Caveats", "",
          f"- holdout_clean has only {h['n']} images; the confidence interval above is wide. "
          "Collect more slides (ideally from multiple patients/sites) before trusting any single number.",
          "- Cases are inferred from filenames (no patient IDs available); a few names are ambiguous "
          "(e.g. `EC 246` vs `EC24623`), so some residual leakage is possible.",
          "- Research prototype — not a medical device and not a diagnosis."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
