"""
Train the cancerous / non-cancerous classifier (MobileNetV2 transfer learning).

    python scripts/prepare_data.py          # once: cache 224x224 images
    python scripts/train.py --cv            # 5-fold case-grouped cross-validation -> reports/cv_results.*
    python scripts/train.py                 # train one model -> models/cell_classifier_candidate.h5
    python scripts/evaluate.py --model models/cell_classifier_candidate.h5
    # if it beats the current model on holdout_clean, promote it (Windows):
    #   copy models\\cell_classifier_candidate.h5 models\\cell_classifier.h5

Fixes vs. the original training script
--------------------------------------
* Split by CASE, not by file: all slides of a case number (every field of view,
  both file formats, both classes) stay on the same side of the split. The original
  random file split let the model memorise patients.
* Images with contradictory labels (same file in both class folders) are dropped.
* Validation images are NOT augmented (the original augmented them too).
* Class weights compensate for the ~2:1 cancerous:non-cancerous imbalance.
* Model selection on validation AUC; BatchNorm stays frozen during fine-tuning.
* Trains from the 224px cache -> minutes instead of hours on CPU.
* Same input contract as the API: RGB, NEAREST-resized, float32 in [0, 1].
* Writes <model>.split.json so evaluate.py scores the right held-out images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tensorflow.keras import layers, models  # noqa: E402

from cell_classifier import config  # noqa: E402
from cell_classifier.data import group_kfold, label_conflicts, list_samples  # noqa: E402


def to_arrays(samples):
    x = np.stack([np.asarray(Image.open(s.path).convert("RGB"), dtype=np.float32) / 255.0 for s in samples])
    # Keras label convention: 0 = cancerous, 1 = non_cancerous (sigmoid = P(non_cancerous))
    y = np.array([0.0 if s.label == "cancerous" else 1.0 for s in samples], dtype=np.float32)
    return x, y


def build_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(*config.IMG_SIZE, 3),
                                             include_top=False, weights="imagenet")
    base.trainable = False
    model = models.Sequential([
        layers.Input(shape=(*config.IMG_SIZE, 3)),
        layers.Rescaling(2.0, offset=-1.0),   # [0,1] -> [-1,1], what MobileNetV2 expects
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model, base


def train_one(train_s, val_s, out: Path, args) -> np.ndarray:
    """Train on train_s, select the best epoch on val_s, save to `out`. Returns P(cancerous) on val_s."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    x_tr, y_tr = to_arrays(train_s)
    x_va, y_va = to_arrays(val_s)

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(30 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.15, 0.15, fill_mode="nearest"),
        layers.RandomZoom(0.2, fill_mode="nearest"),
        layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
        layers.RandomContrast(0.1),
    ])
    train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(len(x_tr), seed=args.seed)
                .batch(args.batch_size)
                .map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0.0, 1.0), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va)).batch(args.batch_size)

    n_pos, n_neg = float(y_tr.sum()), float(len(y_tr) - y_tr.sum())
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

    model, base = build_model()
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cbs = [tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_auc", mode="max",
                                              save_best_only=True, verbose=0),
           tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                            restore_best_weights=True)]
    verbose = 2 if not args.cv else 0

    print("  phase 1: classifier head (base frozen)")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight, callbacks=cbs, verbose=verbose)

    if args.fine_tune_epochs > 0:
        print("  phase 2: fine-tune last 30 layers")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm frozen on tiny datasets
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=metrics)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                  class_weight=class_weight, callbacks=cbs, verbose=verbose)

    if not out.exists():  # checkpoint never fired
        model.save(out)
    best = tf.keras.models.load_model(out, compile=False)
    return 1.0 - best.predict(x_va, verbose=0).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "models" / "cell_classifier_candidate.h5")
    ap.add_argument("--cv", action="store_true", help="run 5-fold case-grouped cross-validation instead")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--fine-tune-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = list_samples(args.data)
    bad = label_conflicts(samples)
    samples = [s for s in samples if s.group not in bad]
    print(f"{len(samples)} images, {len({s.case for s in samples})} cases (dropped conflicting: {sorted(bad)})")

    folds = list(group_kfold(samples, args.folds, args.seed))
    for tr, va in folds:
        assert not ({s.case for s in tr} & {s.case for s in va}), "case leakage!"

    if args.cv:
        rows, all_y, all_p = [], [], []
        tmp = config.PROJECT_ROOT / "models" / "_cv_tmp.h5"
        for i, (tr, va) in enumerate(folds):
            print(f"\nFold {i + 1}/{len(folds)}: train {len(tr)} / val {len(va)} images "
                  f"({len({s.case for s in va})} cases)")
            p = train_one(tr, va, tmp, args)
            y = np.array([s.y for s in va])
            acc = float(np.mean((p >= 0.5) == y))
            auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
            print(f"  fold {i + 1}: accuracy {acc:.3f}  AUC {auc:.3f}")
            rows.append({"fold": i + 1, "n_val": len(va), "accuracy": acc, "roc_auc": auc})
            all_y += list(y)
            all_p += list(p)
        tmp.unlink(missing_ok=True)
        all_y, all_p = np.array(all_y), np.array(all_p)
        summary = {
            "folds": rows,
            "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "std_accuracy": float(np.std([r["accuracy"] for r in rows])),
            "mean_roc_auc": float(np.nanmean([r["roc_auc"] for r in rows])),
            "std_roc_auc": float(np.nanstd([r["roc_auc"] for r in rows])),
            "pooled_accuracy": float(np.mean((all_p >= 0.5) == all_y)),
            "pooled_roc_auc": float(roc_auc_score(all_y, all_p)),
            "settings": {k: v for k, v in vars(args).items() if k not in ("data", "out")},
        }
        config.REPORTS_DIR.mkdir(exist_ok=True)
        (config.REPORTS_DIR / "cv_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nCV accuracy {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f} | "
              f"AUC {summary['mean_roc_auc']:.3f} ± {summary['std_roc_auc']:.3f} "
              f"(pooled AUC {summary['pooled_roc_auc']:.3f}) -> reports/cv_results.json")
        return

    tr, va = folds[0]
    print(f"train {len(tr)} / val {len(va)} images ({len({s.case for s in va})} held-out cases)")
    p = train_one(tr, va, args.out, args)
    split_file = args.out.with_suffix(".split.json")
    split_file.write_text(json.dumps({
        "val": [f"{s.label}/{s.path.name}" for s in va],
        "class_indices": {"cancerous": 0, "non_cancerous": 1},
        "input": "RGB, NEAREST-resized to 224x224, float32 in [0,1]",
    }, indent=2), encoding="utf-8")
    y = np.array([s.y for s in va])
    print(f"\nSaved {args.out} (+ {split_file.name}). Held-out accuracy {np.mean((p >= 0.5) == y):.3f}, "
          f"AUC {roc_auc_score(y, p):.3f}")
    print(f"Next: python scripts/evaluate.py --model {args.out}")


if __name__ == "__main__":
    main()
