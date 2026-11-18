# Model evaluation — `cell_classifier.h5`

Generated 2026-11-18T05:03:30+00:00 by `scripts/evaluate.py`. Positive class = **cancerous**.

## Dataset

- 237 images = **180 unique slides** (56 slides exist as both .png and .tif) = **85 cases** (several fields of view per case number).
- 30 case numbers have slides in *both* classes (likely tumour + normal tissue of one patient), so splitting must be done by case.
- Held-out split: keras validation_split=0.2 → 190 train / 47 held-out images.
- **24 held-out images belong to a case that was also in training** and are excluded from `holdout_clean`.
- Contradictory labels (same file in both folders), excluded: EN 257 23 10X 2.

## Results

| Metric | holdout_clean (honest) | holdout_all | all_images (optimistic) |
|---|---|---|---|
| Images | 23 | 47 | 237 |
| Accuracy | 43.5% | 51.1% | 78.1% |
| Balanced accuracy | 43.9% | 49.3% | 76.5% |
| Sensitivity (recall) | 54.5% | 54.8% | 81.4% |
| Specificity | 33.3% | 43.8% | 71.6% |
| Precision (PPV) | 42.9% | 65.4% | 84.7% |
| NPV | 44.4% | 33.3% | 66.7% |
| F1 | 48.0% | 59.6% | 83.0% |
| ROC-AUC | 0.667 | 0.565 | 0.863 |

95% bootstrap CI on holdout_clean: accuracy 21.7%–65.2%, ROC-AUC 0.413–0.886.

### Confusion matrix (holdout_clean, threshold 0.5)

| | Predicted cancerous | Predicted non-cancerous |
|---|---|---|
| **Actually cancerous** | 6 (TP) | 5 (FN — missed cancer) |
| **Actually non-cancerous** | 8 (FP — false alarm) | 4 (TN) |

### Threshold sweep (holdout_clean)

Set `CANCER_THRESHOLD` to trade false alarms for missed cancers.

| Threshold on P(cancer) | Sensitivity | Specificity | Accuracy |
|---|---|---|---|
| 0.2 | 100.0% | 8.3% | 52.2% |
| 0.3 | 90.9% | 25.0% | 56.5% |
| 0.4 | 81.8% | 33.3% | 56.5% |
| 0.5 | 54.5% | 33.3% | 43.5% |
| 0.6 | 54.5% | 66.7% | 60.9% |
| 0.7 | 54.5% | 75.0% | 65.2% |

## Preprocessing skew check

Same images resized with **BICUBIC** instead of the training-time **NEAREST** filter: mean |ΔP| = 0.128, max |ΔP| = 0.721, **49 of 237 predictions flip**; holdout_clean accuracy becomes 47.8%. The API therefore uses NEAREST.

## Possible label noise

Filename code (`EC` = cancerous, `EN` = non-cancerous) disagrees with the folder for:

- `cancerous/EN 133 10X.tif`
- `cancerous/EN 257 23 10X 2.tif`
- `non_cancerous/1744173958397vv95-EC254.png`
- `non_cancerous/1744173958397vv95-EC254.tif`
- `non_cancerous/1744174058855uzsa-EC329.png`
- `non_cancerous/1744174058855uzsa-EC329.tif`
- `non_cancerous/ec92 10X 2.tif`

Ask the data owner to confirm these labels before retraining.

## Caveats

- holdout_clean has only 23 images; the confidence interval above is wide. Collect more slides (ideally from multiple patients/sites) before trusting any single number.
- Cases are inferred from filenames (no patient IDs available); a few names are ambiguous (e.g. `EC 246` vs `EC24623`), so some residual leakage is possible.
- Research prototype — not a medical device and not a diagnosis.
