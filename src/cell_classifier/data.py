"""Dataset indexing, data-quality checks and leakage-safe splitting.

Filename conventions observed in this dataset
---------------------------------------------
    EC 90 10X.tif, EC 90 10X 2.tif, EC 90 10 X 3.tif   -> 3 fields of view of case 90 (EC = cancerous)
    EN 90 10X.tif                                       -> normal tissue, same case number 90
    EC 250 23 10X.tif / EC25223 10X.tif                 -> case 250 / 252, "23" looks like a year suffix
    1744174255755kuvo-EC25223.png  (+ same-name .tif)   -> exported copy, 2 formats of one slide

So there are three levels of "the same thing":
    file  <  slide (same name, .png vs .tif)  <  case (same case number, many slides, both classes)
Splitting must happen at the CASE level, otherwise the model can memorise the
patient/stain instead of learning the disease.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import config

CLASSES = [config.CLASS_NAMES[i] for i in sorted(config.CLASS_NAMES)]  # ['cancerous', 'non_cancerous']
EXPECTED_CODE = {"cancerous": "EC", "non_cancerous": "EN"}

_EXPORT_PREFIX = re.compile(r"^\d{10,}[a-z0-9]*-", re.I)      # "1744174255755kuvo-"
_MAGNIFICATION = re.compile(r"N?(?:4|10|20|40)\s*X", re.I)     # "10X", "10 X", "N10X"
_CODE = re.compile(r"E[CN]", re.I)


def slide_id(filename: str) -> str:
    """'1744-EC165.tif', '1744-EC165.png' and cached '1744-EC165.tif.png' -> '1744-EC165'."""
    name = filename
    while Path(name).suffix.lower() in config.ALLOWED_EXTENSIONS:
        name = Path(name).stem
    return name


def _case_part(filename: str) -> str:
    """The part of the name before the magnification, export prefix removed."""
    name = _EXPORT_PREFIX.sub("", slide_id(filename))
    return _MAGNIFICATION.split(name, maxsplit=1)[0]


def case_id(filename: str) -> str:
    """Case number shared by all slides of one case, independent of class code.

    'EC 250 23 10 X 2.tif' -> '25023'; 'EN 90 10X.tif' -> '90'; 'EC9310x.tif' -> '93'.
    Falls back to the slide id when the name contains no number.
    """
    digits = "".join(re.findall(r"\d", _case_part(filename)))
    return digits or slide_id(filename).lower()


def filename_code(filename: str) -> str | None:
    """'EC' / 'EN' code written in the filename, or None."""
    m = _CODE.search(_case_part(filename))
    return m.group(0).upper() if m else None


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str          # class folder name
    group: str          # slide id; the .png and .tif of one slide share it
    case: str           # case id; all slides of one case share it

    @property
    def y(self) -> int:
        """1 = cancerous (positive class for metrics)."""
        return int(self.label == "cancerous")


def list_samples(root: Path) -> list[Sample]:
    """Index root/<class>/<image>, sorted exactly like Keras flow_from_directory."""
    samples = []
    for label in CLASSES:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for f in sorted(p.name for p in class_dir.iterdir()):
            if Path(f).suffix.lower() in config.ALLOWED_EXTENSIONS:
                samples.append(Sample(class_dir / f, label, slide_id(f), case_id(f)))
    return samples


def label_conflicts(samples: list[Sample]) -> set[str]:
    """Slide ids present in BOTH class folders (contradictory labels)."""
    labels = defaultdict(set)
    for s in samples:
        labels[s.group].add(s.label)
    return {g for g, ls in labels.items() if len(ls) > 1}


def code_mismatches(samples: list[Sample]) -> list[Sample]:
    """Samples whose EC/EN filename code disagrees with their folder."""
    return [s for s in samples
            if (c := filename_code(s.path.name)) is not None and c != EXPECTED_CODE[s.label]]


def keras_validation_split(samples: list[Sample], validation_split: float = 0.2):
    """Reproduce the split used to train models/cell_classifier.h5.

    Keras' DirectoryIterator takes, per class, files[0 : int(split*n)] (sorted) as
    the validation subset and the rest as training.
    """
    train, val = [], []
    for label in CLASSES:
        cls = [s for s in samples if s.label == label]
        cut = int(validation_split * len(cls))
        val += cls[:cut]
        train += cls[cut:]
    return train, val


def group_kfold(samples: list[Sample], n_splits: int = 5, seed: int = 42):
    """Yield (train, val) folds stratified by class and grouped by CASE (no case on both sides)."""
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = [s.y for s in samples]
    groups = [s.case for s in samples]
    for tr_idx, va_idx in sgkf.split(samples, y, groups):
        yield [samples[i] for i in tr_idx], [samples[i] for i in va_idx]


def group_split(samples: list[Sample], n_splits: int = 5, seed: int = 42, fold: int = 0):
    """One case-grouped, stratified train/val split (fold `fold` of `n_splits`)."""
    for i, split in enumerate(group_kfold(samples, n_splits, seed)):
        if i == fold:
            return split
    raise ValueError(f"fold {fold} out of range")
