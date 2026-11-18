from pathlib import Path

import pytest

from cell_classifier.data import Sample, case_id, code_mismatches, filename_code, group_split, keras_validation_split, label_conflicts, list_samples, slide_id


def test_slide_id_strips_all_image_extensions():
    assert slide_id("1744-EC165.tif") == slide_id("1744-EC165.png") == slide_id("1744-EC165.tif.png") == "1744-EC165"


def _fake(n_c=10, n_n=6, paired=3):
    s = []
    for label, n in (("cancerous", n_c), ("non_cancerous", n_n)):
        for i in range(n):
            s.append(Sample(Path(f"{label}/{label}{i}.tif"), label, f"{label}{i}", f"{label}{i}"))
            if i < paired:
                s.append(Sample(Path(f"{label}/{label}{i}.png"), label, f"{label}{i}", f"{label}{i}"))
    return s


@pytest.mark.parametrize("name,case,code", [
    ("EC 250 23 10 X 2.tif", "25023", "EC"), ("EC9310x.tif", "93", "EC"), ("EC2522310X 3.tif", "25223", "EC"),
    ("EN 120 N10X 2.tif", "120", "EN"), ("1744174255755kuvo-EC25223.png", "25223", "EC"),
    ("EC236 10X.tif 2.tif", "236", "EC"), ("EC 110 10X.tif", "110", "EC"), ("ec92 10X 2.tif", "92", "EC"),
])
def test_case_id_and_code(name, case, code):
    assert case_id(name) == case and filename_code(name) == code


def test_tumour_and_normal_of_same_case_share_case_id():
    assert case_id("EC 90 10X.tif") == case_id("EN 90 10X.tif") == case_id("EC 90 10 X 3.tif")


def test_conflicts_and_code_mismatches():
    a = Sample(Path("cancerous/EN 1 10X.tif"), "cancerous", "EN 1 10X", "1")
    b = Sample(Path("non_cancerous/EN 1 10X.tif"), "non_cancerous", "EN 1 10X", "1")
    assert label_conflicts([a, b]) == {"EN 1 10X"}
    assert code_mismatches([a, b]) == [a]


def test_group_split_has_no_case_leakage_and_is_stratified():
    tr, va = group_split(_fake(), n_splits=3, seed=1)
    assert not ({s.case for s in tr} & {s.case for s in va})
    assert {s.label for s in va} == {"cancerous", "non_cancerous"}
    assert len(tr) + len(va) == len(_fake())


def test_group_split_is_deterministic():
    assert group_split(_fake(), seed=7) == group_split(_fake(), seed=7)


def test_keras_split_takes_first_20_percent_per_class():
    samples = _fake(10, 5, paired=0)
    tr, va = keras_validation_split(samples, 0.2)
    assert [s.group for s in va] == ["cancerous0", "cancerous1", "non_cancerous0"]
    assert len(tr) == 12


def test_real_dataset_layout_if_present():
    from cell_classifier import config
    if not (config.RAW_DIR / "cancerous").is_dir():
        pytest.skip("raw data not present")
    samples = list_samples(config.RAW_DIR)
    assert len(samples) > 0 and {s.label for s in samples} == {"cancerous", "non_cancerous"}
