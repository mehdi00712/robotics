"""Tests for the data pipeline, metrics and prediction helpers.

Run:  pytest -q
Model tests are skipped automatically if models/best.pt is missing.
"""
import csv

import cv2
import numpy as np
import pytest

import config
import split
from download_openimages import crop_box, resize_max
from evaluate import compute_metrics
from predict import describe, draw_panel


def test_classes_are_unique_and_sorted():
    assert len(config.CLASSES) == len(set(config.CLASSES))
    assert config.CLASSES == sorted(config.CLASSES)
    assert 6 <= len(config.CLASSES) <= 10  # brief: 6-10 classes


def test_group_key_keeps_session_and_photo_together():
    from pathlib import Path
    assert split.group_key(Path("mug_openimages_abc123_0.jpg")) == "openimages_abc123"
    assert split.group_key(Path("stapler_bedroom_s01_20261101-125837-541.jpg")) == "bedroom_s01"


def test_assign_groups_covers_every_group_once():
    import random
    groups = [f"g{i}" for i in range(40)]
    out = split.assign_groups(groups, {"test": 0.15, "val": 0.15, "train": 0.7}, random.Random(0))
    assert set(out) == set(groups)
    assert {"train", "val", "test"} <= set(out.values())


def test_split_has_no_leakage_and_respects_openimages_test(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "split"
    rows = []
    img = np.zeros((32, 32, 3), np.uint8)
    for cls in config.CLASSES:
        (raw / cls).mkdir(parents=True)
        for i in range(20):
            name = f"{cls}_openimages_{cls}{i}_0.jpg"
            cv2.imwrite(str(raw / cls / name), img)
            rows.append([f"{cls}/{name}", cls, "x", f"{cls}{i}", "test" if i < 4 else "train"])
    with open(raw / "openimages_sources.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "openimages_class", "openimages_image_id", "split"])
        w.writerows(rows)
    monkeypatch.setattr(split, "RAW_DIR", raw)
    monkeypatch.setattr(split, "SPLIT_DIR", out)
    monkeypatch.setattr("sys.argv", ["split.py"])
    split.main()

    seen = {}
    for p in out.rglob("*.jpg"):
        which, key = p.parts[-3], split.group_key(p)
        assert seen.setdefault(key, which) == which, f"{key} appears in two splits"
        if int(key.split("_")[1][len(p.parent.name):]) < 4:
            assert which == "test"
    assert (out / "split_summary.csv").exists()


def test_crop_and_resize():
    img = np.zeros((400, 600, 3), np.uint8)
    crop = crop_box(img, (0.1, 0.2, 0.5, 0.6), margin=0.1)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop_box(img, (0.5, 0.5, 0.5, 0.5), margin=0) is None
    assert max(resize_max(np.zeros((1000, 800, 3), np.uint8), 512).shape[:2]) == 512


def test_metrics_perfect_and_imperfect():
    y = ["mug", "book", "phone", "mug"]
    m = compute_metrics(y, y, [0.9] * 4, 0.7)
    assert m["accuracy"] == 1.0
    m = compute_metrics(y, ["mug", "book", "mouse", "book"], [0.9, 0.8, 0.5, 0.6], 0.7)
    assert m["accuracy"] == 0.5
    assert m["below_threshold"] == 2
    assert set(m["per_class"]) == set(config.CLASSES)


def test_threshold_marks_uncertain_predictions():
    assert describe([("mug", 0.91)], 0.7)[0] == "mug"
    assert describe([("mug", 0.41)], 0.7)[0].startswith("uncertain")


def test_draw_panel_returns_same_size_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_panel(frame, [("mug", 0.8), ("bottle", 0.15), ("book", 0.05)], 0.7, fps=10)
    assert out.shape == (480, 640, 3)


MODEL = config.MODELS_DIR / "best.pt"


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model at models/best.pt")
def test_model_predicts_valid_probabilities():
    from classifier import OfficeItemClassifier
    clf = OfficeItemClassifier(MODEL)
    assert clf.names == config.CLASSES
    probs = clf.predict([np.full((224, 224, 3), 128, np.uint8)])[0]
    assert probs.shape == (len(config.CLASSES),)
    assert abs(probs.sum() - 1.0) < 1e-3
    name, conf = clf.top_k(probs, 1)[0]
    assert name in config.CLASSES and 0.0 <= conf <= 1.0


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model at models/best.pt")
def test_missing_model_gives_clear_error(tmp_path):
    from classifier import OfficeItemClassifier
    with pytest.raises(FileNotFoundError):
        OfficeItemClassifier(tmp_path / "nope.pt")
