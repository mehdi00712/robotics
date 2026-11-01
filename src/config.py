"""Shared settings for the office-item classifier.

Keeping the class list and folder paths in one place means every script
(capture, split, train, evaluate, predict) agrees on them.
"""
from pathlib import Path

# The seven office-item classes recognised by the classifier (alphabetical).
CLASSES = [
    "book",
    "bottle",
    "keyboard",
    "mouse",
    "mug",
    "phone",
    "stapler",
]

# Project folders (resolved relative to the repository root).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # images exactly as captured, one folder per class
SPLIT_DIR = DATA_DIR / "split"      # train / val / test copies made by split.py
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
