"""Evaluate the trained model on the held-out test set.

Usage:
    python src/evaluate.py                          # data/split/test
    python src/evaluate.py --data path/to/folder    # any folder laid out as <class>/*.jpg

Outputs (in results/ or results/<name>/ when --name is given):
    metrics.json                 accuracy, macro-F1, per-class precision/recall/F1, timing
    classification_report.txt
    confusion_matrix.png         counts
    confusion_matrix_norm.png    row-normalised (recall per class)
    predictions.csv              one row per test image
    errors/<true>_as_<pred>/     copies of misclassified images for error analysis
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from config import CLASSES, IMAGE_EXTENSIONS, RESULTS_DIR, SPLIT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the office-item classifier.")
    p.add_argument("--data", default=str(SPLIT_DIR / "test"), help="Folder with <class>/<image> sub-folders.")
    p.add_argument("--weights", default=None, help="Model weights (default models/best.pt).")
    p.add_argument("--name", default="", help="Sub-folder of results/ for this run, e.g. 'unseen_room'.")
    p.add_argument("--threshold", type=float, default=0.70, help="Confidence below this counts as 'uncertain'.")
    p.add_argument("--batch", type=int, default=32)
    return p.parse_args()


def collect(data_dir):
    items = []
    for cls in CLASSES:
        folder = data_dir / cls
        if folder.is_dir():
            items += [(p, cls) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return items


def plot_cm(cm, labels, path, normalise, title):
    data = cm.astype(float)
    if normalise:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    thresh = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{data[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_metrics(y_true, y_pred, confs, threshold):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return {
        "num_images": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0),
        "below_threshold": int(sum(c < threshold for c in confs)),
        "threshold": threshold,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
    }


def main():
    args = parse_args()
    data_dir = Path(args.data)
    items = collect(data_dir)
    if not items:
        sys.exit(f"No images found in {data_dir} (expected {data_dir}/<class>/*.jpg).")

    from classifier import DEFAULT_WEIGHTS, OfficeItemClassifier
    try:
        clf = OfficeItemClassifier(args.weights or DEFAULT_WEIGHTS)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    out_dir = RESULTS_DIR / args.name if args.name else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    err_dir = out_dir / "errors"
    if err_dir.exists():
        shutil.rmtree(err_dir)

    print(f"Evaluating {len(items)} images from {data_dir} ...")
    y_true, y_pred, confs, rows = [], [], [], []
    start = time.perf_counter()
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        probs = clf.predict([str(p) for p, _ in chunk])
        for (path, true), pr in zip(chunk, probs):
            pred, conf = clf.top_k(pr, 1)[0]
            y_true.append(true)
            y_pred.append(pred)
            confs.append(conf)
            rows.append([str(path), true, pred, f"{conf:.4f}", int(pred == true)])
            if pred != true:
                dest = err_dir / f"{true}_as_{pred}"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest / f"{conf:.2f}_{path.name}")
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(y_true, y_pred, confs, args.threshold)
    metrics["ms_per_image"] = 1000 * elapsed / len(items)
    metrics["data"] = str(data_dir)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    report = classification_report(y_true, y_pred, labels=CLASSES, digits=3, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true", "predicted", "confidence", "correct"])
        w.writerows(rows)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    plot_cm(cm, CLASSES, out_dir / "confusion_matrix.png", False, "Confusion matrix (counts)")
    plot_cm(cm, CLASSES, out_dir / "confusion_matrix_norm.png", True, "Confusion matrix (row-normalised)")

    print("\n" + report)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.3f}")
    print(f"Below {args.threshold:.2f} confidence: {metrics['below_threshold']} of {len(items)}")
    print(f"Speed:     {metrics['ms_per_image']:.1f} ms per image")
    print(f"Misclassified: {sum(t != p for t, p in zip(y_true, y_pred))} (copies in {err_dir})")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
