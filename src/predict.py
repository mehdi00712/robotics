"""Recognise an office item from an image file, a folder of images, or a live camera.

Usage:
    python src/predict.py --image photo.jpg [more.jpg ...]
    python src/predict.py --folder path/to/images      # also saves predictions.csv there
    python src/predict.py --camera 0                   # live webcam

Webcam keys:  Q/Esc quit   S save snapshot   P pause/resume
Predictions below --threshold (default 0.70) are shown as "uncertain".
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import IMAGE_EXTENSIONS, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Classify office items from images or a webcam.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", nargs="+", help="One or more image files.")
    src.add_argument("--folder", help="Folder of images.")
    src.add_argument("--camera", type=int, help="Camera index, usually 0.")
    p.add_argument("--weights", default=None, help="Model weights (default models/best.pt).")
    p.add_argument("--threshold", type=float, default=0.70, help="Minimum confidence for a definite answer.")
    p.add_argument("--top", type=int, default=3, help="How many top classes to show.")
    p.add_argument("--show", action="store_true", help="Also display each image with its label (image/folder mode).")
    return p.parse_args()


def describe(top, threshold):
    name, conf = top[0]
    verdict = name if conf >= threshold else f"uncertain (best guess: {name})"
    return verdict, conf


def draw_panel(frame, top, threshold, fps=None):
    """Draw the prediction and a top-k bar chart on the frame (in place)."""
    h, w = frame.shape[:2]
    name, conf = top[0]
    ok = conf >= threshold
    colour = (60, 180, 75) if ok else (0, 165, 255)
    label = f"{name}  {conf * 100:.0f}%" if ok else f"uncertain ({name} {conf * 100:.0f}%)"

    overlay = frame.copy()
    panel_h = 50 + 26 * len(top)
    cv2.rectangle(overlay, (0, 0), (min(w, 360), panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    for i, (n, c) in enumerate(top):
        y = 50 + 26 * i
        cv2.rectangle(frame, (110, y + 4), (110 + int(220 * c), y + 20), colour if i == 0 else (150, 150, 150), -1)
        cv2.putText(frame, f"{n:<9}", (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{c * 100:.0f}%", (115 + int(220 * c), y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} fps   Q quit  S save  P pause", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def classify_files(clf, paths, args, csv_path=None):
    rows = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path}: could not read image, skipped")
            continue
        top = clf.top_k(clf.predict([img])[0], args.top)
        verdict, conf = describe(top, args.threshold)
        others = ", ".join(f"{n} {c:.2f}" for n, c in top[1:])
        print(f"{path.name}: {verdict} ({conf:.2f})" + (f"   [next: {others}]" if others else ""))
        rows.append([str(path), top[0][0], f"{conf:.4f}", int(conf >= args.threshold)])
        if args.show:
            cv2.imshow("Prediction (any key = next, Q = quit)", draw_panel(img.copy(), top, args.threshold))
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    if args.show:
        cv2.destroyAllWindows()
    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "predicted", "confidence", "confident"])
            w.writerows(rows)
        print(f"\nSaved {len(rows)} predictions to {csv_path}")


def run_camera(clf, index, args):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try --camera 1 or close other apps using the webcam.")
    snap_dir = RESULTS_DIR / "snapshots"
    window = "Office item classifier"
    paused, frame, top, last = False, None, None, time.perf_counter()
    fps = 0.0
    print("Camera running. Q/Esc quit, S save snapshot, P pause.")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break
            top = clf.top_k(clf.predict([frame])[0], args.top)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
            last = now
        shown = draw_panel(frame.copy(), top, args.threshold, fps)
        cv2.imshow(window, shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"{top[0][0]}_{datetime.now():%Y%m%d-%H%M%S}.jpg"
            cv2.imwrite(str(path), shown)
            print(f"saved {path}")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    from classifier import DEFAULT_WEIGHTS, OfficeItemClassifier
    try:
        clf = OfficeItemClassifier(args.weights or DEFAULT_WEIGHTS)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    if args.camera is not None:
        run_camera(clf, args.camera, args)
    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            sys.exit(f"Folder not found: {folder}")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            sys.exit(f"No images in {folder}")
        classify_files(clf, paths, args, csv_path=folder / "predictions.csv")
    else:
        paths = [Path(p) for p in args.image]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"File not found: {', '.join(map(str, missing))}")
        classify_files(clf, paths, args)


if __name__ == "__main__":
    main()
