"""Download office-item images from Open Images V7 and save one-object crops.

Open Images is a detection dataset (many objects per photo, with boxes).
The classifier needs ONE item per image, so this script crops the largest
box of each class from each photo (with a small margin) and saves the crop
into data/raw/<class>/.

Usage (from the repository root, virtual environment active):

    python src/download_openimages.py --per-class 200 --split train
    python src/download_openimages.py --per-class 50  --split test

How it works (light on memory):
1. Gets the Open Images box-label CSV for the chosen split (reused if it is
   already on disk, e.g. from an earlier FiftyOne download).
2. Reads it LINE BY LINE, keeping only boxes of our 7 classes that are not
   "group of" boxes or drawings, and are not tiny.
3. Picks a random set of photos per class, downloads only those photos from
   the public Open Images image store, crops them and saves the crops.

"notebook" does not exist in Open Images, so it is not part of this dataset.
Every saved crop is listed in data/raw/openimages_sources.csv with its
Open Images image id and split, so the dataset card can credit the source
(images: CC BY 2.0, annotations: CC BY 4.0) and split.py can keep the
test split separate.
"""
import argparse
import csv
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from config import RAW_DIR

# Our class name -> Open Images V7 class name
CLASS_MAP = {
    "book": "Book",
    "bottle": "Bottle",
    "keyboard": "Computer keyboard",
    "mouse": "Computer mouse",
    "mug": "Mug",
    "phone": "Mobile phone",
    "stapler": "Stapler",
}

CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
BOXES_URL = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

CACHE_DIR = Path.home() / "openimages_cache"                     # big CSVs live here, not in the repo
FIFTYONE_DIR = Path.home() / "fiftyone" / "open-images-v7"       # reuse files FiftyOne already fetched


def parse_args():
    p = argparse.ArgumentParser(description="Download Open Images crops for the office-item classes.")
    p.add_argument("--per-class", type=int, default=200, help="Number of crops per class (default 200).")
    p.add_argument("--split", default="train", choices=list(BOXES_URL),
                   help="Open Images split to download from (default train).")
    p.add_argument("--classes", nargs="*", default=list(CLASS_MAP),
                   help="Subset of our class names to download (default: all).")
    p.add_argument("--min-area", type=float, default=0.02,
                   help="Skip boxes covering less than this fraction of the photo (default 0.02).")
    p.add_argument("--min-size", type=int, default=64,
                   help="Skip crops smaller than this many pixels on the short side (default 64).")
    p.add_argument("--max-side", type=int, default=512, help="Resize crops so the long side is at most this.")
    p.add_argument("--margin", type=float, default=0.10, help="Margin around each box (fraction of its size).")
    p.add_argument("--threads", type=int, default=8, help="Parallel image downloads (default 8).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for which photos are chosen.")
    return p.parse_args()


# ---------------------------------------------------------------- helpers
def download_file(url, dest):
    """Download a (possibly large) file with a simple progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def find_or_download(candidates, url, cache_name):
    """Return the first existing file from candidates, else download into the cache."""
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            print(f"Using existing file {c}")
            return c
    dest = CACHE_DIR / cache_name
    if not dest.exists():
        download_file(url, dest)
    return dest


def load_class_ids(path, wanted_names):
    """Map Open Images display name -> label id (e.g. 'Mug' -> '/m/02jvh9')."""
    ids = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[1] in wanted_names:
                ids[row[1]] = row[0]
    missing = set(wanted_names) - set(ids)
    if missing:
        sys.exit(f"Classes not found in Open Images: {sorted(missing)}")
    return ids


def scan_boxes(path, id_to_ours, min_area):
    """Stream the box CSV and return {our_class: {image_id: best_box}}.

    best_box = (xmin, ymin, xmax, ymax) in relative coordinates, the largest
    usable box of that class in that photo.
    """
    found = defaultdict(dict)
    rows = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            if rows % 2_000_000 == 0:
                print(f"  scanned {rows / 1e6:.0f}M boxes...", flush=True)
            ours = id_to_ours.get(row["LabelName"])
            if ours is None:
                continue
            if row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1":
                continue
            x0, x1 = float(row["XMin"]), float(row["XMax"])
            y0, y1 = float(row["YMin"]), float(row["YMax"])
            area = (x1 - x0) * (y1 - y0)
            if area < min_area:
                continue
            prev = found[ours].get(row["ImageID"])
            if prev is None or area > (prev[2] - prev[0]) * (prev[3] - prev[1]):
                found[ours][row["ImageID"]] = (x0, y0, x1, y1)
    print(f"  scanned {rows:,} boxes in total")
    return found


def fetch_image(split, image_id):
    """Download one Open Images photo and decode it (None if it fails)."""
    url = IMAGE_URL.format(split=split, image_id=image_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = np.frombuffer(resp.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def crop_box(img, box, margin):
    """Crop a relative (xmin, ymin, xmax, ymax) box with a margin."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    a, b = int(max(0, (x0 - mx) * w)), int(max(0, (y0 - my) * h))
    c, d = int(min(w, (x1 + mx) * w)), int(min(h, (y1 + my) * h))
    if c <= a or d <= b:
        return None
    return img[b:d, a:c]


def resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    unknown = [c for c in args.classes if c not in CLASS_MAP]
    if unknown:
        sys.exit(f"Not available in Open Images: {unknown}")
    wanted = {CLASS_MAP[c]: c for c in args.classes}           # Open Images name -> ours

    fo_dir = FIFTYONE_DIR / args.split
    classes_csv = find_or_download([fo_dir / "metadata" / "classes.csv"], CLASSES_URL, "classes.csv")
    boxes_csv = find_or_download([fo_dir / "labels" / "detections.csv"], BOXES_URL[args.split],
                                 f"{args.split}-boxes.csv")

    name_to_id = load_class_ids(classes_csv, list(wanted))
    id_to_ours = {name_to_id[n]: ours for n, ours in wanted.items()}
    print(f"Scanning {boxes_csv} for {len(id_to_ours)} classes (this takes a few minutes for train)...")
    found = scan_boxes(boxes_csv, id_to_ours, args.min_area)

    # Skip photos that were already saved by an earlier run.
    sources_path = RAW_DIR / "openimages_sources.csv"
    done_ids, split_counts = set(), defaultdict(int)
    if sources_path.exists():
        with open(sources_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done_ids.add((r["class"], r["openimages_image_id"]))   # never save the same photo twice
                if r["split"] == args.split:
                    split_counts[r["class"]] += 1                       # crops already saved from THIS split

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    new_file = not sources_path.exists()
    with open(sources_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["file", "class", "openimages_class", "openimages_image_id", "split"])

        for ours in args.classes:
            candidates = sorted(found.get(ours, {}).items())
            rng.shuffle(candidates)
            already = split_counts[ours]
            need = args.per_class - already
            print(f"\n=== {ours}: {len(candidates)} usable photos available, {already} already saved from this split ===")
            if need <= 0:
                continue
            out_dir = RAW_DIR / ours
            out_dir.mkdir(parents=True, exist_ok=True)
            todo = [(iid, box) for iid, box in candidates if (ours, iid) not in done_ids]

            saved, pos = 0, 0
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                while saved < need and pos < len(todo):
                    batch = todo[pos:pos + (need - saved) * 2]      # fetch a few spare in case some fail
                    pos += len(batch)
                    futures = {pool.submit(fetch_image, args.split, iid): (iid, box) for iid, box in batch}
                    for fut in as_completed(futures):
                        iid, box = futures[fut]
                        img = fut.result()
                        if img is None or saved >= need:
                            continue
                        crop = crop_box(img, box, args.margin)
                        if crop is None or min(crop.shape[:2]) < args.min_size:
                            continue
                        name = f"{ours}_openimages_{iid}_0.jpg"
                        cv2.imwrite(str(out_dir / name), resize_max(crop, args.max_side),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                        writer.writerow([f"{ours}/{name}", ours, CLASS_MAP[ours], iid, args.split])
                        fh.flush()
                        saved += 1
                        if saved % 25 == 0:
                            print(f"  {saved}/{need}", flush=True)
            print(f"saved {saved} new crops to {out_dir}")
            if saved < need:
                print(f"  only {saved} of {need}: not enough usable photos in the '{args.split}' split")

    print(f"\nSource list: {sources_path}")
    print("Next: look through data/raw/<class>/ and delete any wrong or unclear crops.")


if __name__ == "__main__":
    main()