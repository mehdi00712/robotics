"""Download office-item images from Open Images V7 and save one-object crops.

Open Images is a detection dataset (many objects per photo, with boxes).
The classifier needs ONE item per image, so this script crops each labelled
box (with a small margin) and saves the crop into data/raw/<class>/.

Usage (from the repository root, virtual environment active):

    pip install fiftyone
    python src/download_openimages.py --per-class 200
"""
import argparse
import csv
import sys

import cv2

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

# Boxes flagged with these Open Images attributes are skipped:
# group-of boxes contain several objects, depictions are drawings/posters.
SKIP_FLAGS = ("IsGroupOf", "IsDepiction")


def parse_args():
    p = argparse.ArgumentParser(description="Download Open Images crops for the office-item classes.")
    p.add_argument("--per-class", type=int, default=200,
                   help="Target number of crops per class (default 200).")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"],
                   help="Open Images split to download from (default train).")
    p.add_argument("--min-size", type=int, default=64,
                   help="Skip boxes smaller than this many pixels on the short side (default 64).")
    p.add_argument("--max-side", type=int, default=512,
                   help="Resize saved crops so the long side is at most this (default 512).")
    p.add_argument("--margin", type=float, default=0.10,
                   help="Extra margin around each box as a fraction of its size (default 0.10).")
    p.add_argument("--classes", nargs="*", default=list(CLASS_MAP),
                   help="Subset of our class names to download (default: all mapped classes).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for which images are chosen.")
    return p.parse_args()


def flag(det, name):
    """Read an Open Images attribute from a FiftyOne Detection (False if absent)."""
    try:
        return bool(det[name])
    except (KeyError, AttributeError):
        return bool(getattr(det, name, False))


def detections_field(dataset):
    """Find the name of the field that holds the detection boxes."""
    import fiftyone as fo
    fields = dataset.get_field_schema(embedded_doc_type=fo.Detections)
    if not fields:
        sys.exit("No detection field found in the downloaded dataset.")
    return "detections" if "detections" in fields else next(iter(fields))


def crop_box(img, box, margin):
    """Crop a relative [x, y, w, h] box with a margin; returns None if empty."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    x0 = int(max(0, (x - margin * bw) * w))
    y0 = int(max(0, (y - margin * bh) * h))
    x1 = int(min(w, (x + bw + margin * bw) * w))
    y1 = int(min(h, (y + bh + margin * bh) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    try:
        import fiftyone.zoo as foz
    except ImportError:
        sys.exit("FiftyOne is not installed. Run:  pip install fiftyone")

    unknown = [c for c in args.classes if c not in CLASS_MAP]
    if unknown:
        sys.exit(f"Not available in Open Images: {unknown}. Use your own photos for these.")

    sources_path = RAW_DIR / "openimages_sources.csv"
    new_file = not sources_path.exists()
    with open(sources_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["file", "class", "openimages_class", "openimages_image_id", "split"])

        for ours in args.classes:
            oi_name = CLASS_MAP[ours]
            print(f"\n=== {ours} (Open Images: '{oi_name}') ===")
            # Ask for more images than crops needed, because some boxes are skipped.
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=args.split,
                label_types=["detections"],
                classes=[oi_name],
                only_matching=True,
                max_samples=int(args.per_class * 1.5),
                shuffle=True,
                seed=args.seed,
                dataset_name=f"oi-{ours}-{args.split}",
            )
            field = detections_field(dataset)
            out_dir = RAW_DIR / ours
            out_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for sample in dataset:
                if saved >= args.per_class:
                    break
                dets = sample[field]
                if dets is None:
                    continue
                img = cv2.imread(sample.filepath)
                if img is None:
                    continue
                image_id = sample.get_field("open_images_id") if sample.has_field("open_images_id") else sample.filename.rsplit(".", 1)[0]
                for k, det in enumerate(dets.detections):
                    if det.label != oi_name or any(flag(det, f) for f in SKIP_FLAGS):
                        continue
                    crop = crop_box(img, det.bounding_box, args.margin)
                    if crop is None or min(crop.shape[:2]) < args.min_size:
                        continue
                    crop = resize_max(crop, args.max_side)
                    # name format matches capture.py: <class>_<room>_<session>_<id>.jpg
                    # room = "openimages", session = the source image id (keeps crops
                    # from the same photo together when the data is split).
                    name = f"{ours}_openimages_{image_id}_{k}.jpg"
                    cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    writer.writerow([f"{ours}/{name}", ours, oi_name, image_id, args.split])
                    saved += 1
                    if saved >= args.per_class:
                        break
            print(f"saved {saved} crops to {out_dir}")
            if saved < args.per_class:
                print(f"  only {saved} usable crops; re-run with a larger --per-class or another --split")

    print(f"\nSource list: {sources_path}")
    print("Next: look through the folders and delete any wrong or unclear crops.")


if __name__ == "__main__":
    main()