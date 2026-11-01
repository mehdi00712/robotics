"""Split data/raw into data/split/{train,val,test}/<class>/ for YOLO classification.

Usage: python src/split.py [--val 0.15] [--test 0.15] [--seed 42]

Open Images crops from the 'test' split go to test. Everything else is split
by group (same source photo / same capture session) so near-duplicates never
end up in both training and test.
"""
import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict

from config import CLASSES, IMAGE_EXTENSIONS, RAW_DIR, SPLIT_DIR

SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Create train/val/test folders from data/raw.")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_openimages_splits():
    path = RAW_DIR / "openimages_sources.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["file"].replace("\\", "/"): r["split"] for r in csv.DictReader(fh)}


def group_key(path):
    # file names look like <class>_<room>_<session>_<id>.jpg
    parts = path.stem.split("_")
    return f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else path.stem


def assign_groups(groups, fractions, rng):
    groups = sorted(groups)
    rng.shuffle(groups)
    out, start, names = {}, 0, list(fractions)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            chunk = groups[start:]
        else:
            n = round(len(groups) * fractions[name])
            if fractions[name] > 0 and n == 0 and len(groups) - start > 1:
                n = 1
            chunk = groups[start:start + n]
            start += n
        for g in chunk:
            out[g] = name
    return out


def main():
    args = parse_args()
    if args.val + args.test >= 1:
        sys.exit("--val + --test must be less than 1")
    rng = random.Random(args.seed)
    oi_split = load_openimages_splits()

    plan = {}
    for cls in CLASSES:
        folder = RAW_DIR / cls
        images = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"warning: no images for '{cls}'")
            continue
        fixed_test, pool = [], defaultdict(list)
        for img in images:
            if oi_split.get(f"{cls}/{img.name}") == "test":
                fixed_test.append(img)
            else:
                pool[group_key(img)].append(img)
        for img in fixed_test:
            plan[img] = "test"
        if fixed_test:
            fractions = {"val": args.val, "train": 1 - args.val}
        else:
            fractions = {"test": args.test, "val": args.val, "train": 1 - args.val - args.test}
        groups = assign_groups(pool.keys(), fractions, rng)
        for g, imgs in pool.items():
            for img in imgs:
                plan[img] = groups[g]

    if not plan:
        sys.exit("No images found in data/raw.")

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    counts = defaultdict(lambda: defaultdict(int))
    for img, split in plan.items():
        dest = SPLIT_DIR / split / img.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / img.name)
        counts[img.parent.name][split] += 1

    with open(SPLIT_DIR / "split_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", *SPLITS, "total"])
        print(f"\n{'class':<10}{'train':>7}{'val':>7}{'test':>7}{'total':>7}")
        totals = defaultdict(int)
        for cls in CLASSES:
            row = [counts[cls][s] for s in SPLITS]
            w.writerow([cls, *row, sum(row)])
            print(f"{cls:<10}" + "".join(f"{n:>7}" for n in row) + f"{sum(row):>7}")
            for s, n in zip(SPLITS, row):
                totals[s] += n
        all_row = [totals[s] for s in SPLITS]
        w.writerow(["all", *all_row, sum(all_row)])
        print(f"{'all':<10}" + "".join(f"{n:>7}" for n in all_row) + f"{sum(all_row):>7}")
    print(f"\nSplit written to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
