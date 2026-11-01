"""Fine-tune a pretrained YOLO11n-cls model on data/split.

Usage: python src/train.py [--epochs 30] [--batch 32] [--device cpu|0]

Saves the best weights to models/best.pt and training plots to results/train/.
"""
import argparse
import shutil
import sys

from config import MODELS_DIR, RESULTS_DIR, SPLIT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Train the office-item classifier.")
    p.add_argument("--model", default="yolo11n-cls.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    for needed in ("train", "val"):
        if not (SPLIT_DIR / needed).is_dir():
            sys.exit(f"Missing {SPLIT_DIR / needed}. Run python src/split.py first.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = dict(
        data=str(SPLIT_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(RESULTS_DIR),
        name="train",
        exist_ok=True,
        plots=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    model.train(**kwargs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model.trainer.best, MODELS_DIR / "best.pt")
    print(f"\nBest model saved to {MODELS_DIR / 'best.pt'}")
    print("Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
