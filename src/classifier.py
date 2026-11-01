"""Small wrapper around the trained YOLO model, shared by evaluate.py and predict.py."""
from pathlib import Path

import numpy as np

from config import MODELS_DIR

DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"


class OfficeItemClassifier:
    def __init__(self, weights=DEFAULT_WEIGHTS, imgsz=224, device=None):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Model not found: {weights}. Train it first with  python src/train.py")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        self.imgsz = imgsz
        self.device = device

    def predict(self, images):
        """images: list of file paths or BGR numpy arrays -> list of probability vectors."""
        kwargs = dict(imgsz=self.imgsz, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(images, **kwargs)
        return [r.probs.data.cpu().numpy().astype(float) for r in results]

    def top_k(self, probs, k=3):
        order = np.argsort(probs)[::-1][:k]
        return [(self.names[i], float(probs[i])) for i in order]
