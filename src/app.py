"""Desktop app for the office-item classifier.

Usage:
    python src/app.py                      # uses models/best.pt
    python src/app.py --weights models/best_v1_yolo11n.pt

Buttons: open an image, open a folder (step through with Prev/Next or the arrow
keys), start/stop the webcam, save a snapshot. The threshold slider sets the
confidence below which a prediction is shown as "uncertain".
Keys: O open image, F open folder, C camera on/off, S snapshot, Left/Right browse, Esc quit.
"""
import argparse
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from config import CLASSES, IMAGE_EXTENSIONS, RESULTS_DIR

PREVIEW_W, PREVIEW_H = 640, 480
GREEN, ORANGE, GREY = "#2e9e44", "#e08a00", "#9a9a9a"


def fit_to_box(frame, max_w=PREVIEW_W, max_h=PREVIEW_H):
    """Resize a BGR image to fit inside max_w x max_h, keeping its aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


class App:
    def __init__(self, root, weights, camera_index):
        self.root = root
        self.weights = Path(weights)
        self.clf = None
        self.cap = None
        self.running = False
        self.paths, self.index = [], 0
        self.last_frame, self.last_top = None, None
        self.fps, self.last_time = 0.0, None

        root.title("Office Item Classifier")
        root.configure(bg="#f4f4f4")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Top.Horizontal.TProgressbar", background=GREEN, troughcolor="#e2e2e2")
        style.configure("Other.Horizontal.TProgressbar", background=GREY, troughcolor="#e2e2e2")

        # left: image preview
        self.preview = tk.Label(root, bg="#202020", width=PREVIEW_W, height=PREVIEW_H)
        self.preview.grid(row=0, column=0, padx=12, pady=12, sticky="n")
        self.blank = ImageTk.PhotoImage(Image.new("RGB", (PREVIEW_W, PREVIEW_H), (32, 32, 32)))
        self.preview.configure(image=self.blank)

        # right: result + controls
        side = tk.Frame(root, bg="#f4f4f4")
        side.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="n")

        tk.Label(side, text="Prediction", bg="#f4f4f4", fg="#555", font=("Segoe UI", 10)).pack(anchor="w")
        self.result = tk.Label(side, text="–", bg="#f4f4f4", font=("Segoe UI", 26, "bold"), width=14, anchor="w")
        self.result.pack(anchor="w")
        self.detail = tk.Label(side, text="Open an image or start the camera", bg="#f4f4f4", fg="#444",
                               font=("Segoe UI", 10), anchor="w", justify="left")
        self.detail.pack(anchor="w", pady=(0, 10))

        self.bars = []
        for _ in range(3):
            row = tk.Frame(side, bg="#f4f4f4")
            row.pack(anchor="w", pady=2)
            name = tk.Label(row, text="", width=9, anchor="w", bg="#f4f4f4", font=("Segoe UI", 10))
            name.pack(side="left")
            bar = ttk.Progressbar(row, length=170, maximum=100, style="Other.Horizontal.TProgressbar")
            bar.pack(side="left", padx=4)
            pct = tk.Label(row, text="", width=5, anchor="e", bg="#f4f4f4", font=("Segoe UI", 10))
            pct.pack(side="left")
            self.bars.append((name, bar, pct))

        ttk.Separator(side).pack(fill="x", pady=10)
        buttons = [("Open image…  (O)", self.open_image), ("Open folder…  (F)", self.open_folder)]
        for text, cmd in buttons:
            ttk.Button(side, text=text, command=cmd, width=26).pack(anchor="w", pady=2)

        nav = tk.Frame(side, bg="#f4f4f4")
        nav.pack(anchor="w", pady=2)
        ttk.Button(nav, text="◀ Prev", command=lambda: self.step(-1), width=12).pack(side="left")
        ttk.Button(nav, text="Next ▶", command=lambda: self.step(1), width=12).pack(side="left", padx=(4, 0))
        self.position = tk.Label(side, text="", bg="#f4f4f4", fg="#555", font=("Segoe UI", 9))
        self.position.pack(anchor="w")

        cam = tk.Frame(side, bg="#f4f4f4")
        cam.pack(anchor="w", pady=(8, 2))
        self.cam_button = ttk.Button(cam, text="Start camera  (C)", command=self.toggle_camera, width=18)
        self.cam_button.pack(side="left")
        tk.Label(cam, text="index", bg="#f4f4f4").pack(side="left", padx=(6, 2))
        self.cam_index = tk.Spinbox(cam, from_=0, to=5, width=3)
        self.cam_index.delete(0, "end")
        self.cam_index.insert(0, str(camera_index))
        self.cam_index.pack(side="left")
        ttk.Button(side, text="Save snapshot  (S)", command=self.snapshot, width=26).pack(anchor="w", pady=2)

        ttk.Separator(side).pack(fill="x", pady=10)
        tk.Label(side, text="Uncertain below confidence:", bg="#f4f4f4").pack(anchor="w")
        self.threshold = tk.DoubleVar(value=0.70)
        tk.Scale(side, from_=0.30, to=0.95, resolution=0.05, orient="horizontal", length=230,
                 variable=self.threshold, bg="#f4f4f4", highlightthickness=0,
                 command=lambda _v: self.show_result()).pack(anchor="w")
        tk.Label(side, text="Classes: " + ", ".join(CLASSES), bg="#f4f4f4", fg="#555",
                 wraplength=240, justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))

        self.status = tk.Label(root, text="Loading model…", anchor="w", bg="#e6e6e6", fg="#333", padx=8)
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

        for key, fn in (("o", self.open_image), ("f", self.open_folder), ("c", self.toggle_camera),
                        ("s", self.snapshot)):
            root.bind(f"<Key-{key}>", lambda _e, fn=fn: fn())
        root.bind("<Left>", lambda _e: self.step(-1))
        root.bind("<Right>", lambda _e: self.step(1))
        root.bind("<Escape>", lambda _e: self.close())
        root.after(100, self.load_model)

    # ------------------------------------------------------------------ model
    def load_model(self):
        from classifier import OfficeItemClassifier
        try:
            self.clf = OfficeItemClassifier(self.weights)
        except Exception as exc:  # missing file or broken install
            self.status.configure(text=f"Could not load model: {exc}")
            messagebox.showerror("Model not loaded", str(exc))
            return
        self.status.configure(text=f"Model: {self.weights.name}   |   ready")

    def classify(self, frame):
        if self.clf is None:
            self.status.configure(text="Model is not loaded yet.")
            return
        start = time.perf_counter()
        probs = self.clf.predict([frame])[0]
        ms = (time.perf_counter() - start) * 1000
        self.last_frame, self.last_top = frame, self.clf.top_k(probs, 3)
        self.show_image(frame)
        self.show_result()
        info = f"Model: {self.weights.name}   |   {ms:.0f} ms per prediction"
        if self.running and self.fps:
            info += f"   |   {self.fps:.1f} fps"
        self.status.configure(text=info)

    # --------------------------------------------------------------- display
    def show_image(self, frame):
        rgb = cv2.cvtColor(fit_to_box(frame), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))  # keep a reference or Tk drops it
        self.preview.configure(image=self.photo)

    def show_result(self):
        if not self.last_top:
            return
        name, conf = self.last_top[0]
        if conf >= self.threshold.get():
            self.result.configure(text=name, fg=GREEN)
            self.detail.configure(text=f"{conf * 100:.0f}% confident")
        else:
            self.result.configure(text="uncertain", fg=ORANGE)
            self.detail.configure(text=f"best guess: {name} ({conf * 100:.0f}%)\nhold the item closer or improve the lighting")
        for i, (lbl, bar, pct) in enumerate(self.bars):
            n, c = self.last_top[i]
            lbl.configure(text=n)
            bar.configure(value=c * 100, style="Top.Horizontal.TProgressbar" if i == 0 else "Other.Horizontal.TProgressbar")
            pct.configure(text=f"{c * 100:.0f}%")

    # ----------------------------------------------------------------- files
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self.stop_camera()
            self.paths, self.index = [Path(path)], 0
            self.show_path()

    def open_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder of images")
        if not folder:
            return
        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            messagebox.showinfo("No images", "That folder has no .jpg/.png/.bmp/.webp images.")
            return
        self.stop_camera()
        self.paths, self.index = paths, 0
        self.show_path()

    def step(self, delta):
        if self.running or len(self.paths) < 2:
            return
        self.index = (self.index + delta) % len(self.paths)
        self.show_path()

    def show_path(self):
        path = self.paths[self.index]
        frame = cv2.imread(str(path))
        if frame is None:
            messagebox.showerror("Cannot open image", f"Could not read {path.name}")
            return
        self.position.configure(text=f"{self.index + 1} / {len(self.paths)}:  {path.name}")
        self.classify(frame)

    # ---------------------------------------------------------------- camera
    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        try:
            idx = int(self.cam_index.get())
        except ValueError:
            idx = 0
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(idx, backend)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Camera", f"Could not open camera {idx}.\nTry another index or close apps using the webcam.")
            return
        self.running, self.last_time, self.fps = True, None, 0.0
        self.cam_button.configure(text="Stop camera  (C)")
        self.position.configure(text=f"Live camera {idx}")
        self.camera_loop()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cam_button.configure(text="Start camera  (C)")

    def camera_loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            now = time.perf_counter()
            if self.last_time is not None:
                inst = 1.0 / max(now - self.last_time, 1e-6)
                self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
            self.last_time = now
            self.classify(frame)
        self.root.after(10, self.camera_loop)

    # -------------------------------------------------------------- snapshot
    def snapshot(self):
        if self.last_frame is None or not self.last_top:
            messagebox.showinfo("Snapshot", "Nothing to save yet.")
            return
        from predict import draw_panel
        out = draw_panel(self.last_frame.copy(), self.last_top, self.threshold.get())
        folder = RESULTS_DIR / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.last_top[0][0]}_{datetime.now():%Y%m%d-%H%M%S}.jpg"
        cv2.imwrite(str(path), out)
        self.status.configure(text=f"Saved {path}")

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    from classifier import DEFAULT_WEIGHTS
    p = argparse.ArgumentParser(description="Desktop app for the office-item classifier.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Model weights (default models/best.pt).")
    p.add_argument("--camera", type=int, default=0, help="Default camera index.")
    args = p.parse_args()
    root = tk.Tk()
    App(root, args.weights, args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()