"""Webcam tool for collecting the office-item dataset.

Usage (from the repository root, with the virtual environment active):

    python src/capture.py --room bedroom --session s01
    python src/capture.py --room office --session s02 --camera 1

Controls in the camera window:
    1-8      choose the class you are photographing (shown on screen)
    SPACE    save one photo of the current frame
    B        start/stop burst mode (saves a photo every --interval seconds,
             move the object or camera slowly between shots)
    Q / ESC  quit

Each photo is saved as
    data/raw/<class>/<class>_<room>_<session>_<timestamp>.jpg
The room and session are kept in the file name so split.py can keep every
photo from one session in the same split (prevents near-duplicate leakage).
"""
import argparse
import re
import sys
import time
from datetime import datetime

import cv2

from config import CLASSES, RAW_DIR, IMAGE_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Capture labelled webcam photos of office items.")
    parser.add_argument("--room", required=True,
                        help="Where you are shooting, e.g. bedroom, office, library (letters/numbers only).")
    parser.add_argument("--session", required=True,
                        help="Session id, e.g. s01. Use a new id for each sitting or new lighting.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--interval", type=float, default=0.7,
                        help="Seconds between photos in burst mode (default 0.7).")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    return parser.parse_args()


def clean_tag(text):
    """Keep tags safe for file names: lower case, letters/numbers/hyphens only."""
    tag = re.sub(r"[^a-z0-9-]", "", text.lower())
    if not tag:
        sys.exit(f"Invalid tag '{text}': use letters, numbers or hyphens.")
    return tag


def count_images():
    """Return how many images already exist for each class."""
    counts = {}
    for name in CLASSES:
        folder = RAW_DIR / name
        counts[name] = sum(1 for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.exists() else 0
    return counts


def save_frame(frame, class_name, room, session):
    folder = RAW_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = folder / f"{class_name}_{room}_{session}_{stamp}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def draw_overlay(display, current, counts, burst, room, session):
    """Draw the class list, counts and status on the preview (not on saved photos)."""
    h, w = display.shape[:2]
    panel_w = 250
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, 40 + 26 * len(CLASSES) + 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    cv2.putText(display, f"room={room}  session={session}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(CLASSES):
        y = 55 + 26 * i
        selected = name == current
        colour = (0, 255, 255) if selected else (220, 220, 220)
        marker = ">" if selected else " "
        cv2.putText(display, f"{marker} {i + 1}. {name:<9} {counts[name]:>4}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2 if selected else 1, cv2.LINE_AA)

    y = 55 + 26 * len(CLASSES) + 10
    status = "BURST ON (B to stop)" if burst else "SPACE=save  B=burst  Q=quit"
    cv2.putText(display, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255) if burst else (200, 255, 200), 1, cv2.LINE_AA)
    if current is None:
        cv2.putText(display, "Press 1-8 to choose a class", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    room, session = clean_tag(args.room), clean_tag(args.session)

    # CAP_DSHOW opens most Windows webcams faster; fall back to the default backend elsewhere.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1, or close other apps using the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    counts = count_images()
    current, burst, last_shot, flash_until = None, False, 0.0, 0.0
    window = "Office item capture"
    print("Camera open. Press 1-8 to pick a class, SPACE to save, B for burst, Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed; stopping.")
            break

        now = time.time()
        if burst and current and now - last_shot >= args.interval:
            path = save_frame(frame, current, room, session)
            counts[current] += 1
            last_shot, flash_until = now, now + 0.15
            print(f"saved {path.name}")

        display = frame.copy()
        draw_overlay(display, current, counts, burst, room, session)
        if now < flash_until:  # white border = photo taken
            cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), (255, 255, 255), 8)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord(str(len(CLASSES))):
            current = CLASSES[key - ord("1")]
            burst = False
            print(f"class -> {current}")
        elif key == ord(" "):
            if current is None:
                print("Choose a class first (keys 1-8).")
            else:
                path = save_frame(frame, current, room, session)
                counts[current] += 1
                flash_until = now + 0.15
                print(f"saved {path.name}")
        elif key == ord("b"):
            if current is None:
                print("Choose a class first (keys 1-8).")
            else:
                burst = not burst
                print("burst on" if burst else "burst off")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the X button
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nImages per class:")
    for name in CLASSES:
        print(f"  {name:<9} {counts[name]}")


if __name__ == "__main__":
    main()
