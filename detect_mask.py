"""
detect_mask.py - Real-Time Face Mask Detection Inference Script
Supports webcam, video, and image sources.

Usage:
  Webcam:        python detect_mask.py
  Other camera:  python detect_mask.py --source 1
  Image:         python detect_mask.py --source test.jpg
  Video:         python detect_mask.py --source sample.mp4
  Debug mode:    python detect_mask.py --debug
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH   = Path("models/face_mask_model.h5")
CLASSES_PATH = Path("models/classes.json")
IMG_SIZE     = (128, 128)

# ── Fonts & colours ────────────────────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_DUPLEX
COLOR_MASK      = (0,  200,  0)      # green  – with_mask
COLOR_NO_MASK   = (0,   0, 220)      # red    – without_mask
COLOR_INFO      = (220, 220,  0)     # yellow – status text
COLOR_WHITE     = (255, 255, 255)
COLOR_DARK_GRAY = ( 30,  30,  30)


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Real-Time Face Mask Detection using CNN + OpenCV Haar Cascade."
    )
    p.add_argument(
        "--source", default="0",
        help="Camera index (0, 1 ...), image path (.jpg/.png) or video path (.mp4/.avi). Default: 0"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Sigmoid confidence threshold for classification. Default: 0.5"
    )
    p.add_argument(
        "--scale", type=float, default=1.05,
        help="Haar scaleFactor. Smaller = more sensitive (slower). Default: 1.05"
    )
    p.add_argument(
        "--neighbors", type=int, default=4,
        help="Haar minNeighbors. Smaller = more detections (more false positives). Default: 4"
    )
    p.add_argument(
        "--min-face", type=int, default=40,
        help="Minimum face size in pixels. Default: 40"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Overlay debug info (raw prob, face count, FPS)."
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Skip GUI window for image mode (saves annotated_image.jpg only)."
    )
    return p.parse_args()


def load_model_and_classes():
    for path, label in [(MODEL_PATH, "Model"), (CLASSES_PATH, "Classes file")]:
        if not path.exists():
            print(f"[ERROR] {label} not found at '{path}'.", file=sys.stderr)
            print("        Run  python train.py  first.", file=sys.stderr)
            sys.exit(1)

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print(f"[OK] Loaded model : {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Loading model failed: {e}", file=sys.stderr)
        sys.exit(1)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    print(f"[OK] Class labels  : {classes}")
    return model, classes


def draw_label(frame, text, x, y, color, font_scale=0.60, thickness=1):
    """Draw white text on a filled colour banner."""
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    by = max(y - 2, th + 6)
    cv2.rectangle(frame, (x, by - th - 4), (x + tw + 6, by + bl), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, by - 1), FONT, font_scale, COLOR_WHITE, thickness, cv2.LINE_AA)


def draw_status_bar(frame, face_count, fps, debug):
    """Semi-transparent top status bar with face count and FPS."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), COLOR_DARK_GRAY, cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if face_count == 0:
        status = "No face detected - move closer or improve lighting"
        col = COLOR_INFO
    else:
        status = f"{face_count} face{'s' if face_count > 1 else ''} detected"
        col = COLOR_MASK

    cv2.putText(frame, status, (8, 22), FONT, 0.55, col, 1, cv2.LINE_AA)

    if debug and fps:
        fps_txt = f"FPS: {fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_txt, FONT, 0.50, 1)
        cv2.putText(frame, fps_txt, (w - fw - 8, 22), FONT, 0.50, COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, "Press  Q  to quit", (8, h - 8), FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)


def process_frame(frame, face_cascade, model, classes, args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=args.scale,
        minNeighbors=args.neighbors,
        minSize=(args.min_face, args.min_face),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        resized = cv2.resize(face_roi, IMG_SIZE)
        rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb_face.astype(np.float32), axis=0)

        prob = float(model.predict(inp, verbose=0)[0][0])

        if prob >= args.threshold:
            pred_class = classes[1]
            confidence = prob
        else:
            pred_class = classes[0]
            confidence = 1.0 - prob

        is_masked = ("with_mask" in pred_class and "without" not in pred_class)
        color = COLOR_MASK if is_masked else COLOR_NO_MASK
        tag = "Mask" if is_masked else "No Mask"
        label = f"{tag}: {confidence * 100:.1f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_label(frame, label, x, y, color, font_scale=0.62, thickness=1)

        if args.debug:
            raw_txt = f"raw={prob:.3f}"
            cv2.putText(frame, raw_txt, (x, y + h + 14), FONT, 0.42, COLOR_INFO, 1, cv2.LINE_AA)

    return frame, len(faces)


def run_image(image_path, face_cascade, model, classes, args):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Cannot read image: '{image_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Processing image {image_path} ({frame.shape[1]}x{frame.shape[0]})")
    annotated, face_count = process_frame(frame, face_cascade, model, classes, args)
    print(f"[INFO] Faces detected: {face_count}")

    out = Path("annotated_image.jpg")
    cv2.imwrite(str(out), annotated)
    print(f"[OK] Saved -> {out.resolve()}")

    if args.no_display:
        return

    try:
        win = "Face Mask Detection - press any key to close"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"[WARN] GUI window skipped ({e}). Check annotated_image.jpg.")


def run_video_stream(source, face_cascade, model, classes, args):
    capture_src = int(source) if source.isdigit() else source
    is_camera = isinstance(capture_src, int)

    print(f"[INFO] Opening {'camera index' if is_camera else 'file'}: {capture_src}")
    cap = cv2.VideoCapture(capture_src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source '{source}'.", file=sys.stderr)
        if is_camera:
            print("  Checklist:", file=sys.stderr)
            print("  - Close apps using webcam (Zoom, Teams, browser)", file=sys.stderr)
            print("  - Windows Settings -> Privacy -> Camera -> allow Desktop apps", file=sys.stderr)
            print("  - Try: python detect_mask.py --source 1", file=sys.stderr)
        sys.exit(1)

    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win = "Face Mask Detection | Q = quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)
    print("[INFO] Window opened. Focus it and press Q to quit.")

    fps_val = 0.0
    t_prev = time.time()
    frame_n = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended." if not is_camera else "[WARN] Frame grab failed.")
                break

            annotated, face_count = process_frame(frame, face_cascade, model, classes, args)

            frame_n += 1
            if frame_n % 10 == 0:
                now = time.time()
                fps_val = 10.0 / max(now - t_prev, 1e-6)
                t_prev = now

            draw_status_bar(annotated, face_count, fps_val, args.debug)
            cv2.imshow(win, annotated)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                print("[INFO] Quit by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_arguments()
    model, classes = load_model_and_classes()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[ERROR] Haar Cascade not found at '{cascade_path}'.", file=sys.stderr)
        sys.exit(1)

    src = args.source
    path = Path(src)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if path.is_file() and path.suffix.lower() in image_exts:
        run_image(path, face_cascade, model, classes, args)
    else:
        run_video_stream(src, face_cascade, model, classes, args)


if __name__ == "__main__":
    main()
