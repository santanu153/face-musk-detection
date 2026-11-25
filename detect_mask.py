"""Detect faces and classify their mask status in a webcam, image, or video."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras


MODEL_PATH = Path("models/face_mask_model.h5")
CLASSES_PATH = Path("models/classes.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
COLORS = {"with_mask": (0, 200, 0), "without_mask": (0, 0, 255)}


def parse_source(source):
    """Return a camera index for numeric sources, otherwise the path."""
    try:
        return int(source)
    except ValueError:
        return source


def load_detector():
    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        raise SystemExit(
            "Model files are missing. Run `python train.py` before detection."
        )

    model = keras.models.load_model(MODEL_PATH)
    with CLASSES_PATH.open(encoding="utf-8") as classes_file:
        class_names = json.load(classes_file)
    if class_names != ["with_mask", "without_mask"]:
        raise SystemExit(f"Unexpected class names in {CLASSES_PATH}.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit(f"Could not load face detector: {cascade_path}")
    return model, class_names, face_cascade


def annotate_frame(frame, model, class_names, face_cascade, fps):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, width, height in faces:
        face = frame[y : y + height, x : x + width]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        score = float(
            np.asarray(model.predict(np.expand_dims(face, axis=0), verbose=0)).reshape(-1)[0]
        )
        class_index = int(score >= 0.5)
        label = class_names[class_index]
        confidence = score if class_index else 1.0 - score
        color = COLORS[label]

        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            f"{label}: {confidence:.1%}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    status_text = f"FPS: {fps:.1f} | with_mask: green | without_mask: red | Q: quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(
        frame,
        status_text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="webcam index (0 or 1), image path, or video path (default: 0)",
    )
    args = parser.parse_args()

    model, class_names, face_cascade = load_detector()
    source = parse_source(args.source)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        frame = annotate_frame(frame, model, class_names, face_cascade, 0.0)
        cv2.imshow("Face Mask Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    previous_time = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            frame = annotate_frame(frame, model, class_names, face_cascade, fps)
            cv2.imshow("Face Mask Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
