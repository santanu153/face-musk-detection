# Face Mask Detection

A desktop application for training a convolutional neural network and detecting
face masks in images, videos, and webcam streams.

## Setup

Use Python 3.10 or later. Create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place labeled training images in `dataset/with_mask/` and
`dataset/without_mask/`. The directories are included as empty placeholders.
The images should show faces clearly; the training script resizes them to
224 × 224 pixels.

## Train the model

```bash
python train.py
```

Training uses an 80/20 training/validation split and early stopping. The model,
class names, and training curves are saved to `models/`.

## Detect masks

Run the detector with the default webcam (`0`), or select another webcam,
image, or video with `--source`:

```bash
python detect_mask.py
python detect_mask.py --source 1
python detect_mask.py --source path/to/image.jpg
python detect_mask.py --source path/to/video.mp4
```

Press `q` to close the display. Detected faces are labeled `with_mask` (green)
or `without_mask` (red). The display includes an FPS counter and status bar.

## Project structure

```text
.
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── models/                  # Generated model and training outputs
├── detect_mask.py           # OpenCV face detection and mask inference
├── train.py                 # CNN training
└── requirements.txt
```
