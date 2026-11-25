# Face Mask Detection Using CNN, TensorFlow and OpenCV

A real-time computer vision desktop application for detecting human faces and classifying whether each person is wearing a face mask (`with_mask`) or not (`without_mask`).

---

## 1. Project Objective and Overview

This project implements a two-stage deep learning and computer vision pipeline:
1. **Face Detection Stage (OpenCV Haar Cascade):** Scans the input stream (webcam, video, or image) using histogram equalization and `haarcascade_frontalface_default.xml` to detect human faces.
2. **Mask Classification Stage (Keras CNN):** Crops each detected face, converts to RGB (128x128), and passes it to a trained Convolutional Neural Network (CNN) to predict mask status and confidence.
3. **Visual Feedback:** Annotates the live video or image with color-coded bounding boxes and readable labels:
   - **Green Box:** Face wearing a mask (`Mask: <confidence>%`)
   - **Red Box:** Face without a mask (`No Mask: <confidence>%`)

### Deliverables
- `models/face_mask_model.h5`: Trained Keras CNN model weights and architecture.
- `models/classes.json`: Serialized JSON array of class labels for inference alignment.
- `models/training_curves.png`: Accuracy and loss curves over training epochs.
- `train.py`: End-to-end dataset validation, CNN building, training, and model serialization script.
- `detect_mask.py`: Real-time inference application supporting webcam, video, and image inputs.
- `requirements.txt`: Core project dependencies.

---

## 2. Directory Structure

```text
face-musk-detection/
├── dataset/
│   ├── with_mask/          # Masked face images
│   └── without_mask/       # Unmasked face images
├── models/
│   ├── face_mask_model.h5  # Trained CNN model
│   ├── classes.json        # Class label order mapping
│   └── training_curves.png # Training & validation curves
├── train.py                # Model training script
├── detect_mask.py          # Real-time inference script
├── requirements.txt        # Core dependencies
└── README.md               # Complete project documentation
```

---

## 3. System Requirements & Prerequisites

- **Operating System:** Windows 10/11 64-bit (macOS and Linux also supported)
- **Python Version:** 3.11 (64-bit)
- **RAM:** Minimum 8 GB recommended
- **Camera:** Built-in or external USB webcam (optional for image-only inference)

---

## 4. Installation & Environment Setup

### Windows PowerShell

1. Clone the repository and open PowerShell in the project folder:
   ```powershell
   git clone https://github.com/santanu153/face-musk-detection.git
   cd face-musk-detection
   ```

2. Create and activate a Python 3.11 virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.)*

3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## 5. Dataset Organization

Place your training images directly into their respective subdirectories:

```text
dataset/
├── with_mask/
│   ├── 0_0_0.jpg
│   └── ...
└── without_mask/
    ├── 0_0_0.jpg
    └── ...
```

### Dataset Sources
- **Kaggle Face Mask Dataset:** https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
- **Real-World Masked Face Dataset:** https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
- **Prajna Bhandary Face Mask Dataset:** https://github.com/prajnasb/observations

---

## 6. CNN Architecture

| Layer | Configuration | Description / Purpose |
|---|---|---|
| **Input** | `(128, 128, 3)` | RGB face crop input |
| **Rescaling** | `1.0 / 255` | Normalizes pixel values from [0, 255] to [0.0, 1.0] |
| **Augmentation** | `RandomFlip("horizontal")`, `RandomRotation(0.1)` | Enhances generalization against head tilt & orientation |
| **Conv Block 1** | `Conv2D(32, 3x3, relu, same)` + `MaxPool2D(2x2)` | Extracts low-level visual features |
| **Conv Block 2** | `Conv2D(64, 3x3, relu, same)` + `MaxPool2D(2x2)` | Learns intermediate facial shapes and contours |
| **Conv Block 3** | `Conv2D(128, 3x3, relu, same)` + `MaxPool2D(2x2)` | Learns high-level features (mask fabric, mouth, nose) |
| **Pooling** | `GlobalAveragePooling2D()` | Flattens spatial feature maps to prevent overfitting |
| **Dense** | `Dense(64, relu)` | Non-linear dense representation |
| **Regularization** | `Dropout(0.3)` | Prevents co-adaptation of neurons |
| **Output** | `Dense(1, sigmoid)` | Outputs binary probability for mask classification |

---

## 7. Training the Model

Run the training pipeline:
```powershell
python train.py
```

This validates the dataset, creates an 80/20 train/validation split, trains with Adam (`lr=0.001`) and `EarlyStopping` (patience=4), and saves `models/face_mask_model.h5`, `models/classes.json`, and `models/training_curves.png`.

---

## 8. Real-Time Detection & Inference

### Webcam Detection (Default)
```powershell
python detect_mask.py
```
- Green rectangle = **Mask**
- Red rectangle = **No Mask**
- Press **`Q`** to quit

### Optional Flags
```powershell
# Use external USB webcam (index 1)
python detect_mask.py --source 1

# Single image detection (saves annotated_image.jpg)
python detect_mask.py --source test.jpg

# Recorded video detection
python detect_mask.py --source sample.mp4

# Debug mode (shows raw probability and FPS)
python detect_mask.py --debug
```
