# Indian Sign Language (ISL) Detector

A real-time desktop application that detects and translates Indian Sign Language gestures using computer vision and deep learning. The system recognises both static alphabets and dynamic word signs simultaneously through a live camera feed, and accumulates confirmed signs into a readable sentence.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Software & Dependencies](#software--dependencies)
5. [Project Structure](#project-structure)
6. [Dataset](#dataset)
7. [Models](#models)
8. [How Detection Works](#how-detection-works)
9. [Installation & Setup](#installation--setup)
10. [Running the Project](#running-the-project)
11. [GUI Guide](#gui-guide)
12. [Keyboard Shortcuts](#keyboard-shortcuts)
13. [Accuracy](#accuracy)
14. [Known Limitations](#known-limitations)
15. [Future Scope](#future-scope)

---

## Project Overview

Indian Sign Language (ISL) is the primary mode of communication for the hearing-impaired community in India. This project builds a standalone real-time ISL detector that:

- Detects **24 static alphabets** (A–Z excluding H and J) instantly from a single frame
- Detects **8 dynamic word signs** (H, J, hello, bye, namaste, practice, thank_you, sorry) from a sequence of frames
- Displays live predictions with confidence scores
- Builds a sentence word by word from confirmed signs
- Logs every detection with a timestamp in a history panel
- Runs entirely offline on a standard laptop with a webcam

The project does not require any cloud API, internet connection, or dedicated GPU to run.

---

## Features

| Feature | Details |
|---|---|
| Static sign detection | 24 ISL alphabets, instant per-frame prediction |
| Dynamic sign detection | 8 ISL words, sequence-based LSTM prediction |
| Live confidence meter | Colour-coded bar that updates every frame |
| Sentence builder | Accumulates confirmed signs into a sentence |
| Detection history | Timestamped log of every captured sign |
| Corner countdown timer | Circular arc countdown after each confirmed sign |
| Zoom in / out | Up to 3x digital zoom on camera feed |
| Screenshot capture | Saves current frame as PNG to captures/ folder |
| Video recording | Records MP4 of the session to recordings/ folder |
| Dark and light theme | Toggle between themes, all widgets update correctly |
| Scrollable sidebar | All controls visible regardless of screen height |
| Keyboard shortcuts | Full keyboard control without using the mouse |

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Operating System | Windows 10 64-bit | Windows 11 64-bit |
| Python | 3.11.x | 3.11.x |
| RAM | 4 GB | 8 GB |
| CPU | Intel Core i5 (8th gen) | Intel Core i7 (10th gen+) |
| GPU | Not required | Optional (speeds up LSTM inference) |
| Webcam | 720p | 1080p |
| Storage | 2 GB free | 4 GB free |

> The application has been tested on Windows 11 with Python 3.11.0, a standard laptop webcam, and no dedicated GPU.

---

## Software & Dependencies

### Core Language
- **Python 3.11.0** — [python.org](https://www.python.org/downloads/release/python-3110/)

### Python Packages (exact pinned versions)

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Numerical arrays, feature vectors |
| `opencv-python` | 4.8.1.78 | Camera capture, image processing, HUD drawing |
| `mediapipe` | 0.10.9 | Hand landmark detection (21 points per hand) and face mesh |
| `Pillow` | 10.2.0 | Converting OpenCV frames to Tkinter-compatible images |
| `scikit-learn` | 1.4.1.post1 | MLPClassifier for static sign model |
| `joblib` | 1.3.2 | Loading the static model .pkl file |
| `tensorflow` | 2.13.0 | Loading and running the dynamic LSTM model |
| `h5py` | 3.10.0 | Reading the .h5 model file format |
| `matplotlib` | 3.8.3 | Plotting training curves and confusion matrices |
| `seaborn` | 0.13.2 | Styled confusion matrix heatmaps |

### Built-in Python Libraries (no installation needed)
- `tkinter` — GUI framework
- `threading` — Background model loading
- `collections` — deque for sequence buffer, Counter for voting
- `datetime` — Timestamps in history log
- `math` — Arc drawing for countdown timer
- `os`, `time` — File paths, timing

### Development Tools
- **Visual Studio Code** — Recommended IDE
- **VS Code Extensions** — Python (Microsoft), Pylance

---

## Project Structure

```
ISL_FINAL/
│
├── app.py                      <- Main GUI application (run this)
├── feature_extractor.py        <- Hand landmark feature extraction
├── train_dynamic.py            <- Train the dynamic LSTM model
├── collect_static_data.py      <- Collect static sign dataset
├── collect_dynamic_data.py     <- Collect dynamic sign dataset
│
├── static_sign_model.pkl       <- Trained static MLP model
├── dynamic_sign_model.h5       <- Trained dynamic Bi-LSTM model
│
├── dataset/
│   └── dynamic/                <- Dynamic .npy sequence files
│       ├── H/p1/0.npy ... 449.npy
│       ├── J/p1/
│       ├── bye/p1/
│       ├── hello/p1/
│       ├── namaste/p1/
│       ├── practice/p1/
│       ├── sorry/p1/
│       └── thank_you/p1/
│
├── captures/                   <- Screenshots saved here (auto-created)
├── recordings/                 <- Video recordings saved here (auto-created)
│
└── venv/                       <- Python virtual environment
```

---

## Dataset

### Static Signs
- **24 classes**: A B C D E F G I K L M N O P Q R S T U V W X Y Z
  (H and J excluded — they are dynamic)
- **200 samples per class** collected per person
- **Feature format**: 126-dimensional numpy array
  (2 hands x 21 landmarks x 3 axes, wrist-centered and scale-normalised)
- **Storage**: `.npy` files under `dataset/static/<LABEL>/<PERSON_ID>/`

### Dynamic Signs
- **8 classes**: H, J, bye, hello, namaste, practice, sorry, thank_you
- **450 sequences per class** (collected and augmented)
- **Sequence format**: shape `(30, 129)` — 30 frames, 129 features per frame
  (126 hand features + 3 nose-tip coordinates for head-relative motion)
- **Storage**: `.npy` files under `dataset/dynamic/<LABEL>/<PERSON_ID>/`

### Feature Extraction
Both static and dynamic features use the same `extract_static_features()` pipeline:
1. MediaPipe detects 21 landmarks (x, y, z) per hand
2. Coordinates are translated to be wrist-relative (landmark 0 subtracted)
3. Scale-normalised by dividing by the maximum landmark distance
4. Left and right hands concatenated to a 126-dim vector
5. Dynamic adds nose-tip (x, y, z) to give 129 dims

---

## Models

### Static Model — MLPClassifier
- **Architecture**: Multi-Layer Perceptron with hidden layers (256, 128)
- **Activation**: ReLU
- **Solver**: Adam
- **Input**: 126-dim feature vector
- **Output**: 24-class probability distribution
- **File**: `static_sign_model.pkl` (loaded with joblib)
- **Training accuracy**: ~99%

### Dynamic Model — Bidirectional LSTM
- **Architecture**:
  - Bidirectional LSTM (128 units) + BatchNorm + Dropout 0.3
  - Bidirectional LSTM (64 units) + BatchNorm + Dropout 0.3
  - LSTM (32 units) + BatchNorm + Dropout 0.2
  - Dense (64, ReLU) + Dropout 0.3
  - Dense (8, Softmax)
- **Input**: sequence of shape (30, 129)
- **Output**: 8-class probability distribution
- **File**: `dynamic_sign_model.h5` (loaded with TensorFlow/Keras)
- **Test accuracy**: 99.17%
- **Training**: Early stopping at epoch 72, val_accuracy = 99.17%

---

## How Detection Works

### State Machine
The app runs a 3-state detection loop on every camera frame:

```
WAITING ──(hand appears)──> DETECTING ──(5 frames same sign)──> COOLDOWN
   ^                                                                  |
   └──────────────────────(timer expires)────────────────────────────┘
```

- **WAITING**: No hand in frame. Camera is active but no prediction runs.
- **DETECTING**: Hand detected. Every frame runs static prediction instantly.
  Dynamic buffer fills in background. Sign is confirmed after 5 consecutive
  frames with the same label above the confidence threshold.
- **COOLDOWN**: Sign confirmed and added to sentence. A countdown timer
  (circular arc in bottom-right corner) shows time before next detection.
  Default cooldown is 2.5 seconds (adjustable in Settings).

### Static vs Dynamic Decision
Both models run simultaneously on every frame:
- Static model predicts every frame (instant)
- Dynamic model predicts every 3rd frame using a cached result (performance)
- Dynamic overrides static only when:
  - Dynamic confidence >= 0.80 AND
  - Dynamic confidence > static confidence + 10%, OR there is visible hand movement

### Sentence Building
- Confirmed signs are appended to the sentence with spaces
- Consecutive identical signs are not duplicated
- Words can be deleted one at a time or the sentence cleared entirely

---

## Installation & Setup

### Step 1 — Install Python 3.11
Download from [python.org](https://www.python.org/downloads/release/python-3110/).
During installation, check **Add Python to PATH**.

### Step 2 — Create project folder and virtual environment
```bash
cd C:\Users\YourName\Desktop
mkdir ISL_FINAL
cd ISL_FINAL
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.9
pip install Pillow==10.2.0
pip install scikit-learn==1.4.1.post1
pip install joblib==1.3.2
pip install tensorflow==2.13.0
pip install h5py==3.10.0
pip install matplotlib==3.8.3
pip install seaborn==0.13.2
```

### Step 4 — Copy project files
Place all `.py` files, `static_sign_model.pkl`, `dynamic_sign_model.h5`,
and the `dataset/` folder into `ISL_FINAL/`.

### Step 5 — Verify installation
```bash
python -c "
import numpy, cv2, mediapipe, PIL, sklearn, joblib, tensorflow, h5py
print('numpy     :', numpy.__version__)
print('opencv    :', cv2.__version__)
print('mediapipe :', mediapipe.__version__)
print('tensorflow:', tensorflow.__version__)
print('All imports OK')
"
```

---

## Running the Project

### Launch the application
```bash
cd C:\Users\YourName\Desktop\ISL_FINAL
venv\Scripts\activate
python app.py
```

### Train the dynamic model from scratch (optional)
Only needed if you collect new dynamic data:
```bash
python train_dynamic.py
```
This reads from `dataset/dynamic/` and saves a new `dynamic_sign_model.h5`.
Training takes approximately 10–30 minutes depending on hardware.

### Collect new static data (optional)
```bash
python collect_static_data.py
```

### Collect new dynamic data (optional)
```bash
python collect_dynamic_data.py
```

---

## GUI Guide

When the app launches, wait for the status bar to show **Ready | Static + Dynamic** before starting the camera.

| Element | Location | Description |
|---|---|---|
| Status bar | Top header | Shows model loading state and recording status |
| Clock | Top right | Live time and date |
| Camera feed | Left (main area) | Live webcam with landmark overlay and HUD |
| Prediction card | Sidebar top | Shows current sign, confidence %, mode, bar |
| Translated Sentence | Sidebar | Accumulated confirmed signs as a sentence |
| Detection History | Sidebar | Timestamped log of every captured sign |
| Controls | Sidebar | Camera, capture, record, zoom buttons |
| Settings | Sidebar | Theme toggle, confidence threshold, cooldown slider |
| Shortcuts | Sidebar bottom | Keyboard shortcut reference |

### HUD Elements on Camera Feed
- **Top-left panel**: State (WAITING / DETECTING / CAPTURED) and current sign in large text
- **Top-right**: Confidence percentage bar (green >= 80%, blue >= 60%, purple < 60%)
- **Bottom-right**: Circular arc countdown timer during cooldown (green → blue → purple as time runs out)
- **Bottom edge**: Thin horizontal fill bar showing cooldown progress

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Start / Stop camera |
| R | Start / Stop video recording |
| C | Capture screenshot |
| Delete | Remove last word from sentence |
| Escape | Clear entire sentence |

---

## Accuracy

| Model | Test Accuracy |
|---|---|
| Static MLP (24 alphabets) | ~99% |
| Dynamic Bi-LSTM (8 signs) | 99.17% |

Results achieved with:
- 200 samples per static class
- 450 sequences per dynamic class
- Single person dataset collected under consistent indoor lighting

Accuracy may vary with different people, lighting conditions, and camera angles.

---

## Known Limitations

- **Single person trained**: The models were trained on one person's signs. Accuracy with other signers may be lower until the model is retrained with diverse data.
- **Lighting sensitive**: Poor or backlit lighting reduces MediaPipe's landmark detection accuracy and therefore prediction accuracy.
- **Dynamic signs require motion consistency**: If the sign is performed too fast or too slow compared to training speed, dynamic accuracy drops.
- **No grammar correction**: The sentence builder accumulates raw sign labels. It does not apply ISL grammar rules or natural language correction.
- **H and J overlap with static**: Since H and J are dynamic signs, the static model skips them. During still poses, the dynamic model may occasionally misfire for these letters.

---

## Future Scope

- **Multi-person dataset**: Collect data from multiple signers to build a generalised model that works across different people.
- **Expanded vocabulary**: Add more dynamic words and phrases beyond the current 8.
- **NLP sentence correction**: Pass the raw sentence through an LLM API (Claude, GPT) to apply grammar correction and produce natural language output.
  ```
  Prompt: "Fix the grammar of this ISL-translated text: {raw_sentence}"
  ```
- **Text-to-speech**: Convert the translated sentence to audio using Python's `pyttsx3` library.
- **Mobile deployment**: Convert models to TensorFlow Lite for Android/iOS deployment.
- **Continuous sentence mode**: Remove the cooldown gate and use a sliding window with deduplication to allow continuous signing without pauses.
- **Two-hand dynamic signs**: Current dynamic model tracks both hands but was trained primarily on single-dominant-hand signs. Extended two-hand coordination can be added.
