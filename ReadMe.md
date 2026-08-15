# Indian Sign Language (ISL) Detector

A real-time desktop application that detects and translates Indian Sign Language gestures using computer vision and deep learning. The system recognises both static alphabets and dynamic word signs simultaneously through a live camera feed and accumulates confirmed signs into a readable sentence.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [System Requirements](#3-system-requirements)
4. [Software and Dependencies](#4-software-and-dependencies)
5. [Project Structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [Feature Extraction](#7-feature-extraction)
8. [Models](#8-models)
9. [How Detection Works](#9-how-detection-works)
10. [Installation and Setup](#10-installation-and-setup)
11. [Running the Project](#11-running-the-project)
12. [GUI Guide](#12-gui-guide)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)
14. [Accuracy](#14-accuracy)
15. [Known Limitations](#15-known-limitations)
16. [Future Scope](#16-future-scope)

---

## 1. Project Overview

Indian Sign Language (ISL) is the primary mode of communication for the hearing-impaired community in India. This project builds a standalone real-time ISL detector that:

- Detects **24 static alphabets** (A to Z, excluding H and J) instantly from a single camera frame
- Detects **8 dynamic word signs** (H, J, hello, bye, namaste, practice, thank_you, sorry) from a sequence of frames
- Displays live predictions with a colour-coded confidence score
- Builds a sentence sign by sign as detections are confirmed
- Logs every confirmed detection with a timestamp
- Runs entirely offline on a standard laptop — no internet, no cloud API, no GPU required

---

## 2. Features

| Feature | Description |
|---|---|
| Static sign detection | 24 ISL alphabets detected instantly every frame |
| Dynamic sign detection | 8 ISL word signs detected from 30-frame sequences |
| Live confidence meter | Colour-coded bar updating every frame |
| Sentence builder | Confirmed detections accumulate into a readable sentence |
| Detection history | Timestamped log of every captured sign |
| Corner countdown timer | Circular arc countdown after each confirmed sign |
| Zoom in and out | Up to 3x digital zoom on the camera feed |
| Screenshot capture | Saves current frame as PNG to captures/ folder |
| Video recording | Records MP4 session to recordings/ folder |
| Dark and light theme | Full theme toggle, all widgets update correctly |
| Scrollable sidebar | All controls visible regardless of screen size |
| Keyboard shortcuts | Full keyboard control without the mouse |

---

## 3. System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Operating System | Windows 10 64-bit | Windows 11 64-bit |
| Python | 3.11.x | 3.11.0 |
| RAM | 4 GB | 8 GB |
| CPU | Intel Core i5 8th gen | Intel Core i7 10th gen or newer |
| GPU | Not required | Optional, speeds up LSTM inference |
| Webcam | 720p | 1080p |
| Storage | 2 GB free | 4 GB free |

Tested on Windows 11 with Python 3.11.0, a standard laptop webcam, and no dedicated GPU.

---

## 4. Software and Dependencies

### Core Language

**Python 3.11.0** — https://www.python.org/downloads/release/python-3110/
During installation, tick **Add Python to PATH**.

### Python Packages

These are the exact versions installed and tested in this project:

| Package | Version | Purpose |
|---|---|---|
| numpy | 1.26.4 | Numerical arrays and feature vectors |
| opencv-python | 4.8.1.78 | Camera capture, image processing, HUD drawing |
| mediapipe | 0.10.9 | Hand landmark detection (21 points per hand) and face mesh |
| pillow | 12.1.1 | Converting OpenCV frames to Tkinter-compatible images |
| scikit-learn | 1.8.0 | MLPClassifier for static sign model |
| joblib | 1.5.3 | Saving and loading the static model pkl file |
| tensorflow | 2.15.0 | Loading and running the dynamic Bi-LSTM model |
| keras | 2.15.0 | High-level API for the LSTM model |
| h5py | 3.15.1 | Reading the h5 model file format |
| matplotlib | 3.10.8 | Plotting training curves and confusion matrices |
| scipy | 1.17.1 | Scientific computing, used internally by sklearn |
| pandas | 3.0.1 | Data handling during training |
| pyinstaller | 6.19.0 | Packaging the app as a standalone executable |
| sounddevice | 0.5.5 | Audio output (future text-to-speech support) |

### Built-in Python Libraries

No installation needed — these come with Python:

- `tkinter` — GUI framework
- `threading` — Background model loading without freezing the UI
- `collections` — deque for the sequence buffer, Counter for majority voting
- `datetime` — Timestamps in the history log
- `math` — Arc drawing for the countdown timer
- `os`, `time` — File paths and timing

### Development Tools

- Visual Studio Code — https://code.visualstudio.com
- VS Code Python extension (Microsoft)
- VS Code Pylance extension
- Git 2.51.0 — https://git-scm.com

---

## 5. Project Structure

```
SLD/
|
|-- app.py                       Main GUI application — run this to launch
|-- feature_extractor.py         Hand landmark normalisation and feature extraction
|-- train_dynamic.py             Train the dynamic Bi-LSTM model
|-- train_static_model.py        Train the static MLP model
|-- collect_static_data.py       Collect static sign dataset via webcam
|-- collect_dynamic_data.py      Collect dynamic sign dataset via webcam
|-- load_static_dataset.py       Dataset loader utility for static training
|-- load_dynamic_dataset.py      Dataset loader utility for dynamic training
|
|-- static_sign_model.pkl        Trained static MLP classifier
|-- dynamic_sign_model.h5        Trained dynamic Bi-LSTM model
|-- dynamic_sign_model_best.h5   Best checkpoint saved during training
|
|-- ReadMe.md                    This file
|-- requirements.txt             All installed packages with exact versions
|-- .gitignore                   Files excluded from GitHub
|
|-- dataset/
|   |-- dynamic/
|   |   |-- H/          p1/ p2/ p3/ p4/   (npy sequence files)
|   |   |-- J/          p1/ p2/ p3/ p4/
|   |   |-- bye/        p1/ p2/ p3/ p4/
|   |   |-- hello/      p1/ p2/ p3/ p4/
|   |   |-- namaste/    p1/ p2/ p3/ p4/
|   |   |-- practice/   p1/ p2/ p3/ p4/
|   |   |-- sorry/      p1/ p2/ p3/ p4/
|   |   |-- thank_you/  p1/ p2/ p3/ p4/
|
|-- captures/                    Screenshots saved here (auto-created)
|-- recordings/                  Video recordings saved here (auto-created)
|-- venv/                        Python virtual environment (not on GitHub)
|-- build/                       PyInstaller build output (not on GitHub)
|-- dist/                        Standalone executable output (not on GitHub)
```

---

## 6. Dataset

### Static Signs

- 24 classes: A B C D E F G I K L M N O P Q R S T U V W X Y Z
- H and J are excluded because they involve motion and are treated as dynamic signs
- 200 samples collected per class per person
- Data collected from 4 different people (p1, p2, p3, p4)
- Total: approximately 19,200 static samples
- File format: 126-dimensional numpy array saved as .npy
- Folder structure: dataset/static/LABEL/PERSON_ID/N.npy

### Dynamic Signs

- 8 classes: H, J, bye, hello, namaste, practice, sorry, thank_you
- 100 sequences per class per person (4 persons)
- Data collected from 4 different people (p1, p2, p3, p4)
- Total: 3,200 sequences across 8 classes
- File format: numpy array of shape (30, 129) saved as .npy
- Each sequence = 30 consecutive frames, 129 features per frame
- Folder structure: dataset/dynamic/LABEL/PERSON_ID/N.npy

---

## 7. Feature Extraction

All features are extracted in `feature_extractor.py` using MediaPipe hand landmarks.

### normalize_hand()

Every hand goes through two normalisation steps before being used:

1. **Translation invariance** — subtract the wrist landmark (landmark 0) from all 21 points so the hand position in the frame does not matter
2. **Scale invariance** — divide all coordinates by the maximum landmark distance from the wrist, so hand size and distance from the camera do not matter

This means predictions are unaffected by where the hand is in the frame, how large it appears, or how close the person sits to the camera.

### Static features — 126 dimensions

- MediaPipe detects 21 landmarks per hand, each with x, y, z coordinates
- Each hand is normalised using normalize_hand() giving a 63-dimensional vector
- Left hand (63) and right hand (63) are concatenated to give 126 dimensions
- If a hand is not detected, its 63 values are set to zero

### Dynamic features — 129 dimensions

- Same 126-dimensional hand features as static
- Plus 3 additional values: the x, y, z coordinates of the nose tip (MediaPipe FaceMesh landmark 1)
- The nose tip provides a head-relative reference point for signs that move relative to the face
- Total: 126 + 3 = 129 dimensions per frame

---

## 8. Models

### Static Model — MLPClassifier

| Property | Value |
|---|---|
| Algorithm | Multi-Layer Perceptron |
| Hidden layers | (256, 128) |
| Activation | ReLU |
| Solver | Adam |
| Input shape | (126,) |
| Output | 24-class softmax probabilities |
| File | static_sign_model.pkl |
| Loaded with | joblib |
| Training accuracy | approximately 99% |

### Dynamic Model — Bidirectional LSTM

| Property | Value |
|---|---|
| Algorithm | Stacked Bidirectional LSTM |
| Layer 1 | Bidirectional LSTM 128 units + BatchNorm + Dropout 0.3 |
| Layer 2 | Bidirectional LSTM 64 units + BatchNorm + Dropout 0.3 |
| Layer 3 | LSTM 32 units + BatchNorm + Dropout 0.2 |
| Dense head | Dense 64 ReLU + Dropout 0.3 |
| Output | Dense 8 Softmax |
| Input shape | (30, 129) |
| Output | 8-class softmax probabilities |
| File | dynamic_sign_model.h5 |
| Loaded with | tensorflow.keras |
| Test accuracy | 99.17% |
| Training stopped | Early stopping at epoch 72 |

The Bidirectional LSTM reads each 30-frame sequence in both forward and backward directions, capturing temporal patterns from both ends of the gesture motion.

---

## 9. How Detection Works

### Three-State Machine

The app runs a state machine on every camera frame:

```
WAITING ── hand appears ──> DETECTING ── 5 consecutive same sign ──> COOLDOWN
   ^                                                                       |
   └───────────────────── cooldown timer expires ─────────────────────────┘
```

**WAITING** — No hand detected. Camera is running but no prediction is made.

**DETECTING** — Hand is present. Static model runs every frame for instant response. Dynamic model fills a 30-frame buffer and runs inference every 3rd frame using the cached result from the previous run. A sign is confirmed after 5 consecutive frames where the same label is predicted above the confidence threshold.

**COOLDOWN** — Sign is confirmed and added to the sentence. A circular arc countdown timer appears in the bottom-right corner of the camera feed. The user cannot trigger the next sign until the timer expires. Default cooldown is 2.5 seconds and is adjustable in the Settings panel.

### Static vs Dynamic Decision

Both models run simultaneously on every frame. Dynamic overrides static only when:
- Dynamic confidence is at or above 0.80, AND
- Dynamic confidence exceeds static confidence by more than 10%, OR visible hand movement is detected

Otherwise the static prediction is shown.

### Sentence Building

- Confirmed signs are appended to the sentence with a space
- Consecutive identical signs are never duplicated
- The last word can be deleted, or the entire sentence can be cleared

---

## 10. Installation and Setup

### Step 1 — Install Python 3.11

Download from https://www.python.org/downloads/release/python-3110/
Tick **Add Python to PATH** during installation.

### Step 2 — Create a virtual environment

```bash
cd C:\Users\YourName\Desktop
mkdir ISL_PROJECT
cd ISL_PROJECT
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.9
pip install pillow==12.1.1
pip install scikit-learn==1.8.0
pip install joblib==1.5.3
pip install tensorflow==2.15.0
pip install h5py==3.15.1
pip install matplotlib==3.10.8
pip install scipy==1.17.1
pip install pandas==3.0.1
```

### Step 4 — Copy project files

Place all .py files, static_sign_model.pkl, dynamic_sign_model.h5, and the dataset/ folder into your project folder.

### Step 5 — Verify

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

```bash
python -c "
import joblib
from tensorflow.keras.models import load_model
s = joblib.load('static_sign_model.pkl')
d = load_model('dynamic_sign_model.h5')
print('Static model OK  classes:', len(s.classes_))
print('Dynamic model OK  input:', d.input_shape)
"
```

---

## 11. Running the Project

### Launch the application

```bash
venv\Scripts\activate
python app.py
```

Wait for the status bar to show **Ready | Static + Dynamic** before clicking Start Camera.

### Train the dynamic model from scratch

Only needed if you collect new data:

```bash
python train_dynamic.py
```

Reads from dataset/dynamic/ and saves a new dynamic_sign_model.h5.
Takes approximately 10 to 30 minutes depending on hardware.

### Train the static model from scratch

Only needed if you collect new static data:

```bash
python train_static_model.py
```

### Collect new static data

```bash
python collect_static_data.py
```

Enter your person ID when prompted. The program collects 200 samples per class with a 3-second countdown before each letter.

### Collect new dynamic data

```bash
python collect_dynamic_data.py
```

Enter your person ID when prompted. Press R to redo the current sequence, Q to quit.

---

## 12. GUI Guide

| Element | Location | Description |
|---|---|---|
| Status bar | Top header | Model loading state and recording indicator |
| Clock | Top right header | Live time and date |
| Camera feed | Main left area | Live webcam with hand landmarks and HUD overlay |
| Prediction card | Sidebar top | Current sign label, confidence percentage, mode, colour bar |
| Translated Sentence | Sidebar | Accumulated confirmed signs as a sentence |
| Detection History | Sidebar | Timestamped log of every confirmed sign |
| Controls | Sidebar | Camera, capture, record, zoom buttons |
| Settings | Sidebar | Theme toggle, confidence threshold slider, cooldown slider |
| Keyboard Shortcuts | Sidebar bottom | Quick reference for all keyboard shortcuts |

### Camera Feed HUD

| Element | Location | Description |
|---|---|---|
| State label | Top left | WAITING, DETECTING, or CAPTURED |
| Current sign | Top left large | The predicted label in large white text |
| Confidence bar | Top right | Green above 80%, blue above 60%, purple below 60% |
| Countdown circle | Bottom right | Circular arc countdown during cooldown only |
| Progress bar | Bottom edge | Thin bar showing cooldown fill progress |

---

## 13. Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Start or stop the camera |
| R | Start or stop video recording |
| C | Capture a screenshot |
| Delete | Remove the last word from the sentence |
| Escape | Clear the entire sentence |

---

## 14. Accuracy

| Model | Metric | Value |
|---|---|---|
| Static MLP | Test accuracy | approximately 99% |
| Dynamic Bi-LSTM | Test accuracy | 99.17% |
| Dynamic Bi-LSTM | Val accuracy at best epoch | 99.17% |
| Dynamic Bi-LSTM | Training stopped at epoch | 72 of 150 |

Dataset details:
- Static: 200 samples per class per person, 24 classes, 4 persons
- Dynamic: 100 sequences per class per person, 8 classes, 4 persons
- Features are translation and scale normalised making them robust to hand size, position, and distance from camera

---

## 15. Known Limitations

- **Lighting sensitivity** — Poor or backlit lighting reduces MediaPipe landmark detection quality which directly affects prediction accuracy. Consistent front-facing lighting gives the best results.
- **Dynamic sign speed** — If the sign is performed significantly faster or slower than the training data, dynamic accuracy may drop because the 30-frame sequence captures a different portion of the motion.
- **No grammar correction** — The sentence builder outputs raw sign labels in detection order. It does not apply ISL grammar rules or correct natural language structure.
- **H and J in still poses** — Because H and J are dynamic signs, the static model does not predict them. If the hand is held still in an H or J pose for a long time, the dynamic model may not fire because there is no motion to distinguish it from surrounding signs.
- **Single environment tested** — The application has been tested on Windows 11 only. Behaviour on Windows 10, Mac, or Linux has not been verified.

---

## 16. Future Scope

- **Expanded vocabulary** — Add more dynamic words, phrases, and numbers beyond the current 8 signs.
- **NLP sentence correction** — Pass the raw detected sentence through a language model API to apply grammar correction and produce natural language output.
- **Text to speech** — Convert the corrected sentence to audio output using the pyttsx3 library which is already cross-platform and offline.
- **Mobile deployment** — Convert both models to TensorFlow Lite format for Android or iOS deployment using a phone camera.
- **Continuous signing mode** — Remove the cooldown gate and use a sliding window with deduplication to allow uninterrupted signing without pauses between each sign.
- **More persons in dataset** — Collecting data from more signers will improve generalisation across different individuals, signing speeds, and styles.
- **Two-hand coordination** — Extend the dynamic model to explicitly encode relative motion between both hands for signs that require coordinated two-hand gestures.
