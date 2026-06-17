# 🤟 Real-Time Sign Language to Text

A real-time Hebrew Sign Language recognition system that translates hand signs into text using computer vision and machine learning.

The system captures hand gestures through a webcam, extracts 21 landmark points per hand using MediaPipe, classifies them with a trained Random Forest model, and accumulates stable predictions into a sentence displayed on screen.

---

## 🎬 Demo

| 👁️ Detecting a sign | 📝 Accumulated text |
|---|---|
| `Current: ALEF (92%)` | `Text: אבג` |

---

## 🏗️ Architecture

```
live_prediction.py          # Entry point — main loop
src/
  config.py                 # All constants, paths, and settings
  feature_extractor.py      # MediaPipe hand landmark extraction
  classifier.py             # Random Forest prediction wrapper
  text_accumulator.py       # Stability + cooldown logic
  renderer.py               # OpenCV + PIL UI rendering
scripts/
  data_collection.py        # Collect training data per letter
  train_model.py            # Train and evaluate the model
  check_data.py             # Inspect dataset distribution
tests/
  test_text_accumulator.py
  test_classifier.py
data/
  dataset.csv               # Collected hand landmark samples
artifacts/
  model.pkl                 # Trained classifier
  hand_landmarker.task      # MediaPipe model file
```

### 🔄 Pipeline

```
📷 Webcam frame
    → 🖐️ FeatureExtractor   (126 normalized floats: 2 hands × 21 points × x,y,z)
    → 🧠 SignClassifier     (predicted label + confidence)
    → 📝 TextAccumulator    (stable for 20 frames → add to sentence)
    → 🖥️ Renderer           (draw UI overlay on frame)
```
Demo
Detecting a sign	Accumulated text
Current: ALEF (92%)	Text: אבג
---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.10+
- Webcam

### 📦 Installation

```bash
git clone https://github.com/Rivka445/Real-Time-Sign-Language-to-Text.git
cd Real-Time-Sign-Language-to-Text
pip install -r requirements.txt
```

Download the MediaPipe hand landmark model:

```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'artifacts/hand_landmarker.task'
)
print('Downloaded.')
"
```

### 🐳 Run with Docker

```bash
docker build -t sign-language-translator .
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix sign-language-translator
```

---

## 📖 Usage

### 1. 🎥 Collect training data

```bash
python scripts/data_collection.py
```

Enter the letter name when prompted (e.g. `ALEF`, `BEIT`, `SPACE`).
Collect at least 100 samples per letter in varied lighting and distances.

### 2. 🧠 Train the model

```bash
python scripts/train_model.py
```

Outputs `artifacts/model.pkl` and a confusion matrix image.

### 3. ▶️ Run the live app

```bash
python live_prediction.py
```

| Key | Action |
|-----|--------|
| `Q` | 🚪 Quit |
| `C` | 🗑️ Clear sentence |
| `S` | 💾 Save sentence to `exported_text.txt` |

> **💡 Note:** Click on the camera window before using keyboard shortcuts.

---

## ⚙️ Configuration

All settings are in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_PREDICTION_CONFIDENCE` | `0.4` | Minimum confidence to accept a prediction |
| `STABILITY_FRAMES` | `20` | Frames needed to confirm a letter (~0.6s at 30fps) |
| `COOLDOWN_SECONDS` | `1.5` | Wait time before the same letter can repeat |
| `NUM_SAMPLES_PER_LETTER` | `100` | Samples collected per letter |

To add a new letter, add it to `TRANSLATION_MAP` in `src/config.py`:

```python
TRANSLATION_MAP = {
    "ALEF": "א",
    "YOUR_NEW_LETTER": "X",   # add here
    ...
}
```

---

## 🧪 Running Tests

```bash
python tests/test_text_accumulator.py
python tests/test_classifier.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🖐️ MediaPipe | Hand landmark detection |
| 🌲 scikit-learn | Random Forest classifier |
| 📷 OpenCV | Camera capture and UI |
| 🖼️ Pillow | Unicode/Hebrew text rendering |
| 🔢 NumPy / Pandas | Data processing |
| 📊 Matplotlib / Seaborn | Evaluation charts |
