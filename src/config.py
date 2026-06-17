"""
config.py — Central configuration for the Sign Language Translator.

All constants, paths, and tunable parameters are defined here.
To modify behavior (e.g. add letters, change confidence threshold),
edit this file only — no need to touch other source files.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# --- Paths ---
# MODEL_FILE: trained Random Forest classifier saved after running scripts/train_model.py
MODEL_FILE = ROOT_DIR / "artifacts" / "model.pkl"

# TASK_MODEL_PATH: MediaPipe hand landmark detection model.
# NOTE: path must not contain non-ASCII characters (e.g. Hebrew) — MediaPipe will fail to load it.
TASK_MODEL_PATH = Path(r"C:\Users\user1\hand_landmarker.task")

# CSV_FILE: raw dataset collected by scripts/data_collection.py
CSV_FILE = ROOT_DIR / "data" / "dataset.csv"

# EXPORT_FILE: output file when user presses S in the live app
EXPORT_FILE = ROOT_DIR / "exported_text.txt"

# --- Feature Extraction ---
NUM_HANDS = 2                               # Max hands to detect per frame
MIN_DETECTION_CONFIDENCE = 0.6             # Below this, MediaPipe ignores the detection
NUM_LANDMARKS = 21                          # MediaPipe always returns 21 points per hand
FEATURES_PER_HAND = NUM_LANDMARKS * 3      # 21 points x (x, y, z) = 63 floats per hand

# --- Classification ---
# Predictions below this confidence are ignored (shown as None, not added to sentence)
# Lower = more permissive but noisier. Higher = stricter but may miss valid signs.
MIN_PREDICTION_CONFIDENCE = 0.4

# --- Text Accumulation ---
# A letter is confirmed only after this many consecutive identical predictions
# At ~30fps, 20 frames ≈ 0.6 seconds of holding the sign
STABILITY_FRAMES = 20

# After adding a letter, this cooldown prevents the same letter from being
# added again immediately (user must hold the sign for another cooldown period)
COOLDOWN_SECONDS = 1.5

# --- Translation Map ---
# Maps model class labels (as trained in dataset) to display characters.
# To add a new letter: collect data with data_collection.py, retrain, then add it here.
TRANSLATION_MAP = {
    "ALEF":  "א",
    "BEIT":  "ב",
    "GIMEL": "ג",
    "DALET": "ד",
    "HEY":   "ה",
    "VAV":   "ו",
    "ZAIN":  "ז",
    "CHEIT": "ח",
    "TEIT":  "ט",
    "YUD":   "י",
    "SPACE": " ",
}

# --- Data Collection ---
# Number of samples to collect per letter in data_collection.py
NUM_SAMPLES_PER_LETTER = 100

# --- UI ---
FONT_PATH = "arial.ttf"   # Must support the display language (Hebrew requires a Unicode font)
FONT_SIZE = 30
