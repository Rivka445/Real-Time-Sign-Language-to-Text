from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Paths
MODEL_FILE = ROOT_DIR / "artifacts" / "model.pkl"
TASK_MODEL_PATH = Path(r"C:\Users\user1\hand_landmarker.task")
CSV_FILE = ROOT_DIR / "data" / "dataset.csv"
EXPORT_FILE = ROOT_DIR / "exported_text.txt"

# Feature extraction
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3  # x, y, z

# Classification
MIN_PREDICTION_CONFIDENCE = 0.4

# Text accumulation
STABILITY_FRAMES = 20
COOLDOWN_SECONDS = 1.5

# Translation
TRANSLATION_MAP = {
    "ALEF": "א",
    "BEIT": "ב",
    "GIMEL": "ג",
    "DALET": "ד",
    "HEY": "ה",
    "VAV": "ו",
    "ZAIN": "ז",
    "CHEIT": "ח",
    "TEIT": "ט",
    "YUD": "י",
    "SPACE": " "
}

# Data collection
NUM_SAMPLES_PER_LETTER = 100

# UI
FONT_PATH = "arial.ttf"
FONT_SIZE = 30
