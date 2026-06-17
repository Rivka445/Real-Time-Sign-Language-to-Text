import cv2
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src import FeatureExtractor, SignClassifier, TextAccumulator, Renderer, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

MODEL_FILE = config.MODEL_FILE

if not os.path.exists(config.MODEL_FILE):
    logger.error("Model file '%s' not found", config.MODEL_FILE)
    sys.exit(1)

extractor = FeatureExtractor(model_path=str(config.TASK_MODEL_PATH))
classifier = SignClassifier(model_path=str(config.MODEL_FILE))
accumulator = TextAccumulator()
renderer = Renderer()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open camera")
    sys.exit(1)

logger.info("Application started. Click camera window then press Q=quit, C=clear, S=save")

while True:
    ret, frame = cap.read()
    if not ret:
        logger.warning("Failed to read frame from camera")
        break

    features = extractor.extract(frame)
    detected, confidence = (None, 0.0)

    if features is not None:
        detected, confidence = classifier.predict(features)

    accumulator.update(detected)

    frame = renderer.render(frame, detected, confidence, accumulator.sentence)
    cv2.imshow('Sign Language Translator', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        logger.info("User requested exit")
        break
    elif key == ord('c'):
        accumulator.clear()
    elif key == ord('s'):
        accumulator.save(str(config.EXPORT_FILE))

cap.release()
cv2.destroyAllWindows()
extractor.close()
logger.info("Application closed")
