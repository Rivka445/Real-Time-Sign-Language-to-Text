import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import cv2
import logging
from .config import NUM_HANDS, MIN_DETECTION_CONFIDENCE, FEATURES_PER_HAND

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, model_path: str, num_hands: int = NUM_HANDS, min_confidence: float = MIN_DETECTION_CONFIDENCE):
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        logger.info("FeatureExtractor initialized with model: %s", model_path)

    def extract(self, bgr_frame) -> list[float] | None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)

        if not results.hand_landmarks:
            return None

        hands_data = []
        for idx in range(2):
            if idx < len(results.hand_landmarks):
                lm = results.hand_landmarks[idx]
                base_x, base_y, base_z = lm[0].x, lm[0].y, lm[0].z
                hands_data.extend([val for p in lm for val in (p.x - base_x, p.y - base_y, p.z - base_z)])
            else:
                hands_data.extend([0.0] * FEATURES_PER_HAND)

        return hands_data

    def close(self):
        self._detector.close()
        logger.info("FeatureExtractor closed")
