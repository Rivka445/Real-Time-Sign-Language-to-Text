import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import cv2
import logging
from .config import NUM_HANDS, MIN_DETECTION_CONFIDENCE, FEATURES_PER_HAND

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Responsible for detecting hands in a video frame and extracting
    21 landmark points (x, y, z) per hand using MediaPipe.

    The landmarks are normalized relative to the wrist (point 0),
    so the model is position-independent — it doesn't matter where
    the hand appears on screen.

    Output: a flat list of 126 floats (2 hands x 21 points x 3 coords).
    If only one hand is detected, the second hand is padded with zeros.
    """

    def __init__(self, model_path: str, num_hands: int = NUM_HANDS, min_confidence: float = MIN_DETECTION_CONFIDENCE):
        """
        Args:
            model_path: Path to the MediaPipe hand_landmarker.task file.
            num_hands: Max number of hands to detect (default: 2).
            min_confidence: Minimum detection confidence to accept a result (default: 0.6).
        """
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        logger.info("FeatureExtractor initialized with model: %s", model_path)

    def extract(self, bgr_frame) -> list[float] | None:
        """
        Extracts normalized hand landmarks from a single BGR video frame.

        Converts the frame to RGB (required by MediaPipe), runs detection,
        then normalizes each landmark relative to the wrist (landmark 0).
        Normalization makes the features scale and position invariant.

        Args:
            bgr_frame: A single frame from OpenCV (BGR format, numpy array).

        Returns:
            A flat list of 126 floats if at least one hand is detected, None otherwise.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)

        if not results.hand_landmarks:
            return None

        hands_data = []
        for idx in range(2):
            if idx < len(results.hand_landmarks):
                lm = results.hand_landmarks[idx]
                # Normalize relative to wrist (point 0) for position invariance
                base_x, base_y, base_z = lm[0].x, lm[0].y, lm[0].z
                hands_data.extend([val for p in lm for val in (p.x - base_x, p.y - base_y, p.z - base_z)])
            else:
                # Pad with zeros if this hand is not detected
                hands_data.extend([0.0] * FEATURES_PER_HAND)

        return hands_data

    def close(self):
        """Releases the MediaPipe detector resources. Call this when done."""
        self._detector.close()
        logger.info("FeatureExtractor closed")
