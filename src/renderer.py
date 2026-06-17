import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from .config import FONT_PATH, FONT_SIZE

logger = logging.getLogger(__name__)


class Renderer:
    """
    Handles all drawing and UI rendering onto the video frame.

    Uses OpenCV for rectangles and ASCII text, and PIL (Pillow) for
    the sentence text at the bottom — because OpenCV does not support
    non-Latin characters (Hebrew would show as ??).

    Layout:
        - Top bar (dark): current detected letter + confidence bar
        - Middle: keyboard shortcut hints
        - Bottom bar (darker): accumulated sentence text
    """

    def __init__(self, font_path: str = FONT_PATH, font_size: int = FONT_SIZE):
        """
        Args:
            font_path: Path to a .ttf font file that supports the display language (default: arial.ttf).
                       If not found, falls back to PIL's built-in default font.
            font_size: Font size for the sentence text (default: 30).
        """
        try:
            self._font = ImageFont.truetype(font_path, font_size)
        except Exception:
            self._font = ImageFont.load_default()
            logger.warning("Font '%s' not found, using default", font_path)

    def render(self, frame, detected: str | None, confidence: float, sentence: str):
        """
        Draws all UI elements onto the frame and returns the annotated frame.

        Steps:
        1. Draw top bar with current detected letter and confidence bar (green fill).
        2. Draw shortcut hints below the top bar.
        3. Draw bottom bar with the accumulated sentence using PIL (supports Unicode/Hebrew).
        4. Convert back from PIL to OpenCV BGR format for display.

        Args:
            frame: Current BGR video frame (numpy array from OpenCV).
            detected: Currently detected letter label, or None if no hand.
            confidence: Prediction confidence (0.0 - 1.0).
            sentence: The full accumulated sentence to display at the bottom.

        Returns:
            Annotated BGR frame ready to be shown with cv2.imshow().
        """
        h, w = frame.shape[:2]

        # Top bar background
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        if detected:
            cv2.putText(frame, f"Current: {detected} ({confidence:.0%})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Confidence bar: width proportional to confidence score
            bar_width = int(confidence * 150)
            cv2.rectangle(frame, (320, 30), (320 + bar_width, 50), (0, 255, 0), -1)
            cv2.rectangle(frame, (320, 30), (470, 50), (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Waiting for hand...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Shortcut hints
        cv2.putText(frame, "[Q]:Exit  [C]:Clear  [S]:Save", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Bottom bar with sentence — use PIL to support Hebrew/Unicode characters
        cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 15), -1)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((20, h - 50), f"Text: {sentence}", font=self._font, fill=(255, 255, 255))

        # Convert back to BGR for OpenCV
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
