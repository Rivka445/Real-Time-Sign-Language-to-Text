import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from .config import FONT_PATH, FONT_SIZE

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self, font_path: str = FONT_PATH, font_size: int = FONT_SIZE):
        try:
            self._font = ImageFont.truetype(font_path, font_size)
        except Exception:
            self._font = ImageFont.load_default()
            logger.warning("Font '%s' not found, using default", font_path)

    def render(self, frame, detected: str | None, confidence: float, sentence: str):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        if detected:
            cv2.putText(frame, f"Current: {detected} ({confidence:.0%})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            bar_width = int(confidence * 150)
            cv2.rectangle(frame, (320, 30), (320 + bar_width, 50), (0, 255, 0), -1)
            cv2.rectangle(frame, (320, 30), (470, 50), (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Waiting for hand...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, "[Q]:Exit  [C]:Clear  [S]:Save", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 15), -1)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((20, h - 50), f"Text: {sentence}", font=self._font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
