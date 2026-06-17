from collections import deque
import time
import logging
from .config import STABILITY_FRAMES, COOLDOWN_SECONDS, TRANSLATION_MAP as DEFAULT_TRANSLATION_MAP

logger = logging.getLogger(__name__)


class TextAccumulator:
    def __init__(self, stability_frames: int = STABILITY_FRAMES, cooldown_seconds: float = COOLDOWN_SECONDS,
                 translation_map: dict = None):
        self._history = deque(maxlen=stability_frames)
        self._stability_frames = stability_frames
        self._cooldown_seconds = cooldown_seconds
        self._translation_map = translation_map or DEFAULT_TRANSLATION_MAP
        self._sentence = ""
        self._last_letter = None
        self._last_time = 0.0

    @property
    def sentence(self) -> str:
        return self._sentence

    def update(self, detected: str | None) -> bool:
        if detected:
            self._history.append(detected)
        else:
            self._history.clear()
            return False

        if len(self._history) == self._stability_frames and len(set(self._history)) == 1:
            stable = self._history[0]
            now = time.time()
            if stable != self._last_letter or (now - self._last_time) > self._cooldown_seconds:
                char = self._translation_map.get(stable, stable)
                self._sentence += char
                self._last_letter = stable
                self._last_time = now
                self._history.clear()
                logger.info("Letter added: %s -> '%s' | Sentence: %s", stable, char, self._sentence)
                return True
        return False

    def clear(self):
        self._sentence = ""
        self._last_letter = None
        self._history.clear()
        logger.info("Sentence cleared")

    def save(self, path: str = "exported_text.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._sentence)
        logger.info("Sentence saved to %s", path)
