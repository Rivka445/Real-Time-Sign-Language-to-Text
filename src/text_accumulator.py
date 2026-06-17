from collections import deque
import time
import logging
from .config import STABILITY_FRAMES, COOLDOWN_SECONDS, TRANSLATION_MAP as DEFAULT_TRANSLATION_MAP

logger = logging.getLogger(__name__)


class TextAccumulator:
    """
    Converts a stream of per-frame predictions into stable text.

    A letter is added to the sentence only when:
    1. The same letter appears consistently for STABILITY_FRAMES frames in a row.
    2. Either a different letter was last added, OR enough time has passed (COOLDOWN_SECONDS).

    This prevents flickering detections from spamming letters,
    and allows the same letter to be typed again after a cooldown.

    Uses a deque as a sliding window over recent predictions.
    """

    def __init__(self, stability_frames: int = STABILITY_FRAMES, cooldown_seconds: float = COOLDOWN_SECONDS,
                 translation_map: dict = None):
        """
        Args:
            stability_frames: How many consecutive identical frames needed to confirm a letter (default: 20).
            cooldown_seconds: Minimum time before the same letter can be added again (default: 1.5s).
            translation_map: Dict mapping class labels to display characters (e.g. "ALEF" -> "א").
                             Falls back to TRANSLATION_MAP from config if not provided.
        """
        self._history = deque(maxlen=stability_frames)
        self._stability_frames = stability_frames
        self._cooldown_seconds = cooldown_seconds
        self._translation_map = translation_map or DEFAULT_TRANSLATION_MAP
        self._sentence = ""
        self._last_letter = None
        self._last_time = 0.0

    @property
    def sentence(self) -> str:
        """The accumulated text sentence so far."""
        return self._sentence

    def update(self, detected: str | None) -> bool:
        """
        Processes one frame's detection result and updates the sentence if stable.

        Args:
            detected: The predicted label for this frame, or None if no hand detected.

        Returns:
            True if a new letter was added to the sentence, False otherwise.
        """
        if detected:
            self._history.append(detected)
        else:
            # No hand in frame — reset history to require fresh stability
            self._history.clear()
            return False

        # Check if the last N frames all show the same letter
        if len(self._history) == self._stability_frames and len(set(self._history)) == 1:
            stable = self._history[0]
            now = time.time()
            # Add only if it's a new letter, or cooldown has expired (allows repeating same letter)
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
        """Resets the sentence and all internal state."""
        self._sentence = ""
        self._last_letter = None
        self._history.clear()
        logger.info("Sentence cleared")

    def save(self, path: str = "exported_text.txt"):
        """
        Saves the current sentence to a text file.

        Args:
            path: Output file path (default: exported_text.txt).
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._sentence)
        logger.info("Sentence saved to %s", path)
