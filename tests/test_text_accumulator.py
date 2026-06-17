import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_accumulator import TextAccumulator


TRANSLATION = {"ALEF": "א", "BEIT": "ב"}


def make_accumulator():
    return TextAccumulator(stability_frames=3, cooldown_seconds=0.01, translation_map=TRANSLATION)


def test_stable_letter_added():
    acc = make_accumulator()
    acc._last_letter = None
    for _ in range(3):
        acc.update("ALEF")
    assert acc.sentence == "א", f"Expected 'א' but got '{acc.sentence}'"


def test_unstable_not_added():
    acc = make_accumulator()
    acc.update("ALEF")
    acc.update("BEIT")
    acc.update("ALEF")
    assert acc.sentence == ""


def test_no_detection_clears_history():
    acc = make_accumulator()
    acc.update("ALEF")
    acc.update("ALEF")
    acc.update(None)
    acc.update("ALEF")
    acc.update("ALEF")
    assert acc.sentence == ""


def test_clear_resets_sentence():
    acc = make_accumulator()
    for _ in range(3):
        acc.update("ALEF")
    acc.clear()
    assert acc.sentence == ""


def test_unknown_letter_passthrough():
    acc = make_accumulator()
    for _ in range(3):
        acc.update("UNKNOWN")
    assert acc.sentence == "UNKNOWN"


def test_space_translation():
    acc = make_accumulator()
    for _ in range(3):
        acc.update("BEIT")
    assert acc.sentence == "ב"


if __name__ == "__main__":
    tests = [test_stable_letter_added, test_unstable_not_added,
             test_no_detection_clears_history, test_clear_resets_sentence,
             test_unknown_letter_passthrough, test_space_translation]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__} - {e}")
