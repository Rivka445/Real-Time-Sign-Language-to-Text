import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import tempfile
import numpy as np
from src.classifier import SignClassifier


class MockModel:
    def __init__(self, label, confidence, n_features=126):
        self.feature_names_in_ = [f"f{i}" for i in range(n_features)]
        self._label = label
        self._confidence = confidence

    def predict(self, X):
        return [self._label]

    def predict_proba(self, X):
        proba = np.zeros((1, 3))
        proba[0][0] = self._confidence
        return proba


def make_classifier(predicted_label: str, confidence: float, min_confidence: float = 0.4):
    model = MockModel(predicted_label, confidence)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(model, f)
        path = f.name
    clf = SignClassifier(model_path=path, min_confidence=min_confidence)
    os.unlink(path)
    return clf


def test_predict_above_threshold():
    clf = make_classifier("ALEF", 0.9)
    label, conf = clf.predict([0.0] * 126)
    assert label == "ALEF"
    assert conf == 0.9


def test_predict_below_threshold_returns_none():
    clf = make_classifier("ALEF", 0.2, min_confidence=0.4)
    label, conf = clf.predict([0.0] * 126)
    assert label is None
    assert conf == 0.2


def test_predict_at_exact_threshold():
    clf = make_classifier("BEIT", 0.4, min_confidence=0.4)
    label, conf = clf.predict([0.0] * 126)
    assert label == "BEIT"


if __name__ == "__main__":
    tests = [test_predict_above_threshold, test_predict_below_threshold_returns_none,
             test_predict_at_exact_threshold]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__} - {e}")
