import pickle
import numpy as np
import pandas as pd
import logging
from .config import MIN_PREDICTION_CONFIDENCE

logger = logging.getLogger(__name__)


class SignClassifier:
    """
    Wraps the trained Random Forest model and handles prediction.

    Loads a pre-trained sklearn model from a .pkl file and predicts
    which sign letter matches the given hand features.
    Predictions below the confidence threshold are discarded to avoid
    showing unreliable results on screen.
    """

    def __init__(self, model_path: str, min_confidence: float = MIN_PREDICTION_CONFIDENCE):
        """
        Args:
            model_path: Path to the trained model .pkl file.
            min_confidence: Minimum probability required to accept a prediction (default: 0.4).
                            Predictions below this are returned as (None, confidence).
        """
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        self.min_confidence = min_confidence
        logger.info("SignClassifier loaded from: %s", model_path)

    def predict(self, features: list[float]) -> tuple[str, float] | tuple[None, float]:
        """
        Predicts the sign letter from a list of hand landmark features.

        Wraps the features in a DataFrame to match the model's expected
        input format (feature names must match training data columns).

        Args:
            features: Flat list of 126 floats from FeatureExtractor.extract().

        Returns:
            (label, confidence) if confidence >= min_confidence.
            (None, confidence) if confidence is too low to trust.
        """
        input_data = pd.DataFrame([features], columns=self._model.feature_names_in_)
        predicted = self._model.predict(input_data)[0]
        confidence = float(np.max(self._model.predict_proba(input_data)))

        if confidence < self.min_confidence:
            logger.debug("Low confidence %.2f for prediction %s - skipped", confidence, predicted)
            return None, confidence

        return predicted, confidence
