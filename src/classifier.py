import pickle
import numpy as np
import pandas as pd
import logging
from .config import MIN_PREDICTION_CONFIDENCE

logger = logging.getLogger(__name__)


class SignClassifier:
    def __init__(self, model_path: str, min_confidence: float = MIN_PREDICTION_CONFIDENCE):
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        self.min_confidence = min_confidence
        logger.info("SignClassifier loaded from: %s", model_path)

    def predict(self, features: list[float]) -> tuple[str, float] | tuple[None, float]:
        input_data = pd.DataFrame([features], columns=self._model.feature_names_in_)
        predicted = self._model.predict(input_data)[0]
        confidence = float(np.max(self._model.predict_proba(input_data)))

        if confidence < self.min_confidence:
            logger.debug("Low confidence %.2f for prediction %s - skipped", confidence, predicted)
            return None, confidence

        return predicted, confidence
