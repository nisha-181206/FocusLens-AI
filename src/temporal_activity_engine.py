import numpy as np
import joblib
from collections import deque

from temporal_activity_features import TemporalActivityFeatureExtractor


class TemporalActivityEngine:

    def __init__(
        self,
        model_path,
        sequence_length=30,
        smoothing_window=5
    ):
        self.model = joblib.load(model_path)

        self.extractor = TemporalActivityFeatureExtractor()

        self.sequence_length = sequence_length

        # Prediction smoothing
        self.prediction_history = deque(
            maxlen=smoothing_window
        )

        # Confidence smoothing
        self.confidence_history = deque(
            maxlen=smoothing_window
        )

        self.sequence = []

        self.current_activity = "WAITING"
        self.current_confidence = 0.0

    def update(self, landmarks):

        # -------------------------------------------------
        # NO BODY DETECTED
        # -------------------------------------------------

        if landmarks is None:

            self.sequence.clear()
            self.prediction_history.clear()
            self.confidence_history.clear()

            self.current_activity = "NO BODY"
            self.current_confidence = 0.0

            return self.get_result()

        # -------------------------------------------------
        # EXTRACT CURRENT FRAME FEATURES
        # -------------------------------------------------

        frame_features = (
            self.extractor.extract_frame_features(
                landmarks
            )
        )

        self.sequence.append(frame_features)

        # Keep only the latest 30 frames
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)

        # -------------------------------------------------
        # WAIT UNTIL 30 FRAMES ARE AVAILABLE
        # -------------------------------------------------

        if len(self.sequence) < self.sequence_length:

            self.current_activity = "COLLECTING"
            self.current_confidence = 0.0

            return self.get_result()

        # -------------------------------------------------
        # EXTRACT TEMPORAL FEATURES
        # -------------------------------------------------

        temporal_features = (
            self.extractor.extract_sequence_features(
                self.sequence
            )
        )

        temporal_features = (
            temporal_features.reshape(1, -1)
        )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = self.model.predict(
            temporal_features
        )[0]

        probabilities = self.model.predict_proba(
            temporal_features
        )[0]

        # Raw model confidence
        raw_confidence = (
            float(np.max(probabilities)) * 100
        )

        # -------------------------------------------------
        # PREDICTION SMOOTHING
        # -------------------------------------------------

        self.prediction_history.append(
            prediction
        )

        counts = {}

        for item in self.prediction_history:

            counts[item] = (
                counts.get(item, 0) + 1
            )

        # Most frequently predicted activity
        stable_activity = max(
            counts,
            key=counts.get
        )

        # -------------------------------------------------
        # CONFIDENCE SMOOTHING
        # -------------------------------------------------

        self.confidence_history.append(
            raw_confidence
        )

        smoothed_confidence = (
            sum(self.confidence_history)
            / len(self.confidence_history)
        )

        # -------------------------------------------------
        # STABILITY BONUS
        # -------------------------------------------------
        #
        # This does NOT change the model's actual
        # probability. It only represents how consistently
        # the recent predictions agree.
        #
        # Example:
        # 5/5 predictions = very stable
        # 3/5 predictions = less stable
        #

        stability = (
            counts[stable_activity]
            / len(self.prediction_history)
        )

        # Combine model confidence with prediction stability.
        #
        # 80% = model confidence
        # 20% = recent prediction stability
        #
        # This is a UI confidence/stability indicator,
        # not a newly measured model accuracy.

        display_confidence = (
            0.80 * smoothed_confidence
            + 0.20 * (stability * 100)
        )

        # -------------------------------------------------
        # UPDATE CURRENT RESULT
        # -------------------------------------------------

        self.current_activity = stable_activity

        self.current_confidence = (
            float(display_confidence)
        )

        return self.get_result()

    def get_result(self):

        return {
            "activity": self.current_activity,
            "confidence": self.current_confidence,
            "frames": len(self.sequence)
        }