import math
import numpy as np
from collections import deque


class PostureAnalyzer:

    def __init__(self, smoothing_window=15):
        self.smoothing_window = smoothing_window
        self.posture_history = deque(maxlen=smoothing_window)

    # ============================================================
    # CALCULATE ANGLE BETWEEN THREE POINTS
    # ============================================================

    def calculate_angle(self, a, b, c):

        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)

        ba = a - b
        bc = c - b

        denominator = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denominator == 0:
            return 0.0

        cosine_angle = np.dot(ba, bc) / denominator

        cosine_angle = np.clip(
            cosine_angle,
            -1.0,
            1.0
        )

        angle = np.degrees(
            np.arccos(cosine_angle)
        )

        return float(angle)

    # ============================================================
    # NORMALIZE LINE ANGLE
    # ============================================================

    def normalize_line_angle(self, angle):

        # Convert angles such as +175° or -175°
        # into a small angle close to 0°.

        if angle > 90:
            angle -= 180

        elif angle < -90:
            angle += 180

        return angle

    # ============================================================
    # MAIN POSTURE ANALYSIS
    # ============================================================

    def analyze(self, landmarks):

        # --------------------------------------------------------
        # IMPORTANT LANDMARKS
        # --------------------------------------------------------

        nose = landmarks[0]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # --------------------------------------------------------
        # CONVERT LANDMARKS TO 2D POINTS
        # --------------------------------------------------------

        nose_point = [
            nose.x,
            nose.y
        ]

        left_shoulder_point = [
            left_shoulder.x,
            left_shoulder.y
        ]

        right_shoulder_point = [
            right_shoulder.x,
            right_shoulder.y
        ]

        left_hip_point = [
            left_hip.x,
            left_hip.y
        ]

        right_hip_point = [
            right_hip.x,
            right_hip.y
        ]

        # --------------------------------------------------------
        # SHOULDER CENTER
        # --------------------------------------------------------

        shoulder_center = (
            np.array(left_shoulder_point) +
            np.array(right_shoulder_point)
        ) / 2

        # --------------------------------------------------------
        # HIP CENTER
        # --------------------------------------------------------

        hip_center = (
            np.array(left_hip_point) +
            np.array(right_hip_point)
        ) / 2

        # ========================================================
        # 1. SHOULDER ANGLE
        # ========================================================

        raw_shoulder_angle = math.degrees(
            math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            )
        )

        # Fix the ±180° problem
        shoulder_angle = self.normalize_line_angle(
            raw_shoulder_angle
        )

        # ========================================================
        # 2. TORSO ANGLE
        # ========================================================

        dx = (
            shoulder_center[0] -
            hip_center[0]
        )

        dy = (
            shoulder_center[1] -
            hip_center[1]
        )

        torso_angle = math.degrees(
            math.atan2(
                dx,
                -dy
            )
        )

        # Normalize torso angle too
        torso_angle = self.normalize_line_angle(
            torso_angle
        )

        # ========================================================
        # 3. NECK ANGLE
        # ========================================================

        neck_angle = self.calculate_angle(
            nose_point,
            shoulder_center,
            hip_center
        )

        # ========================================================
        # 4. POSTURE CLASSIFICATION
        # ========================================================

        # More tolerant thresholds.

        neck_bad = neck_angle < 125

        torso_bad = abs(torso_angle) > 25

        shoulders_bad = abs(shoulder_angle) > 15

        # --------------------------------------------------------
        # PRIORITY-BASED CLASSIFICATION
        # --------------------------------------------------------

        if neck_bad:

            raw_posture = "SLOUCHING"

        elif torso_bad:

            raw_posture = "LEANING"

        elif shoulders_bad:

            raw_posture = "SHOULDERS UNEVEN"

        else:

            raw_posture = "GOOD POSTURE"

        # ========================================================
        # TEMPORAL SMOOTHING
        # ========================================================

        self.posture_history.append(
            raw_posture
        )

        posture_counts = {}

        for posture in self.posture_history:

            posture_counts[posture] = (
                posture_counts.get(posture, 0) + 1
            )

        stable_posture = max(
            posture_counts,
            key=posture_counts.get
        )

        # ========================================================
        # STABILITY
        # ========================================================

        stability = (
            posture_counts[stable_posture]
            / len(self.posture_history)
        ) * 100

        # ========================================================
        # RETURN RESULTS
        # ========================================================

        return {

            "neck_angle": neck_angle,

            "shoulder_angle": shoulder_angle,

            "torso_angle": torso_angle,

            "posture": stable_posture,

            "stability": stability,

            "neck_bad": neck_bad,

            "shoulders_bad": shoulders_bad,

            "torso_bad": torso_bad
        }