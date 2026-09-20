import time
from collections import deque


class ErgonomicAnalyzer:

    def __init__(self, bad_posture_limit=10):
        self.bad_posture_limit = bad_posture_limit
        self.bad_posture_start = None
        self.history = deque(maxlen=15)

    def analyze(
        self,
        neck_angle,
        shoulder_angle,
        torso_angle,
        posture
    ):

        reasons = []

        # --------------------------------------------------------
        # Detect significant posture problems
        # --------------------------------------------------------

        neck_bad = neck_angle < 125
        shoulder_bad = abs(shoulder_angle) > 15
        torso_bad = abs(torso_angle) > 25

        if neck_bad:
            reasons.append("Neck bent")

        if shoulder_bad:
            reasons.append("Uneven shoulders")

        if torso_bad:
            reasons.append("Torso leaning")

        # --------------------------------------------------------
        # Determine whether posture is actually problematic
        # --------------------------------------------------------

        problem_count = sum([
            neck_bad,
            shoulder_bad,
            torso_bad
        ])

        bad_posture = problem_count >= 2

        # --------------------------------------------------------
        # Track how long bad posture continues
        # --------------------------------------------------------

        current_time = time.time()

        if bad_posture:

            if self.bad_posture_start is None:
                self.bad_posture_start = current_time

            bad_duration = (
                current_time -
                self.bad_posture_start
            )

        else:

            self.bad_posture_start = None
            bad_duration = 0

        # --------------------------------------------------------
        # ERGONOMIC RISK
        # --------------------------------------------------------

        if bad_duration >= self.bad_posture_limit:

            risk = "HIGH RISK"

        elif bad_duration >= 5:

            risk = "MODERATE RISK"

        elif bad_posture:

            risk = "LOW RISK"

        else:

            risk = "LOW RISK"

        # --------------------------------------------------------
        # RECOMMENDATION
        # --------------------------------------------------------

        if risk == "HIGH RISK":

            recommendation = (
                "Please correct your posture"
            )

        elif risk == "MODERATE RISK":

            recommendation = (
                "Adjust your sitting position"
            )

        elif bad_posture:

            recommendation = (
                "Slight posture adjustment recommended"
            )

        else:

            recommendation = (
                "Posture looks good"
            )

        # --------------------------------------------------------
        # STORE HISTORY
        # --------------------------------------------------------

        self.history.append(risk)

        # --------------------------------------------------------
        # RETURN RESULTS
        # --------------------------------------------------------

        return {
            "bad_posture": bad_posture,
            "bad_duration": bad_duration,
            "risk": risk,
            "reasons": reasons,
            "recommendation": recommendation
        }