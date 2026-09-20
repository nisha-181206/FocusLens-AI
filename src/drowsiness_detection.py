import time


class DrowsinessDetector:

    def __init__(
        self,
        closed_eye_limit=2.0,
        warning_limit=1.0
    ):
        self.closed_eye_limit = closed_eye_limit
        self.warning_limit = warning_limit

        self.eye_closed_start = None
        self.current_status = "ALERT"
        self.drowsiness_score = 0.0

    def update(self, eye_state, attention_status):

        current_time = time.time()

        # ==========================================
        # EYE CLOSURE TRACKING
        # ==========================================

        if eye_state == "CLOSED":

            if self.eye_closed_start is None:
                self.eye_closed_start = current_time

            closed_duration = (
                current_time - self.eye_closed_start
            )

        else:

            self.eye_closed_start = None
            closed_duration = 0.0

        # ==========================================
        # DROWSINESS ANALYSIS
        # ==========================================

        if closed_duration >= self.closed_eye_limit:

            self.drowsiness_score = 100.0
            self.current_status = "DROWSY"

        elif closed_duration >= self.warning_limit:

            self.drowsiness_score = 60.0
            self.current_status = "WARNING"

        elif attention_status in [
            "DISTRACTED",
            "LOW ATTENTION"
        ]:

            self.drowsiness_score = 30.0
            self.current_status = "LOW ALERTNESS"

        else:

            self.drowsiness_score = 0.0
            self.current_status = "ALERT"

        return self.get_result()

    def get_result(self):

        if self.eye_closed_start is None:
            closed_duration = 0.0
        else:
            closed_duration = (
                time.time() -
                self.eye_closed_start
            )

        return {
            "status": self.current_status,
            "drowsiness_score": self.drowsiness_score,
            "closed_duration": closed_duration
        }