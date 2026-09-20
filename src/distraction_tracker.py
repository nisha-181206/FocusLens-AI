import time


class DistractionTracker:

    def __init__(self):

        self.distraction_start = None
        self.distraction_duration = 0.0

    def update(
        self,
        attention_status,
        head_direction,
        eye_direction
    ):

        distracted = (
            attention_status in [
                "DISTRACTED",
                "LOW ATTENTION"
            ]
            or head_direction != "CENTER"
            or eye_direction != "CENTER"
        )

        current_time = time.time()

        if distracted:

            if self.distraction_start is None:
                self.distraction_start = current_time

            self.distraction_duration = (
                current_time -
                self.distraction_start
            )

        else:

            self.distraction_start = None
            self.distraction_duration = 0.0

        return self.get_result()

    def get_result(self):

        return {
            "is_distracted": (
                self.distraction_start is not None
            ),
            "distraction_duration": (
                self.distraction_duration
            )
        }