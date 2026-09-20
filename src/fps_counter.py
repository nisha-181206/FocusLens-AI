import time
from collections import deque


class FPSCounter:

    def __init__(self, smoothing_window=30):

        self.smoothing_window = smoothing_window

        self.frame_times = deque(
            maxlen=smoothing_window
        )

        self.last_time = time.perf_counter()

        self.current_fps = 0.0

    def update(self):

        current_time = time.perf_counter()

        elapsed = current_time - self.last_time

        self.last_time = current_time

        if elapsed > 0:

            self.frame_times.append(elapsed)

        if len(self.frame_times) > 0:

            average_frame_time = (
                sum(self.frame_times)
                / len(self.frame_times)
            )

            if average_frame_time > 0:

                self.current_fps = (
                    1.0 / average_frame_time
                )

        return self.current_fps

    def get_result(self):

        return {
            "fps": self.current_fps
        }