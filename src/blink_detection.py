import time
from collections import deque


class BlinkDetector:
    """
    Fast real-time blink detector using Eye Aspect Ratio (EAR).

    Designed for webcam-based MediaPipe Face Landmarker input.

    Main goals:
    - Detect short blinks quickly
    - Use both eyes
    - Avoid heavy smoothing that can hide fast blinks
    - Count a blink when the eyes reopen after being closed
    """

    def __init__(
        self,
        ear_threshold=0.25,
        min_closed_frames=1,
        smoothing_window=1
    ):
        # EAR below this value is considered an eye closure.
        self.ear_threshold = ear_threshold

        # A blink can be detected after just one closed frame.
        self.min_closed_frames = min_closed_frames

        # 1 = no temporal smoothing, giving the fastest response.
        self.smoothing_window = smoothing_window

        self.closed_frames = 0
        self.blink_count = 0
        self.total_frames = 0

        self.start_time = time.time()
        self.last_blink_time = 0.0

        self.ear_history = deque(
            maxlen=self.smoothing_window
        )

        self.current_ear = 0.0
        self.eye_state = "OPEN"

    def calculate_eye_ratio(
        self,
        landmarks,
        top_index,
        bottom_index,
        left_index,
        right_index
    ):
        """
        Calculate Eye Aspect Ratio (EAR).

        Smaller EAR values indicate a more closed eye.
        """

        top = landmarks[top_index]
        bottom = landmarks[bottom_index]
        left = landmarks[left_index]
        right = landmarks[right_index]

        vertical_distance = (
            (top.x - bottom.x) ** 2
            + (top.y - bottom.y) ** 2
        ) ** 0.5

        horizontal_distance = (
            (left.x - right.x) ** 2
            + (left.y - right.y) ** 2
        ) ** 0.5

        if horizontal_distance <= 1e-8:
            return 0.0

        return (
            vertical_distance
            / horizontal_distance
        )

    def update(self, landmarks):
        """
        Process one frame and update blink information.

        Returns:
            dict containing blink count, blink rate,
            eye state, EAR and closed-frame count.
        """

        if landmarks is None:
            self.current_ear = 0.0
            self.eye_state = "NO FACE"
            self.closed_frames = 0
            return self.get_result()

        # -------------------------------------------------
        # LEFT EYE EAR
        # -------------------------------------------------

        left_ear = self.calculate_eye_ratio(
            landmarks,
            159,   # upper eyelid
            145,   # lower eyelid
            33,    # outer eye corner
            133    # inner eye corner
        )

        # -------------------------------------------------
        # RIGHT EYE EAR
        # -------------------------------------------------

        right_ear = self.calculate_eye_ratio(
            landmarks,
            386,   # upper eyelid
            374,   # lower eyelid
            362,   # inner eye corner
            263    # outer eye corner
        )

        # Average both eyes.
        ear = (left_ear + right_ear) / 2.0

        # -------------------------------------------------
        # VERY LIGHT / NO SMOOTHING
        # -------------------------------------------------

        self.ear_history.append(ear)

        smoothed_ear = (
            sum(self.ear_history)
            / len(self.ear_history)
        )

        self.current_ear = smoothed_ear
        self.total_frames += 1

        # -------------------------------------------------
        # FAST BLINK DETECTION
        # -------------------------------------------------

        if smoothed_ear < self.ear_threshold:

            # Eye is currently closed.
            self.closed_frames += 1
            self.eye_state = "CLOSED"

        else:

            # Eye has opened again.
            # If it was closed for the required number
            # of frames, count exactly one blink.
            if (
                self.closed_frames
                >= self.min_closed_frames
            ):
                self.blink_count += 1
                self.last_blink_time = time.time()

            self.closed_frames = 0
            self.eye_state = "OPEN"

        return self.get_result()

    def get_blink_rate(self):
        """
        Return current blink rate in blinks per minute.
        """

        elapsed = time.time() - self.start_time

        if elapsed < 1.0:
            return 0.0

        minutes = elapsed / 60.0

        return self.blink_count / minutes

    def get_result(self):
        """
        Return current detector state.
        """

        return {
            "blink_count": self.blink_count,
            "blink_rate": self.get_blink_rate(),
            "eye_state": self.eye_state,
            "eye_ratio": self.current_ear,
            "closed_frames": self.closed_frames
        }
