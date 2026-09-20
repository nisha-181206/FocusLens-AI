from collections import deque


class AttentionEngine:

    def __init__(self, smoothing_window=15):

        self.smoothing_window = smoothing_window

        self.score_history = deque(
            maxlen=smoothing_window
        )

    # --------------------------------------------------
    # Calculate attention score
    # --------------------------------------------------

    def calculate_score(
        self,
        face_detected,
        head_direction,
        eye_direction
    ):

        # ----------------------------------------------
        # No face
        # ----------------------------------------------

        if not face_detected:

            score = 0

            self.score_history.append(score)

            return self.get_smoothed_score()


        # ----------------------------------------------
        # Starting score
        # ----------------------------------------------

        score = 50


        # ----------------------------------------------
        # Head direction
        # ----------------------------------------------

        if head_direction == "CENTER":

            score += 30

        elif head_direction in ["LEFT", "RIGHT"]:

            score -= 25

        elif head_direction in ["UP", "DOWN"]:

            score -= 30


        # ----------------------------------------------
        # Eye direction
        # ----------------------------------------------

        if eye_direction == "CENTER":

            score += 20

        elif eye_direction in ["LEFT", "RIGHT"]:

            score -= 20

        elif eye_direction in ["UP", "DOWN"]:

            score -= 15

        elif eye_direction == "UNKNOWN":

            score -= 5


        # ----------------------------------------------
        # Limit score
        # ----------------------------------------------

        score = max(
            0,
            min(100, score)
        )


        # ----------------------------------------------
        # Store score
        # ----------------------------------------------

        self.score_history.append(score)


        return self.get_smoothed_score()


    # --------------------------------------------------
    # Smooth score
    # --------------------------------------------------

    def get_smoothed_score(self):

        if not self.score_history:

            return 0

        return sum(
            self.score_history
        ) / len(self.score_history)


    # --------------------------------------------------
    # Attention status
    # --------------------------------------------------

    def get_status(self, score):

        if score >= 75:

            return "FOCUSED"

        elif score >= 50:

            return "PARTIALLY FOCUSED"

        elif score >= 30:

            return "DISTRACTED"

        else:

            return "NOT ATTENTIVE"