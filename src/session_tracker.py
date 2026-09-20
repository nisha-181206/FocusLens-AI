import time
from collections import Counter


class SessionTracker:

    def __init__(self):
        self.start_time = time.time()

        self.attention_scores = []
        self.attention_statuses = []

        self.posture_history = []
        self.activity_history = []
        self.ergonomic_history = []

    def update(
        self,
        attention_score,
        attention_status,
        posture,
        activity,
        ergonomic_risk
    ):
        # Store attention
        if attention_score is not None:
            self.attention_scores.append(
                float(attention_score)
            )

        if attention_status:
            self.attention_statuses.append(
                attention_status
            )

        # Store posture
        if posture:
            self.posture_history.append(
                posture
            )

        # Store activity
        if activity:
            self.activity_history.append(
                activity
            )

        # Store ergonomic risk
        if ergonomic_risk:
            self.ergonomic_history.append(
                ergonomic_risk
            )

    def get_session_duration(self):
        return time.time() - self.start_time

    def get_average_attention(self):
        if not self.attention_scores:
            return 0.0

        return sum(
            self.attention_scores
        ) / len(self.attention_scores)

    def get_attention_distribution(self):
        if not self.attention_statuses:
            return {}

        total = len(self.attention_statuses)

        counts = Counter(
            self.attention_statuses
        )

        return {
            status: (
                count / total
            ) * 100
            for status, count in counts.items()
        }

    def get_posture_distribution(self):
        if not self.posture_history:
            return {}

        total = len(self.posture_history)

        counts = Counter(
            self.posture_history
        )

        return {
            posture: (
                count / total
            ) * 100
            for posture, count in counts.items()
        }

    def get_activity_distribution(self):
        if not self.activity_history:
            return {}

        total = len(self.activity_history)

        counts = Counter(
            self.activity_history
        )

        return {
            activity: (
                count / total
            ) * 100
            for activity, count in counts.items()
        }

    def get_ergonomic_distribution(self):
        if not self.ergonomic_history:
            return {}

        total = len(self.ergonomic_history)

        counts = Counter(
            self.ergonomic_history
        )

        return {
            risk: (
                count / total
            ) * 100
            for risk, count in counts.items()
        }

    def get_summary(self):

        duration = self.get_session_duration()

        return {
            "duration_seconds": duration,

            "average_attention":
                self.get_average_attention(),

            "attention_distribution":
                self.get_attention_distribution(),

            "posture_distribution":
                self.get_posture_distribution(),

            "activity_distribution":
                self.get_activity_distribution(),

            "ergonomic_distribution":
                self.get_ergonomic_distribution()
        }

    def reset(self):
        self.start_time = time.time()

        self.attention_scores.clear()
        self.attention_statuses.clear()

        self.posture_history.clear()
        self.activity_history.clear()
        self.ergonomic_history.clear()