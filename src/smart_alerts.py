import time


class SmartAlertSystem:

    def __init__(
        self,
        distraction_limit=2.0,
        posture_limit=5.0,
        critical_distraction_limit=5.0,
        cooldown=5.0
    ):

        # Alert thresholds
        self.distraction_limit = distraction_limit
        self.posture_limit = posture_limit
        self.critical_distraction_limit = (
            critical_distraction_limit
        )

        # Minimum time before the same alert
        # can be generated again
        self.cooldown = cooldown

        # Timers
        self.distraction_start = None
        self.posture_start = None

        # Current alert
        self.current_alert = "No active alerts"
        self.current_severity = "INFO"

        # Last generated alert
        self.last_alert = None
        self.last_alert_time = 0.0

        # Total number of generated alerts
        self.alert_count = 0

        # Duration of the current condition
        self.current_alert_duration = 0.0

    # =========================================================
    # CHECK COOLDOWN
    # =========================================================

    def can_generate_alert(
        self,
        alert,
        current_time
    ):

        # Different alert → allow immediately
        if alert != self.last_alert:
            return True

        # Same alert → wait for cooldown
        if (
            current_time -
            self.last_alert_time
            >= self.cooldown
        ):
            return True

        return False

    # =========================================================
    # REGISTER ALERT
    # =========================================================

    def register_alert(
        self,
        alert,
        severity,
        current_time
    ):

        if self.can_generate_alert(
            alert,
            current_time
        ):

            self.last_alert = alert
            self.last_alert_time = current_time

            self.current_alert = alert
            self.current_severity = severity

            self.alert_count += 1

    # =========================================================
    # CLEAR CURRENT ALERT
    # =========================================================

    def clear_alert(self):

        self.current_alert = "No active alerts"
        self.current_severity = "INFO"
        self.current_alert_duration = 0.0

    # =========================================================
    # MAIN ALERT ANALYSIS
    # =========================================================

    def generate_alert(
        self,
        attention_status,
        head_direction,
        eye_direction,
        posture,
        drowsiness_status,
        ergonomic_risk,
        face_detected,
        body_detected
    ):

        current_time = time.time()

        # Start each frame assuming
        # there is no new alert.
        active_alert_generated = False

        # =====================================================
        # 1. FACE NOT DETECTED
        # =====================================================

        if not face_detected:

            self.distraction_start = None
            self.posture_start = None

            self.current_alert_duration = 0.0

            alert = (
                "Please position yourself "
                "in front of the camera."
            )

            self.register_alert(
                alert,
                "WARNING",
                current_time
            )

            active_alert_generated = True

            return self.get_result()

        # =====================================================
        # 2. BODY NOT DETECTED
        # =====================================================

        if not body_detected:

            self.posture_start = None
            self.current_alert_duration = 0.0

            alert = (
                "Please ensure your body "
                "is visible to the camera."
            )

            self.register_alert(
                alert,
                "WARNING",
                current_time
            )

            active_alert_generated = True

            return self.get_result()

        # =====================================================
        # 3. DROWSINESS
        # =====================================================

        if drowsiness_status == "DROWSY":

            alert = (
                "Possible drowsiness detected."
            )

            self.register_alert(
                alert,
                "CRITICAL",
                current_time
            )

            active_alert_generated = True

            return self.get_result()

        # =====================================================
        # 4. HIGH ERGONOMIC RISK
        # =====================================================

        if ergonomic_risk == "HIGH RISK":

            alert = (
                "Take a posture break."
            )

            self.register_alert(
                alert,
                "CRITICAL",
                current_time
            )

            active_alert_generated = True

            return self.get_result()

        # =====================================================
        # 5. POSTURE MONITORING
        # =====================================================

        bad_posture = posture in [
            "SLOUCHING",
            "LEANING",
            "SHOULDERS UNEVEN"
        ]

        if bad_posture:

            if self.posture_start is None:
                self.posture_start = current_time

            posture_duration = (
                current_time -
                self.posture_start
            )

            self.current_alert_duration = (
                posture_duration
            )

            if posture_duration >= self.posture_limit:

                alert = (
                    "Please correct your posture."
                )

                self.register_alert(
                    alert,
                    "WARNING",
                    current_time
                )

                active_alert_generated = True

                return self.get_result()

        else:

            self.posture_start = None

        # =====================================================
        # 6. DISTRACTION MONITORING
        # =====================================================

        distracted = (
            head_direction != "CENTER"
            or eye_direction != "CENTER"
            or attention_status in [
                "DISTRACTED",
                "LOW ATTENTION"
            ]
        )

        if distracted:

            if self.distraction_start is None:
                self.distraction_start = current_time

            distraction_duration = (
                current_time -
                self.distraction_start
            )

            self.current_alert_duration = (
                distraction_duration
            )

            # Critical distraction
            if (
                distraction_duration
                >= self.critical_distraction_limit
            ):

                alert = (
                    "You have been distracted "
                    "for an extended period."
                )

                self.register_alert(
                    alert,
                    "CRITICAL",
                    current_time
                )

                active_alert_generated = True

                return self.get_result()

            # Warning distraction
            elif (
                distraction_duration
                >= self.distraction_limit
            ):

                alert = (
                    "You are looking away "
                    "from the screen."
                )

                self.register_alert(
                    alert,
                    "WARNING",
                    current_time
                )

                active_alert_generated = True

                return self.get_result()

        else:

            self.distraction_start = None

        # =====================================================
        # 7. LOW ATTENTION
        # =====================================================

        if attention_status == "LOW ATTENTION":

            alert = (
                "Focus on the screen."
            )

            self.register_alert(
                alert,
                "WARNING",
                current_time
            )

            active_alert_generated = True

            return self.get_result()

        # =====================================================
        # 8. DISTRACTED ATTENTION
        # =====================================================

        if attention_status == "DISTRACTED":

            alert = (
                "Try to maintain your "
                "attention on the screen."
            )

            self.register_alert(
                alert,
                "INFO",
                current_time
            )

            active_alert_generated = True

            return self.get_result()

        # =====================================================
        # 9. NORMAL CONDITION
        # =====================================================

        if not active_alert_generated:

            self.clear_alert()

        return self.get_result()

    # =========================================================
    # RESULT
    # =========================================================

    def get_result(self):

        return {
            "alert": self.current_alert,
            "severity": self.current_severity,
            "alert_count": self.alert_count,
            "alert_duration": (
                self.current_alert_duration
            )
        }