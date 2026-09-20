class LiveRecommendation:

    def __init__(self):

        self.current_recommendation = (
            "System is monitoring your activity."
        )

    def generate(
        self,
        attention_status,
        posture,
        activity,
        ergonomic_risk,
        drowsiness_status,
        face_detected,
        body_detected,
        head_direction,
        eye_direction
    ):

        # -----------------------------
        # FACE NOT DETECTED
        # -----------------------------

        if not face_detected:

            recommendation = (
                "Please position yourself "
                "in front of the camera."
            )

        # -----------------------------
        # BODY NOT DETECTED
        # -----------------------------

        elif not body_detected:

            recommendation = (
                "Please ensure your body "
                "is visible to the camera."
            )

        # -----------------------------
        # DROWSINESS
        # -----------------------------

        elif drowsiness_status == "DROWSY":

            recommendation = (
                "Take a short break and "
                "stay alert."
            )

        # -----------------------------
        # HIGH ERGONOMIC RISK
        # -----------------------------

        elif ergonomic_risk == "HIGH RISK":

            recommendation = (
                "Take a posture break "
                "and adjust your position."
            )

        # -----------------------------
        # POOR POSTURE
        # -----------------------------

        elif posture == "SLOUCHING":

            recommendation = (
                "Sit upright and "
                "straighten your back."
            )

        elif posture == "LEANING":

            recommendation = (
                "Try to maintain a "
                "balanced sitting position."
            )

        elif posture == "SHOULDERS UNEVEN":

            recommendation = (
                "Relax and align "
                "your shoulders."
            )

        # -----------------------------
        # DISTRACTION
        # -----------------------------

        elif (
            head_direction != "CENTER"
            or eye_direction != "CENTER"
        ):

            recommendation = (
                "Please focus your attention "
                "on the screen."
            )

        # -----------------------------
        # LOW ATTENTION
        # -----------------------------

        elif attention_status == "LOW ATTENTION":

            recommendation = (
                "Try to maintain your "
                "attention on the screen."
            )

        # -----------------------------
        # PHONE USAGE
        # -----------------------------

        elif activity == "PHONE_USAGE":

            recommendation = (
                "Avoid unnecessary phone "
                "usage during your session."
            )

        # -----------------------------
        # GOOD STATE
        # -----------------------------

        else:

            recommendation = (
                "Your posture and attention "
                "look good. Keep it up!"
            )

        self.current_recommendation = recommendation

        return self.get_result()

    def get_result(self):

        return {
            "recommendation": (
                self.current_recommendation
            )
        }