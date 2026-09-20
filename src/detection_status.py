class DetectionStatus:

    def __init__(self):
        self.face_detected = False
        self.body_detected = False

    def update(self, face_detected, body_detected):

        self.face_detected = bool(face_detected)
        self.body_detected = bool(body_detected)

        if self.face_detected:
            face_status = "DETECTED"
        else:
            face_status = "NOT DETECTED"

        if self.body_detected:
            body_status = "DETECTED"
        else:
            body_status = "NOT DETECTED"

        return {
            "face_detected": self.face_detected,
            "body_detected": self.body_detected,
            "face_status": face_status,
            "body_status": body_status
        }

    def get_result(self):

        return {
            "face_detected": self.face_detected,
            "body_detected": self.body_detected,
            "face_status": (
                "DETECTED"
                if self.face_detected
                else "NOT DETECTED"
            ),
            "body_status": (
                "DETECTED"
                if self.body_detected
                else "NOT DETECTED"
            )
        }