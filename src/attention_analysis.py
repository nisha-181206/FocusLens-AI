import cv2
import numpy as np


class AttentionAnalyzer:

    def __init__(self, calibration_frames=60):

        # ========================================================
        # CALIBRATION
        # ========================================================

        self.calibration_frames = calibration_frames
        self.calibration_count = 0
        self.calibrated = False

        self.neutral_nose_x = 0.0
        self.neutral_nose_y = 0.0

        self.neutral_eye_h = 0.5
        self.neutral_eye_v = 0.5

        self.nose_x_values = []
        self.nose_y_values = []

        self.eye_h_values = []
        self.eye_v_values = []

        # ========================================================
        # SMOOTHING
        # ========================================================

        self.previous_score = 0.0

        self.head_history = []
        self.eye_history = []

    # ============================================================
    # GET FACE GEOMETRY
    # ============================================================

    def get_face_geometry(self, landmarks):

        # --------------------------------------------------------
        # IMPORTANT LANDMARKS
        # --------------------------------------------------------

        nose = landmarks[1]

        left_eye_outer = landmarks[33]
        left_eye_inner = landmarks[133]

        right_eye_inner = landmarks[362]
        right_eye_outer = landmarks[263]

        # --------------------------------------------------------
        # EYE CENTER
        # --------------------------------------------------------

        left_eye_center_x = (
            left_eye_outer.x +
            left_eye_inner.x
        ) / 2

        left_eye_center_y = (
            left_eye_outer.y +
            left_eye_inner.y
        ) / 2

        right_eye_center_x = (
            right_eye_inner.x +
            right_eye_outer.x
        ) / 2

        right_eye_center_y = (
            right_eye_inner.y +
            right_eye_outer.y
        ) / 2

        eye_center_x = (
            left_eye_center_x +
            right_eye_center_x
        ) / 2

        eye_center_y = (
            left_eye_center_y +
            right_eye_center_y
        ) / 2

        # --------------------------------------------------------
        # EYE DISTANCE
        # --------------------------------------------------------

        eye_distance = np.sqrt(
            (
                right_eye_center_x -
                left_eye_center_x
            ) ** 2
            +
            (
                right_eye_center_y -
                left_eye_center_y
            ) ** 2
        )

        if eye_distance < 0.001:
            eye_distance = 0.001

        # --------------------------------------------------------
        # NOSE POSITION RELATIVE TO EYES
        # --------------------------------------------------------

        nose_x = (
            nose.x -
            eye_center_x
        ) / eye_distance

        nose_y = (
            nose.y -
            eye_center_y
        ) / eye_distance

        return (
            nose_x,
            nose_y,
            eye_center_x,
            eye_center_y,
            eye_distance
        )

    # ============================================================
    # EYE GAZE
    # ============================================================

    def calculate_eye_gaze(self, landmarks):

        # --------------------------------------------------------
        # LEFT EYE
        # --------------------------------------------------------

        left_outer = landmarks[33]
        left_inner = landmarks[133]

        left_top = landmarks[159]
        left_bottom = landmarks[145]

        # --------------------------------------------------------
        # RIGHT EYE
        # --------------------------------------------------------

        right_inner = landmarks[362]
        right_outer = landmarks[263]

        right_top = landmarks[386]
        right_bottom = landmarks[374]

        # --------------------------------------------------------
        # IRIS
        # --------------------------------------------------------

        left_iris = landmarks[468:473]
        right_iris = landmarks[473:478]

        if len(left_iris) == 0 or len(right_iris) == 0:
            return "UNKNOWN", 0.5, 0.5

        # --------------------------------------------------------
        # IRIS CENTERS
        # --------------------------------------------------------

        left_iris_x = np.mean(
            [p.x for p in left_iris]
        )

        left_iris_y = np.mean(
            [p.y for p in left_iris]
        )

        right_iris_x = np.mean(
            [p.x for p in right_iris]
        )

        right_iris_y = np.mean(
            [p.y for p in right_iris]
        )

        # --------------------------------------------------------
        # HORIZONTAL EYE RATIO
        # --------------------------------------------------------

        left_min_x = min(
            left_outer.x,
            left_inner.x
        )

        left_max_x = max(
            left_outer.x,
            left_inner.x
        )

        right_min_x = min(
            right_inner.x,
            right_outer.x
        )

        right_max_x = max(
            right_inner.x,
            right_outer.x
        )

        left_width = (
            left_max_x -
            left_min_x
        )

        right_width = (
            right_max_x -
            right_min_x
        )

        if left_width <= 0 or right_width <= 0:
            return "UNKNOWN", 0.5, 0.5

        left_horizontal = (
            left_iris_x -
            left_min_x
        ) / left_width

        right_horizontal = (
            right_iris_x -
            right_min_x
        ) / right_width

        horizontal_ratio = (
            left_horizontal +
            right_horizontal
        ) / 2

        # --------------------------------------------------------
        # VERTICAL EYE RATIO
        # --------------------------------------------------------

        left_min_y = min(
            left_top.y,
            left_bottom.y
        )

        left_max_y = max(
            left_top.y,
            left_bottom.y
        )

        right_min_y = min(
            right_top.y,
            right_bottom.y
        )

        right_max_y = max(
            right_top.y,
            right_bottom.y
        )

        left_height = (
            left_max_y -
            left_min_y
        )

        right_height = (
            right_max_y -
            right_min_y
        )

        if left_height <= 0 or right_height <= 0:
            return "UNKNOWN", 0.5, 0.5

        left_vertical = (
            left_iris_y -
            left_min_y
        ) / left_height

        right_vertical = (
            right_iris_y -
            right_min_y
        ) / right_height

        vertical_ratio = (
            left_vertical +
            right_vertical
        ) / 2

        horizontal_ratio = float(
            np.clip(
                horizontal_ratio,
                0.0,
                1.0
            )
        )

        vertical_ratio = float(
            np.clip(
                vertical_ratio,
                0.0,
                1.0
            )
        )

        # --------------------------------------------------------
        # CALIBRATION
        # --------------------------------------------------------

        if not self.calibrated:

            return (
                "CENTER",
                horizontal_ratio,
                vertical_ratio
            )

        # --------------------------------------------------------
        # DIFFERENCE FROM PERSONAL BASELINE
        # --------------------------------------------------------

        horizontal_difference = (
            horizontal_ratio -
            self.neutral_eye_h
        )

        vertical_difference = (
            vertical_ratio -
            self.neutral_eye_v
        )

        # --------------------------------------------------------
        # HORIZONTAL
        # --------------------------------------------------------

        if horizontal_difference < -0.13:

            horizontal_direction = "LEFT"

        elif horizontal_difference > 0.13:

            horizontal_direction = "RIGHT"

        else:

            horizontal_direction = "CENTER"

        # --------------------------------------------------------
        # VERTICAL
        # --------------------------------------------------------

        if vertical_difference < -0.16:

            vertical_direction = "UP"

        elif vertical_difference > 0.16:

            vertical_direction = "DOWN"

        else:

            vertical_direction = "CENTER"

        # --------------------------------------------------------
        # FINAL EYE DIRECTION
        # --------------------------------------------------------

        if (
            horizontal_direction == "CENTER"
            and
            vertical_direction == "CENTER"
        ):

            direction = "CENTER"

        elif horizontal_direction != "CENTER":

            direction = horizontal_direction

        else:

            direction = vertical_direction

        return (
            direction,
            horizontal_ratio,
            vertical_ratio
        )

    # ============================================================
    # HEAD DIRECTION
    # ============================================================

    def calculate_head_pose(
        self,
        landmarks,
        frame_width,
        frame_height
    ):

        (
            nose_x,
            nose_y,
            eye_center_x,
            eye_center_y,
            eye_distance
        ) = self.get_face_geometry(
            landmarks
        )

        # ========================================================
        # BEFORE CALIBRATION
        # ========================================================

        if not self.calibrated:

            return (
                nose_x,
                nose_y,
                0.0,
                "CENTER"
            )

        # ========================================================
        # DIFFERENCE FROM NEUTRAL POSITION
        # ========================================================

        horizontal_difference = (
            nose_x -
            self.neutral_nose_x
        )

        vertical_difference = (
            nose_y -
            self.neutral_nose_y
        )

        # ========================================================
        # HEAD DIRECTION
        # ========================================================

        # Horizontal movement has priority.
        #
        # Because the webcam image is mirrored,
        # the labels are intentionally reversed here.

        if horizontal_difference > 0.16:

            direction = "LEFT"

        elif horizontal_difference < -0.16:

            direction = "RIGHT"

        elif vertical_difference > 0.22:

            direction = "DOWN"

        elif vertical_difference < -0.22:

            direction = "UP"

        else:

            direction = "CENTER"

        # ========================================================
        # RETURN
        # ========================================================

        return (
            horizontal_difference,
            vertical_difference,
            eye_distance,
            direction
        )

    # ============================================================
    # CALIBRATION
    # ============================================================

    def update_calibration(
        self,
        landmarks,
        eye_horizontal,
        eye_vertical
    ):

        if self.calibrated:
            return

        (
            nose_x,
            nose_y,
            _,
            _,
            _
        ) = self.get_face_geometry(
            landmarks
        )

        # --------------------------------------------------------
        # STORE VALUES
        # --------------------------------------------------------

        self.nose_x_values.append(
            nose_x
        )

        self.nose_y_values.append(
            nose_y
        )

        self.eye_h_values.append(
            eye_horizontal
        )

        self.eye_v_values.append(
            eye_vertical
        )

        self.calibration_count += 1

        # --------------------------------------------------------
        # FINISH CALIBRATION
        # --------------------------------------------------------

        if (
            self.calibration_count >=
            self.calibration_frames
        ):

            self.neutral_nose_x = float(
                np.median(
                    self.nose_x_values
                )
            )

            self.neutral_nose_y = float(
                np.median(
                    self.nose_y_values
                )
            )

            self.neutral_eye_h = float(
                np.median(
                    self.eye_h_values
                )
            )

            self.neutral_eye_v = float(
                np.median(
                    self.eye_v_values
                )
            )

            self.calibrated = True

            print()
            print("======================================")
            print(" ATTENTION CALIBRATION COMPLETE")
            print("======================================")

            print(
                f"Neutral Nose X: "
                f"{self.neutral_nose_x:.3f}"
            )

            print(
                f"Neutral Nose Y: "
                f"{self.neutral_nose_y:.3f}"
            )

            print(
                f"Neutral Eye H: "
                f"{self.neutral_eye_h:.3f}"
            )

            print(
                f"Neutral Eye V: "
                f"{self.neutral_eye_v:.3f}"
            )

            print("======================================")
            print()

    # ============================================================
    # CALIBRATION PROGRESS
    # ============================================================

    def get_calibration_progress(self):

        if self.calibrated:

            return 100.0

        return min(
            100.0,
            (
                self.calibration_count /
                self.calibration_frames
            ) * 100
        )

    # ============================================================
    # ATTENTION SCORE
    # ============================================================

    def calculate_attention_score(
        self,
        head_direction,
        eye_direction
    ):

        # --------------------------------------------------------
        # HEAD SCORE
        # --------------------------------------------------------

        if head_direction == "CENTER":

            head_score = 60

        elif head_direction in [
            "LEFT",
            "RIGHT"
        ]:

            head_score = 25

        elif head_direction in [
            "UP",
            "DOWN"
        ]:

            head_score = 15

        else:

            head_score = 0

        # --------------------------------------------------------
        # EYE SCORE
        # --------------------------------------------------------

        if eye_direction == "CENTER":

            eye_score = 40

        elif eye_direction in [
            "LEFT",
            "RIGHT"
        ]:

            eye_score = 15

        elif eye_direction in [
            "UP",
            "DOWN"
        ]:

            eye_score = 10

        else:

            eye_score = 0

        # --------------------------------------------------------
        # TOTAL
        # --------------------------------------------------------

        score = (
            head_score +
            eye_score
        )

        score = max(
            0,
            min(
                100,
                score
            )
        )

        # --------------------------------------------------------
        # SMOOTHING
        # --------------------------------------------------------

        self.previous_score = (
            0.8 *
            self.previous_score
            +
            0.2 *
            score
        )

        return self.previous_score

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(self, score):

        if score >= 75:

            return "FOCUSED"

        elif score >= 50:

            return "PARTIALLY FOCUSED"

        elif score >= 30:

            return "DISTRACTED"

        else:

            return "NOT ATTENTIVE"