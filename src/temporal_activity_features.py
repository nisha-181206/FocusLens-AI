import numpy as np


class TemporalActivityFeatureExtractor:

    def __init__(self):
        self.base_feature_count = 15

    # =========================================================
    # DISTANCE
    # =========================================================

    def distance(self, a, b):

        return np.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2
        )

    # =========================================================
    # ANGLE
    # =========================================================

    def angle(self, a, b, c):

        ba = np.array([
            a.x - b.x,
            a.y - b.y
        ])

        bc = np.array([
            c.x - b.x,
            c.y - b.y
        ])

        denominator = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denominator == 0:
            return 0.0

        cosine = np.dot(ba, bc) / denominator

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return np.degrees(
            np.arccos(cosine)
        )

    # =========================================================
    # SINGLE-FRAME FEATURES
    # =========================================================

    def extract_frame_features(self, landmarks):

        nose = landmarks[0]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_elbow = landmarks[13]
        right_elbow = landmarks[14]

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        left_knee = landmarks[25]
        right_knee = landmarks[26]

        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # -----------------------------------------------------
        # Distances
        # -----------------------------------------------------

        shoulder_width = self.distance(
            left_shoulder,
            right_shoulder
        )

        hip_width = self.distance(
            left_hip,
            right_hip
        )

        nose_to_shoulder = (
            self.distance(
                nose,
                left_shoulder
            )
            +
            self.distance(
                nose,
                right_shoulder
            )
        ) / 2

        wrist_distance = self.distance(
            left_wrist,
            right_wrist
        )

        wrist_to_nose_left = self.distance(
            left_wrist,
            nose
        )

        wrist_to_nose_right = self.distance(
            right_wrist,
            nose
        )

        # -----------------------------------------------------
        # Joint angles
        # -----------------------------------------------------

        left_elbow_angle = self.angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_elbow_angle = self.angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        left_knee_angle = self.angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = self.angle(
            right_hip,
            right_knee,
            right_ankle
        )

        # -----------------------------------------------------
        # Body height
        # -----------------------------------------------------

        body_height = (
            self.distance(
                nose,
                left_ankle
            )
            +
            self.distance(
                nose,
                right_ankle
            )
        ) / 2

        # -----------------------------------------------------
        # Vertical positions
        # -----------------------------------------------------

        shoulder_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2

        hip_y = (
            left_hip.y +
            right_hip.y
        ) / 2

        knee_y = (
            left_knee.y +
            right_knee.y
        ) / 2

        wrist_y = (
            left_wrist.y +
            right_wrist.y
        ) / 2

        return np.array([
            shoulder_width,
            hip_width,
            nose_to_shoulder,
            wrist_distance,
            wrist_to_nose_left,
            wrist_to_nose_right,
            left_elbow_angle,
            right_elbow_angle,
            left_knee_angle,
            right_knee_angle,
            body_height,
            shoulder_y,
            hip_y,
            knee_y,
            wrist_y
        ], dtype=np.float32)

    # =========================================================
    # TEMPORAL FEATURES
    # =========================================================

    def extract_sequence_features(self, sequence):

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.ndim != 2:

            raise ValueError(
                "Sequence must have shape "
                "(number_of_frames, number_of_features)"
            )

        if sequence.shape[1] != self.base_feature_count:

            raise ValueError(
                f"Expected {self.base_feature_count} "
                f"features per frame, but received "
                f"{sequence.shape[1]}"
            )

        # -----------------------------------------------------
        # Spatial statistics
        # -----------------------------------------------------

        mean_features = np.mean(
            sequence,
            axis=0
        )

        std_features = np.std(
            sequence,
            axis=0
        )

        min_features = np.min(
            sequence,
            axis=0
        )

        max_features = np.max(
            sequence,
            axis=0
        )

        range_features = (
            max_features -
            min_features
        )

        # -----------------------------------------------------
        # Frame-to-frame changes
        # -----------------------------------------------------

        differences = np.diff(
            sequence,
            axis=0
        )

        if len(differences) > 0:

            mean_change = np.mean(
                np.abs(differences),
                axis=0
            )

            std_change = np.std(
                differences,
                axis=0
            )

            max_change = np.max(
                np.abs(differences),
                axis=0
            )

            total_change = np.sum(
                np.abs(differences),
                axis=0
            )

        else:

            mean_change = np.zeros(
                self.base_feature_count
            )

            std_change = np.zeros(
                self.base_feature_count
            )

            max_change = np.zeros(
                self.base_feature_count
            )

            total_change = np.zeros(
                self.base_feature_count
            )

        # -----------------------------------------------------
        # Overall movement
        # -----------------------------------------------------

        if len(differences) > 0:

            overall_movement = np.mean(
                np.abs(differences)
            )

            movement_std = np.std(
                np.abs(differences)
            )

            movement_max = np.max(
                np.abs(differences)
            )

        else:

            overall_movement = 0.0
            movement_std = 0.0
            movement_max = 0.0

        # -----------------------------------------------------
        # Combine features
        # -----------------------------------------------------

        features = np.concatenate([
            mean_features,
            std_features,
            min_features,
            max_features,
            range_features,
            mean_change,
            std_change,
            max_change,
            total_change,
            np.array([
                overall_movement,
                movement_std,
                movement_max
            ])
        ])

        return features.astype(
            np.float32
        )