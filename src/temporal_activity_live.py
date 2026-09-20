import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from temporal_activity_features import TemporalActivityFeatureExtractor


# ==========================================
# PATHS
# ==========================================

POSE_MODEL = "models/pose_landmarker.task"
ACTIVITY_MODEL = "models/temporal_activity_classifier.joblib"


# ==========================================
# SETTINGS
# ==========================================

SEQUENCE_LENGTH = 30
SMOOTHING_WINDOW = 5


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading activity model...")

model = joblib.load(ACTIVITY_MODEL)

print("Activity model loaded successfully.")


# ==========================================
# FEATURE EXTRACTOR
# ==========================================

extractor = TemporalActivityFeatureExtractor()


# ==========================================
# MEDIAPIPE POSE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    options
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    pose_landmarker.close()
    exit()


# ==========================================
# VARIABLES
# ==========================================

sequence = []

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)

timestamp_ms = 0

current_activity = "COLLECTING..."

current_confidence = 0.0


# ==========================================
# MAIN LOOP
# ==========================================

print()
print("==========================================")
print("FOCUSLENS AI - LIVE ACTIVITY V2")
print("==========================================")
print()
print("30 frames are required for each prediction.")
print("Press Q to quit.")
print()


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # --------------------------------------
    # Convert to MediaPipe image
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp_ms += 33


    # --------------------------------------
    # Pose detection
    # --------------------------------------

    result = pose_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------
    # Process pose
    # --------------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]


        # Extract frame features
        frame_features = (
            extractor.extract_frame_features(
                landmarks
            )
        )


        sequence.append(frame_features)


        # ----------------------------------
        # Draw landmarks
        # ----------------------------------

        h, w, _ = frame.shape

        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            if (
                0 <= x < w
                and 0 <= y < h
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


        # ----------------------------------
        # Prediction
        # ----------------------------------

        if len(sequence) >= SEQUENCE_LENGTH:

            temporal_features = (
                extractor.extract_sequence_features(
                    sequence[-SEQUENCE_LENGTH:]
                )
            )


            # Reshape for model
            temporal_features = (
                temporal_features.reshape(1, -1)
            )


            # Prediction
            prediction = model.predict(
                temporal_features
            )[0]


            # Probability
            probabilities = model.predict_proba(
                temporal_features
            )[0]


            current_confidence = (
                np.max(probabilities) * 100
            )


            # Smoothing
            prediction_history.append(
                prediction
            )


            counts = {}

            for item in prediction_history:

                counts[item] = (
                    counts.get(item, 0) + 1
                )


            current_activity = max(
                counts,
                key=counts.get
            )


    else:

        current_activity = "NO BODY DETECTED"

        current_confidence = 0.0

        sequence.clear()

        prediction_history.clear()


    # ======================================
    # DASHBOARD
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (470, 150),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "FOCUSLENS AI",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Activity: {current_activity}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {current_confidence:.1f}%",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frames: {min(len(sequence), SEQUENCE_LENGTH)}/{SEQUENCE_LENGTH}",
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "FocusLens AI - Live Activity V2",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

pose_landmarker.close()

print()
print("Live activity detection stopped.")