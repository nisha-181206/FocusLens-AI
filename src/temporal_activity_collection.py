import cv2
import csv
import os
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from temporal_activity_features import TemporalActivityFeatureExtractor


# ============================================================
# SETTINGS
# ============================================================

POSE_MODEL = "models/pose_landmarker.task"

OUTPUT_DIR = "data/temporal"
OUTPUT_FILE = "data/temporal/activity_temporal_dataset.csv"

SEQUENCE_LENGTH = 30

# Number of sequences to collect for each activity
TARGET_SEQUENCES = 60


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ACTIVITIES
# ============================================================

activities = {
    ord("1"): "SITTING",
    ord("2"): "STANDING",
    ord("3"): "IDLE",
    ord("4"): "MOVING",
    ord("5"): "PHONE_USAGE"
}


# ============================================================
# TEMPORAL FEATURE EXTRACTOR
# ============================================================

extractor = TemporalActivityFeatureExtractor()


# ============================================================
# FEATURE NAMES
# ============================================================

feature_count = 138

feature_names = [
    f"feature_{i:03d}"
    for i in range(feature_count)
]


# ============================================================
# CSV
# ============================================================

file_exists = os.path.exists(
    OUTPUT_FILE
)

csv_file = open(
    OUTPUT_FILE,
    "a",
    newline=""
)

writer = csv.writer(
    csv_file
)

if not file_exists:

    writer.writerow(
        feature_names + ["label"]
    )


# ============================================================
# MEDIAPIPE POSE
# ============================================================

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

pose_landmarker = (
    vision.PoseLandmarker.create_from_options(
        options
    )
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    csv_file.close()
    pose_landmarker.close()
    exit()


# ============================================================
# VARIABLES
# ============================================================

current_activity = None

sequence = []

sequence_count = 0

timestamp_ms = 0


print()
print("==========================================")
print("FOCUSLENS AI - TEMPORAL DATA COLLECTION")
print("==========================================")

print()
print("Controls:")
print("1 -> SITTING")
print("2 -> STANDING")
print("3 -> IDLE")
print("4 -> MOVING")
print("5 -> PHONE_USAGE")
print("Q -> QUIT")

print()
print(
    f"Each sample = {SEQUENCE_LENGTH} frames"
)

print(
    f"Target = {TARGET_SEQUENCES} sequences/activity"
)

print()
print("Start by pressing 1, 2, 3, 4 or 5.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read frame."
        )

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "1:SIT  2:STAND  3:IDLE  4:MOVE  5:PHONE  Q:QUIT",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    if current_activity is None:

        cv2.putText(
            frame,
            "SELECT ACTIVITY",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Activity: {current_activity}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Sequences: {sequence_count}/{TARGET_SEQUENCES}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # CONVERT TO MEDIAPIPE IMAGE
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # POSE DETECTION
    # --------------------------------------------------------

    timestamp_ms += 33

    result = pose_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # EXTRACT CURRENT FRAME FEATURES
    # --------------------------------------------------------

    if (
        current_activity is not None
        and result.pose_landmarks
    ):

        landmarks = result.pose_landmarks[0]

        frame_features = (
            extractor.extract_frame_features(
                landmarks
            )
        )

        sequence.append(
            frame_features
        )


    # --------------------------------------------------------
    # COMPLETE ONE SEQUENCE
    # --------------------------------------------------------

    if len(sequence) >= SEQUENCE_LENGTH:

        temporal_features = (
            extractor.extract_sequence_features(
                sequence
            )
        )

        writer.writerow(
            list(temporal_features)
            + [current_activity]
        )

        csv_file.flush()

        sequence_count += 1

        print(
            f"{current_activity}: "
            f"sequence {sequence_count}/{TARGET_SEQUENCES}"
        )

        # Start a completely new sequence
        sequence = []


        # ----------------------------------------------------
        # ACTIVITY COMPLETE
        # ----------------------------------------------------

        if sequence_count >= TARGET_SEQUENCES:

            print()
            print(
                f"{current_activity} completed."
            )

            print(
                "Press another activity key "
                "to continue."
            )

            current_activity = None

            sequence_count = 0


    # --------------------------------------------------------
    # DRAW POSE
    # --------------------------------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        h, w, _ = frame.shape

        for landmark in landmarks:

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


    # --------------------------------------------------------
    # SHOW WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "FocusLens AI - Temporal Data Collection",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # SELECT ACTIVITY
    # --------------------------------------------------------

    if key in activities:

        current_activity = activities[key]

        sequence = []

        sequence_count = 0

        print()
        print(
            "=========================================="
        )

        print(
            f"Collecting: {current_activity}"
        )

        print(
            f"Target sequences: {TARGET_SEQUENCES}"
        )

        print(
            "Perform the activity naturally."
        )

        print(
            "=========================================="
        )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

csv_file.close()

pose_landmarker.close()


print()
print("==========================================")
print("TEMPORAL DATA COLLECTION FINISHED")
print("==========================================")

print()
print(
    f"Dataset saved at:\n{OUTPUT_FILE}"
)