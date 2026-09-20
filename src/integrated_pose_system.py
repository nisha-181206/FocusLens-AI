import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from posture_analysis import PostureAnalyzer
from ergonomics import ErgonomicAnalyzer
from activity_detection import ActivityFeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

POSE_MODEL = "models/pose_landmarker.task"
ACTIVITY_MODEL = "models/activity_classifier.joblib"


# ============================================================
# LOAD POSE MODEL
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

landmarker = vision.PoseLandmarker.create_from_options(
    options
)


# ============================================================
# LOAD ACTIVITY MODEL
# ============================================================

activity_model = joblib.load(
    ACTIVITY_MODEL
)

print("Activity model loaded successfully.")


# ============================================================
# INITIALIZE ANALYZERS
# ============================================================

posture_analyzer = PostureAnalyzer(
    smoothing_window=15
)

ergonomic_analyzer = ErgonomicAnalyzer(
    bad_posture_limit=10
)

feature_extractor = ActivityFeatureExtractor()


# ============================================================
# ACTIVITY SMOOTHING
# ============================================================

activity_history = deque(maxlen=10)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print()
print("========================================")
print(" FocusLens AI")
print(" Integrated Pose Analysis")
print("========================================")
print()
print("Camera opened successfully.")
print("Press Q to quit.")
print()


# ============================================================
# TIMESTAMP
# ============================================================

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")
        break


    # --------------------------------------------------------
    # MIRROR IMAGE
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # IMAGE CONVERSION
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
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    # --------------------------------------------------------
    # POSE DETECTION
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # NO PERSON DETECTED
    # ========================================================

    if not result.pose_landmarks:

        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "FocusLens AI - Integrated System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # ========================================================
    # GET LANDMARKS
    # ========================================================

    landmarks = result.pose_landmarks[0]


    # ========================================================
    # POSTURE ANALYSIS
    # ========================================================

    posture_result = posture_analyzer.analyze(
        landmarks
    )


    # ========================================================
    # ERGONOMIC ANALYSIS
    # ========================================================

    ergonomic_result = ergonomic_analyzer.analyze(

        posture_result["neck_angle"],

        posture_result["shoulder_angle"],

        posture_result["torso_angle"],

        posture_result["posture"]
    )


    # ========================================================
    # ACTIVITY FEATURE EXTRACTION
    # ========================================================

    features = feature_extractor.extract(
        landmarks
    )


    # ========================================================
    # ACTIVITY PREDICTION
    # ========================================================

    features_2d = features.reshape(
        1,
        -1
    )


    activity_prediction = activity_model.predict(
        features_2d
    )[0]


    # --------------------------------------------------------
    # ACTIVITY PROBABILITY
    # --------------------------------------------------------

    probabilities = activity_model.predict_proba(
        features_2d
    )[0]

    activity_confidence = (
        np.max(probabilities) * 100
    )


    # ========================================================
    # ACTIVITY SMOOTHING
    # ========================================================

    activity_history.append(
        activity_prediction
    )


    activity_counts = {}

    for activity in activity_history:

        activity_counts[activity] = (
            activity_counts.get(activity, 0) + 1
        )


    stable_activity = max(
        activity_counts,
        key=activity_counts.get
    )


    # ========================================================
    # DRAW POSE LANDMARKS
    # ========================================================

    h, w, _ = frame.shape


    for landmark in landmarks:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        if (
            0 <= x < w
            and
            0 <= y < h
        ):

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


    # ========================================================
    # DASHBOARD PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (470, 360),
        (20, 20, 20),
        -1
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "FOCUSLENS AI",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Real-Time Human Analysis",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )


    # ========================================================
    # ACTIVITY
    # ========================================================

    cv2.putText(
        frame,
        f"Activity: {stable_activity}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {activity_confidence:.1f}%",
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )


    # ========================================================
    # POSTURE
    # ========================================================

    cv2.putText(
        frame,
        f"Posture: {posture_result['posture']}",
        (25, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # ANGLES
    # ========================================================

    cv2.putText(
        frame,
        f"Neck: {posture_result['neck_angle']:.1f}",
        (25, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


    cv2.putText(
        frame,
        f"Shoulder: {posture_result['shoulder_angle']:.1f}",
        (25, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


    cv2.putText(
        frame,
        f"Torso: {posture_result['torso_angle']:.1f}",
        (25, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


    # ========================================================
    # ERGONOMIC STATUS
    # ========================================================

    cv2.putText(
        frame,
        f"Ergonomic Risk: {ergonomic_result['risk']}",
        (25, 275),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # BAD POSTURE TIMER
    # ========================================================

    cv2.putText(
        frame,
        f"Bad Posture: "
        f"{ergonomic_result['bad_duration']:.1f}s",
        (25, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if ergonomic_result["risk"] == "HIGH RISK":

        message = "WARNING: CORRECT POSTURE"

        cv2.putText(
            frame,
            message,
            (25, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    elif ergonomic_result["risk"] == "MODERATE RISK":

        message = "ADJUST YOUR POSTURE"

        cv2.putText(
            frame,
            message,
            (25, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

    else:

        message = "POSTURE LOOKS GOOD"

        cv2.putText(
            frame,
            message,
            (25, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "FocusLens AI - Integrated System",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print()
print("FocusLens AI stopped.")