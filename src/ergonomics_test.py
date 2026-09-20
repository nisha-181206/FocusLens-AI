import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from posture_analysis import PostureAnalyzer
from ergonomics import ErgonomicAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/pose_landmarker.task"


# ============================================================
# MEDIAPIPE POSE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
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
# ANALYZERS
# ============================================================

posture_analyzer = PostureAnalyzer(
    smoothing_window=15
)

ergonomic_analyzer = ErgonomicAnalyzer(
    bad_posture_limit=10
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully!")
print("Ergonomic analysis started.")
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp
    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    # Detect pose
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # --------------------------------------------------------
    # NO BODY DETECTED
    # --------------------------------------------------------

    if not result.pose_landmarks:

        cv2.putText(
            frame,
            "NO BODY DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "FocusLens AI - Ergonomic Analysis",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # --------------------------------------------------------
    # GET LANDMARKS
    # --------------------------------------------------------

    landmarks = result.pose_landmarks[0]

    # Analyze posture
    posture_result = posture_analyzer.analyze(
        landmarks
    )

    # Analyze ergonomics
    ergonomic_result = ergonomic_analyzer.analyze(
        posture_result["neck_angle"],
        posture_result["shoulder_angle"],
        posture_result["torso_angle"],
        posture_result["posture"]
    )

    # --------------------------------------------------------
    # DRAW LANDMARKS
    # --------------------------------------------------------

    h, w, _ = frame.shape

    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        if 0 <= x < w and 0 <= y < h:

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    y_position = 30

    cv2.putText(
        frame,
        f"Posture: {posture_result['posture']}",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y_position += 30

    cv2.putText(
        frame,
        f"Neck Angle: {posture_result['neck_angle']:.1f}",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y_position += 30

    cv2.putText(
        frame,
        f"Shoulder Angle: {posture_result['shoulder_angle']:.1f}",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y_position += 30

    cv2.putText(
        frame,
        f"Torso Angle: {posture_result['torso_angle']:.1f}",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y_position += 30

    cv2.putText(
        frame,
        f"Risk: {ergonomic_result['risk']}",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    y_position += 30

    cv2.putText(
        frame,
        f"Bad Posture: {ergonomic_result['bad_duration']:.1f}s",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SHOW RECOMMENDATION
    # --------------------------------------------------------

    if ergonomic_result["risk"] == "HIGH RISK":

        cv2.putText(
            frame,
            "WARNING: CORRECT YOUR POSTURE!",
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

    elif ergonomic_result["risk"] == "MODERATE RISK":

        cv2.putText(
            frame,
            "Adjust your sitting position",
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Posture looks good",
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # DISPLAY WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "FocusLens AI - Ergonomic Analysis",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Ergonomic analysis stopped.")