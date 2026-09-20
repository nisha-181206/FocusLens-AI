import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from posture_analysis import PostureAnalyzer


# ==================================================
# MODEL
# ==================================================

MODEL_PATH = "models/pose_landmarker.task"

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


# ==================================================
# POSTURE ANALYZER
# ==================================================

posture_analyzer = PostureAnalyzer(
    smoothing_window=15
)


# ==================================================
# CAMERA
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    landmarker.close()

    exit()


print("======================================")
print("FocusLens AI - Posture Analysis")
print("======================================")
print("Camera opened successfully.")
print("Press Q to exit.")


# ==================================================
# TIMESTAMP
# ==================================================

timestamp = 0


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera.")

        break

    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    # ----------------------------------------------
    # Convert BGR → RGB
    # ----------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # ----------------------------------------------
    # Timestamp
    # ----------------------------------------------

    timestamp += 33

    # ----------------------------------------------
    # Pose detection
    # ----------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ==================================================
    # BODY DETECTED
    # ==================================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # ----------------------------------------------
        # Analyze posture
        # ----------------------------------------------

        data = posture_analyzer.analyze(
            landmarks
        )

        posture = data["posture"]

        neck_angle = data["neck_angle"]

        shoulder_angle = data["shoulder_angle"]

        torso_angle = data["torso_angle"]

        stability = data["stability"]


        # ==================================================
        # DRAW LANDMARKS
        # ==================================================

        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            if (
                0 <= x < width
                and
                0 <= y < height
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


        # ==================================================
        # IMPORTANT POINTS
        # ==================================================

        nose = landmarks[0]

        left_shoulder = landmarks[11]

        right_shoulder = landmarks[12]

        left_hip = landmarks[23]

        right_hip = landmarks[24]


        nose_point = (
            int(nose.x * width),
            int(nose.y * height)
        )

        left_shoulder_point = (
            int(left_shoulder.x * width),
            int(left_shoulder.y * height)
        )

        right_shoulder_point = (
            int(right_shoulder.x * width),
            int(right_shoulder.y * height)
        )

        left_hip_point = (
            int(left_hip.x * width),
            int(left_hip.y * height)
        )

        right_hip_point = (
            int(right_hip.x * width),
            int(right_hip.y * height)
        )


        # ==================================================
        # DRAW SHOULDERS
        # ==================================================

        cv2.line(
            frame,
            left_shoulder_point,
            right_shoulder_point,
            (255, 0, 0),
            3
        )


        # ==================================================
        # DRAW TORSO
        # ==================================================

        cv2.line(
            frame,
            left_shoulder_point,
            left_hip_point,
            (255, 255, 0),
            2
        )

        cv2.line(
            frame,
            right_shoulder_point,
            right_hip_point,
            (255, 255, 0),
            2
        )


        # ==================================================
        # NOSE
        # ==================================================

        cv2.circle(
            frame,
            nose_point,
            6,
            (0, 0, 255),
            -1
        )


        # ==================================================
        # INFORMATION PANEL
        # ==================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (460, 210),
            (20, 20, 20),
            -1
        )


        cv2.putText(
            frame,
            f"Posture: {posture}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Neck Angle: {neck_angle:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Shoulder Angle: {shoulder_angle:.1f}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Torso Angle: {torso_angle:.1f}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Stability: {stability:.0f}%",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 0),
            2
        )


    # ==================================================
    # NO BODY
    # ==================================================

    else:

        cv2.putText(
            frame,
            "NO BODY DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow(
        "FocusLens AI - Posture Analysis",
        frame
    )


    # ==================================================
    # EXIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==================================================
# CLEANUP
# ==================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Posture analysis stopped.")