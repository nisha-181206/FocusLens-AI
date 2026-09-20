import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. Model
# --------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# 2. Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Head Pose Detection Started")
print("Press Q to exit")

frame_timestamp = 0


# --------------------------------------------------
# 3. Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    frame_timestamp += 33

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # --------------------------------------------------
    # 4. Face detected
    # --------------------------------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        h, w, _ = frame.shape


        # --------------------------------------------------
        # Important facial landmarks
        # --------------------------------------------------

        nose = landmarks[1]
        forehead = landmarks[10]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]


        # --------------------------------------------------
        # Convert to image coordinates
        # --------------------------------------------------

        nose_point = np.array([
            nose.x * w,
            nose.y * h
        ])

        forehead_point = np.array([
            forehead.x * w,
            forehead.y * h
        ])

        chin_point = np.array([
            chin.x * w,
            chin.y * h
        ])

        left_eye_point = np.array([
            left_eye.x * w,
            left_eye.y * h
        ])

        right_eye_point = np.array([
            right_eye.x * w,
            right_eye.y * h
        ])


        # --------------------------------------------------
        # Eye-line angle
        # --------------------------------------------------

        dx = right_eye_point[0] - left_eye_point[0]
        dy = right_eye_point[1] - left_eye_point[1]

        roll = np.degrees(
            np.arctan2(dy, dx)
        )


        # --------------------------------------------------
        # Face center
        # --------------------------------------------------

        eye_center = (
            left_eye_point + right_eye_point
        ) / 2

        face_height = np.linalg.norm(
            chin_point - forehead_point
        )


        if face_height > 0:

            vertical_position = (
                nose_point[1] - eye_center[1]
            ) / face_height

            horizontal_position = (
                nose_point[0] - eye_center[0]
            ) / face_height


            # --------------------------------------------------
            # Estimate pitch
            # --------------------------------------------------

            pitch = vertical_position * 100


            # --------------------------------------------------
            # Estimate yaw
            # --------------------------------------------------

            yaw = horizontal_position * 100


            # --------------------------------------------------
            # Head direction
            # --------------------------------------------------

            if yaw > 8:
                direction = "RIGHT"

            elif yaw < -8:
                direction = "LEFT"

            elif pitch > 12:
                direction = "DOWN"

            elif pitch < -8:
                direction = "UP"

            else:
                direction = "CENTER"


            # --------------------------------------------------
            # Draw nose point
            # --------------------------------------------------

            cv2.circle(
                frame,
                tuple(nose_point.astype(int)),
                6,
                (0, 255, 0),
                -1
            )


            # --------------------------------------------------
            # Display values
            # --------------------------------------------------

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Direction: {direction}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


    else:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    cv2.imshow(
        "FocusLens AI - Head Pose",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Head Pose Detection stopped.")