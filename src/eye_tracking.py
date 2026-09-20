import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. Model path
# --------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"


# --------------------------------------------------
# 2. Create Face Landmarker
# --------------------------------------------------

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
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# 3. Webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Eye Tracking Started")
print("Press Q to exit")


frame_timestamp = 0


# --------------------------------------------------
# 4. Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp += 33

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # --------------------------------------------------
    # 5. Face detected
    # --------------------------------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        height, width, _ = frame.shape


        # --------------------------------------------------
        # Eye landmark indices
        # --------------------------------------------------

        # Left eye
        left_eye_indices = [
            33, 133, 159, 145
        ]

        # Right eye
        right_eye_indices = [
            362, 263, 386, 374
        ]

        # Left iris
        left_iris_indices = [
            468, 469, 470, 471, 472
        ]

        # Right iris
        right_iris_indices = [
            473, 474, 475, 476, 477
        ]


        # --------------------------------------------------
        # Draw eye landmarks
        # --------------------------------------------------

        for index in left_eye_indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 0, 0),
                -1
            )


        for index in right_eye_indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 0, 0),
                -1
            )


        # --------------------------------------------------
        # Draw iris landmarks
        # --------------------------------------------------

        for index in left_iris_indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 255),
                -1
            )


        for index in right_iris_indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 255),
                -1
            )


        # --------------------------------------------------
        # Calculate iris centers
        # --------------------------------------------------

        left_iris_x = sum(
            landmarks[i].x for i in left_iris_indices
        ) / len(left_iris_indices)

        left_iris_y = sum(
            landmarks[i].y for i in left_iris_indices
        ) / len(left_iris_indices)


        right_iris_x = sum(
            landmarks[i].x for i in right_iris_indices
        ) / len(right_iris_indices)

        right_iris_y = sum(
            landmarks[i].y for i in right_iris_indices
        ) / len(right_iris_indices)


        # Convert to pixels

        left_x = int(left_iris_x * width)
        left_y = int(left_iris_y * height)

        right_x = int(right_iris_x * width)
        right_y = int(right_iris_y * height)


        # --------------------------------------------------
        # Display iris centers
        # --------------------------------------------------

        cv2.circle(
            frame,
            (left_x, left_y),
            6,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (right_x, right_y),
            6,
            (0, 255, 0),
            -1
        )


        # --------------------------------------------------
        # Display status
        # --------------------------------------------------

        cv2.putText(
            frame,
            "Eyes Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
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
        "FocusLens AI - Eye Tracking",
        frame
    )


    # --------------------------------------------------
    # Quit
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Eye Tracking stopped.")