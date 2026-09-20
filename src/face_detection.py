import cv2
import time
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
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# 3. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Face Landmarker started!")
print("Press Q to exit.")


# --------------------------------------------------
# 4. Process webcam frames
# --------------------------------------------------

frame_timestamp = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp must continuously increase
    frame_timestamp += 33

    # Detect face landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    # --------------------------------------------------
    # 5. Draw landmarks manually
    # --------------------------------------------------

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        height, width, _ = frame.shape

        for landmark in face_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

        # Face detected message

        cv2.putText(
            frame,
            "Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Landmarks: {len(face_landmarks)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
    # 6. Display
    # --------------------------------------------------

    cv2.imshow(
        "FocusLens AI - Face Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 7. Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Face Landmarker stopped.")