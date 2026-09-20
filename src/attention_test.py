import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from attention_analysis import AttentionAnalyzer


MODEL_PATH = "models/face_landmarker.task"


# ============================================================
# MEDIAPIPE
# ============================================================

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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# ATTENTION ANALYZER
# ============================================================

analyzer = AttentionAnalyzer(
    calibration_frames=60
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print()
print("========================================")
print(" FOCUSLENS AI - ATTENTION TEST")
print("========================================")
print()
print("Look naturally at the screen.")
print("Do not move during calibration.")
print()
print("Press Q to quit.")
print()


start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break

    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    # BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(
        (time.time() - start_time) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # ========================================================
    # NO FACE
    # ========================================================

    if not result.face_landmarks:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "FocusLens AI - Attention Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # ========================================================
    # LANDMARKS
    # ========================================================

    landmarks = result.face_landmarks[0]

    # ========================================================
    # EYE GAZE
    # ========================================================

    (
        eye_direction,
        gaze_h,
        gaze_v
    ) = analyzer.calculate_eye_gaze(
        landmarks
    )

    # ========================================================
    # HEAD GEOMETRY
    # ========================================================

    (
        head_x,
        head_y,
        face_scale,
        head_direction
    ) = analyzer.calculate_head_pose(
        landmarks,
        width,
        height
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    if not analyzer.calibrated:

        analyzer.update_calibration(
            landmarks,
            gaze_h,
            gaze_v
        )

        progress = (
            analyzer
            .get_calibration_progress()
        )

        # ----------------------------------------------------
        # CALIBRATION SCREEN
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (560, 190),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "FOCUSLENS AI",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "CALIBRATING...",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Look naturally at the screen",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Progress: {progress:.0f}%",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ========================================================
    # AFTER CALIBRATION
    # ========================================================

    else:

        score = analyzer.calculate_attention_score(
            head_direction,
            eye_direction
        )

        status = analyzer.get_status(
            score
        )

        # ----------------------------------------------------
        # DRAW IMPORTANT LANDMARKS
        # ----------------------------------------------------

        important_points = [
            1,
            33,
            133,
            263,
            362,
            468,
            473,
            152,
            61,
            291
        ]

        for index in important_points:

            point = landmarks[index]

            x = int(
                point.x * width
            )

            y = int(
                point.y * height
            )

            if (
                0 <= x < width
                and
                0 <= y < height
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (570, 390),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "FOCUSLENS AI",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Attention: {score:.1f}%",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Eye Direction: {eye_direction}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"Head Direction: {head_direction}",
            (30, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        # ----------------------------------------------------
        # HEAD MOVEMENT VALUES
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Head X: {head_x:.3f}",
            (30, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            f"Head Y: {head_y:.3f}",
            (200, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            f"Face Scale: {face_scale:.3f}",
            (360, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # ----------------------------------------------------
        # GAZE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Gaze H: {gaze_h:.2f}",
            (30, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        cv2.putText(
            frame,
            f"Gaze V: {gaze_v:.2f}",
            (190, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        if status == "FOCUSED":

            message = "FOCUSED ON SCREEN"

        elif status == "PARTIALLY FOCUSED":

            message = "PARTIALLY FOCUSED"

        elif status == "DISTRACTED":

            message = "LOOKING AWAY"

        else:

            message = "NOT ATTENTIVE"

        cv2.putText(
            frame,
            message,
            (30, 325),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Move head LEFT / RIGHT / UP / DOWN",
            (30, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "FocusLens AI - Attention Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print()
print("Attention detection stopped.")