import cv2
import time
import traceback
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from attention_analysis import AttentionAnalyzer
from posture_analysis import PostureAnalyzer
from temporal_activity_engine import TemporalActivityEngine
from ergonomics import ErgonomicAnalyzer
from session_tracker import SessionTracker
from blink_detection import BlinkDetector
from drowsiness_detection import DrowsinessDetector
from smart_alerts import SmartAlertSystem
from fps_counter import FPSCounter
from detection_status import DetectionStatus
from calibration_status import CalibrationStatus
from distraction_tracker import DistractionTracker
from live_recommendation import LiveRecommendation


# ============================================================
# PATHS
# ============================================================

FACE_MODEL = "models/face_landmarker.task"
POSE_MODEL = "models/pose_landmarker.task"
ACTIVITY_MODEL = "models/temporal_activity_classifier.joblib"


# ============================================================
# DISPLAY SETTINGS
# ============================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CAMERA_WIDTH = 900
CAMERA_HEIGHT = 720

PANEL_WIDTH = DISPLAY_WIDTH - CAMERA_WIDTH


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Camera resolution
CAMERA_CAPTURE_WIDTH = 640
CAMERA_CAPTURE_HEIGHT = 480

# MediaPipe processing resolution
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

# Face processing frequency
# 1 = every frame
# 2 = every second frame
FACE_PROCESS_INTERVAL = 2

# Pose is more expensive, so process less frequently
POSE_PROCESS_INTERVAL = 4

# Do not draw all 478 face landmarks
DRAW_FULL_FACE_MESH = False

# Pose drawing is also kept lightweight
DRAW_POSE = True


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

face_base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    face_options
)


# ============================================================
# MEDIAPIPE POSE LANDMARKER
# ============================================================

pose_base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# ANALYZERS
# ============================================================

attention_analyzer = AttentionAnalyzer()

posture_analyzer = PostureAnalyzer()

activity_engine = TemporalActivityEngine(
    ACTIVITY_MODEL,
    sequence_length=30,
    smoothing_window=5
)

ergonomic_analyzer = ErgonomicAnalyzer(
    bad_posture_limit=10
)

session_tracker = SessionTracker()

blink_detector = BlinkDetector()

drowsiness_detector = DrowsinessDetector()

smart_alerts = SmartAlertSystem(
    distraction_limit=2.0,
    posture_limit=5.0,
    critical_distraction_limit=5.0,
    cooldown=5.0
)

fps_counter = FPSCounter()

detection_status = DetectionStatus()

calibration_status = CalibrationStatus(
    total_frames=60
)

distraction_tracker = DistractionTracker()

live_recommendation = LiveRecommendation()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_CAPTURE_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_CAPTURE_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)

# Reduce camera buffering
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "FocusLens AI"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)


# ============================================================
# SESSION VARIABLES
# ============================================================

start_time = time.time()

loop_start_time = time.perf_counter()

last_timestamp_ms = 0

frame_counter = 0


# ============================================================
# CACHE VARIABLES
# ============================================================

last_face_landmarks = None
last_face_detected = False

last_pose_landmarks = None
last_pose_detected = False


# ============================================================
# ATTENTION VARIABLES
# ============================================================

attention_score = 0.0

attention_status = "CALIBRATING"

head_direction = "CENTER"

eye_direction = "CENTER"

head_x = 0.0

head_y = 0.0

face_scale = 0.0

gaze_h = 0.5

gaze_v = 0.5


# ============================================================
# POSTURE VARIABLES
# ============================================================

posture_result = {
    "posture": "NO BODY",
    "neck_angle": 0.0,
    "shoulder_angle": 0.0,
    "torso_angle": 0.0,
    "stability": 0.0
}


# ============================================================
# ACTIVITY VARIABLES
# ============================================================

activity_result = {
    "activity": "WAITING",
    "confidence": 0.0,
    "frames": 0
}


# ============================================================
# ERGONOMIC VARIABLES
# ============================================================

ergonomic_result = {
    "risk": "GOOD",
    "bad_duration": 0.0,
    "reasons": [],
    "recommendation": "Posture looks good"
}


# ============================================================
# BLINK VARIABLES
# ============================================================

blink_result = {
    "blink_count": 0,
    "blink_rate": 0.0,
    "eye_state": "OPEN",
    "eye_ratio": 0.0,
    "closed_frames": 0
}


# ============================================================
# DROWSINESS VARIABLES
# ============================================================

drowsiness_result = {
    "status": "ALERT",
    "drowsiness_score": 0.0,
    "closed_duration": 0.0
}


# ============================================================
# OTHER VARIABLES
# ============================================================

fps_result = {
    "fps": 0.0
}

detection_result = {
    "face_detected": False,
    "body_detected": False,
    "face_status": "NOT DETECTED",
    "body_status": "NOT DETECTED"
}

calibration_result = {
    "progress": 0.0,
    "current_frame": 0,
    "total_frames": 60,
    "completed": False,
    "status": "CALIBRATING"
}

distraction_result = {
    "is_distracted": False,
    "distraction_duration": 0.0
}

alert_result = {
    "alert": "No active alerts",
    "severity": "INFO",
    "alert_count": 0,
    "alert_duration": 0.0
}

recommendation_result = {
    "recommendation": "System is monitoring your activity."
}


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    image,
    text,
    position,
    scale=0.6,
    thickness=2
):

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# EYE GAZE
# ============================================================

def get_eye_gaze_values(landmarks):

    try:

        # ----------------------------------------------------
        # LEFT EYE
        # ----------------------------------------------------

        left_outer = landmarks[33]
        left_inner = landmarks[133]

        left_top = landmarks[159]
        left_bottom = landmarks[145]

        left_iris_points = landmarks[468:473]

        left_iris_x = sum(
            point.x
            for point in left_iris_points
        ) / len(left_iris_points)

        left_iris_y = sum(
            point.y
            for point in left_iris_points
        ) / len(left_iris_points)

        left_eye_width = (
            left_inner.x -
            left_outer.x
        )

        left_eye_height = (
            left_bottom.y -
            left_top.y
        )

        if abs(left_eye_width) > 0.0001:

            left_horizontal = (
                left_iris_x -
                left_outer.x
            ) / left_eye_width

        else:

            left_horizontal = 0.5

        if abs(left_eye_height) > 0.0001:

            left_vertical = (
                left_iris_y -
                left_top.y
            ) / left_eye_height

        else:

            left_vertical = 0.5


        # ----------------------------------------------------
        # RIGHT EYE
        # ----------------------------------------------------

        right_inner = landmarks[362]
        right_outer = landmarks[263]

        right_top = landmarks[386]
        right_bottom = landmarks[374]

        right_iris_points = landmarks[473:478]

        right_iris_x = sum(
            point.x
            for point in right_iris_points
        ) / len(right_iris_points)

        right_iris_y = sum(
            point.y
            for point in right_iris_points
        ) / len(right_iris_points)

        right_eye_width = (
            right_outer.x -
            right_inner.x
        )

        right_eye_height = (
            right_bottom.y -
            right_top.y
        )

        if abs(right_eye_width) > 0.0001:

            right_horizontal = (
                right_iris_x -
                right_inner.x
            ) / right_eye_width

        else:

            right_horizontal = 0.5

        if abs(right_eye_height) > 0.0001:

            right_vertical = (
                right_iris_y -
                right_top.y
            ) / right_eye_height

        else:

            right_vertical = 0.5


        # ----------------------------------------------------
        # AVERAGE
        # ----------------------------------------------------

        horizontal = (
            left_horizontal +
            right_horizontal
        ) / 2.0

        vertical = (
            left_vertical +
            right_vertical
        ) / 2.0


        horizontal = max(
            0.0,
            min(1.0, horizontal)
        )

        vertical = max(
            0.0,
            min(1.0, vertical)
        )

        return horizontal, vertical


    except Exception as error:

        print(
            "Eye gaze calculation error:",
            error
        )

        return 0.5, 0.5


# ============================================================
# EYE DIRECTION
# ============================================================

def get_eye_direction(
    horizontal,
    vertical
):

    horizontal_difference = (
        horizontal - 0.5
    )

    vertical_difference = (
        vertical - 0.5
    )

    horizontal_strength = abs(
        horizontal_difference
    )

    vertical_strength = abs(
        vertical_difference
    )

    threshold = 0.08


    if (
        horizontal_strength < threshold
        and
        vertical_strength < threshold
    ):

        return "CENTER"


    if (
        horizontal_strength >=
        vertical_strength
    ):

        if horizontal_difference < -threshold:

            return "LEFT"

        elif horizontal_difference > threshold:

            return "RIGHT"


    else:

        if vertical_difference < -threshold:

            return "UP"

        elif vertical_difference > threshold:

            return "DOWN"


    return "CENTER"


# ============================================================
# DRAW FACE LANDMARKS
# ============================================================

def draw_face_landmarks(
    frame,
    landmarks
):

    height, width = frame.shape[:2]

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
                1,
                (0, 255, 255),
                -1
            )


# ============================================================
# DRAW IRIS
# ============================================================

def draw_iris(
    frame,
    landmarks
):

    height, width = frame.shape[:2]

    for index in range(468, 478):

        landmark = landmarks[index]

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
                (0, 0, 255),
                -1
            )


# ============================================================
# DRAW POSE
# ============================================================

def draw_pose_landmarks(
    frame,
    landmarks
):

    height, width = frame.shape[:2]

    connections = [

        (11, 12),

        (11, 13),
        (13, 15),

        (12, 14),
        (14, 16),

        (11, 23),
        (12, 24),

        (23, 24),

        (23, 25),
        (25, 27),

        (24, 26),
        (26, 28)
    ]


    for start, end in connections:

        p1 = landmarks[start]
        p2 = landmarks[end]

        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )


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


# ============================================================
# DASHBOARD
# ============================================================

def create_dashboard(
    camera_frame,
    elapsed_time,
    attention_score,
    attention_status,
    head_direction,
    eye_direction,
    head_x,
    head_y,
    face_scale,
    gaze_h,
    gaze_v,
    posture_result,
    activity_result,
    ergonomic_result,
    blink_result,
    drowsiness_result,
    fps_result,
    detection_result,
    calibration_result,
    distraction_result,
    alert_result,
    recommendation_result
):

    camera_frame = cv2.resize(
        camera_frame,
        (
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    panel = np.full(
        (
            CAMERA_HEIGHT,
            PANEL_WIDTH,
            3
        ),
        35,
        dtype=np.uint8
    )


    # ========================================================
    # TITLE
    # ========================================================

    draw_text(
        panel,
        "FOCUSLENS AI",
        (20, 32),
        0.62,
        2
    )

    draw_text(
        panel,
        f"Session: {elapsed_time}",
        (20, 58),
        0.40,
        1
    )

    draw_text(
        panel,
        f"FPS: {fps_result['fps']:.1f}",
        (220, 58),
        0.40,
        1
    )


    # ========================================================
    # SYSTEM
    # ========================================================

    draw_text(
        panel,
        "SYSTEM STATUS",
        (20, 88),
        0.48,
        2
    )

    draw_text(
        panel,
        f"Face: {detection_result['face_status']}",
        (20, 112),
        0.36,
        1
    )

    draw_text(
        panel,
        f"Body: {detection_result['body_status']}",
        (205, 112),
        0.36,
        1
    )


    # ========================================================
    # CALIBRATION
    # ========================================================

    draw_text(
        panel,
        "CALIBRATION",
        (20, 140),
        0.48,
        2
    )

    draw_text(
        panel,
        (
            f"{calibration_result['status']} - "
            f"{calibration_result['progress']:.0f}%"
        ),
        (20, 164),
        0.36,
        1
    )


    # ========================================================
    # ATTENTION
    # ========================================================

    draw_text(
        panel,
        "ATTENTION",
        (20, 194),
        0.48,
        2
    )

    draw_text(
        panel,
        f"Score: {attention_score:.1f}%",
        (20, 218),
        0.38,
        1
    )

    draw_text(
        panel,
        f"Status: {attention_status}",
        (20, 242),
        0.36,
        1
    )

    draw_text(
        panel,
        f"Head: {head_direction}",
        (200, 218),
        0.36,
        1
    )

    draw_text(
        panel,
        f"Eyes: {eye_direction}",
        (200, 242),
        0.36,
        1
    )


    # ========================================================
    # HEAD / EYES
    # ========================================================

    draw_text(
        panel,
        "HEAD / EYES",
        (20, 272),
        0.48,
        2
    )

    draw_text(
        panel,
        f"Head X: {head_x:.2f}",
        (20, 296),
        0.34,
        1
    )

    draw_text(
        panel,
        f"Head Y: {head_y:.2f}",
        (20, 318),
        0.34,
        1
    )

    draw_text(
        panel,
        f"Face: {face_scale:.3f}",
        (20, 340),
        0.34,
        1
    )

    draw_text(
        panel,
        f"Gaze H: {gaze_h:.2f}",
        (200, 296),
        0.34,
        1
    )

    draw_text(
        panel,
        f"Gaze V: {gaze_v:.2f}",
        (200, 318),
        0.34,
        1
    )


    # ========================================================
    # ALERTNESS
    # ========================================================

    draw_text(
        panel,
        "ALERTNESS",
        (20, 370),
        0.48,
        2
    )

    draw_text(
        panel,
        f"Blinks: {blink_result['blink_count']}",
        (20, 394),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Blink rate: "
            f"{blink_result['blink_rate']:.1f}/min"
        ),
        (160, 394),
        0.34,
        1
    )

    draw_text(
        panel,
        f"Eyes: {blink_result['eye_state']}",
        (20, 416),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Drowsiness: "
            f"{drowsiness_result['status']}"
        ),
        (160, 416),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Score: "
            f"{drowsiness_result['drowsiness_score']:.0f}%"
        ),
        (20, 438),
        0.34,
        1
    )


    # ========================================================
    # POSTURE
    # ========================================================

    draw_text(
        panel,
        "POSTURE",
        (20, 468),
        0.48,
        2
    )

    draw_text(
        panel,
        posture_result["posture"],
        (20, 492),
        0.36,
        1
    )

    draw_text(
        panel,
        (
            f"Neck: "
            f"{posture_result['neck_angle']:.1f}"
        ),
        (20, 514),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Shoulder: "
            f"{posture_result['shoulder_angle']:.1f}"
        ),
        (150, 514),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Torso: "
            f"{posture_result['torso_angle']:.1f}"
        ),
        (20, 536),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Ergonomic: "
            f"{ergonomic_result['risk']}"
        ),
        (150, 536),
        0.34,
        1
    )


    # ========================================================
    # ACTIVITY
    # ========================================================

    draw_text(
        panel,
        "ACTIVITY",
        (20, 566),
        0.48,
        2
    )

    draw_text(
        panel,
        activity_result["activity"],
        (20, 590),
        0.36,
        1
    )

    draw_text(
        panel,
        (
            f"Confidence: "
            f"{activity_result['confidence']:.1f}%"
        ),
        (150, 590),
        0.34,
        1
    )

    draw_text(
        panel,
        (
            f"Distraction: "
            f"{distraction_result['distraction_duration']:.1f}s"
        ),
        (20, 612),
        0.34,
        1
    )


    # ========================================================
    # ALERTS
    # ========================================================

    draw_text(
        panel,
        f"Alerts: {alert_result['alert_count']}",
        (20, 640),
        0.34,
        1
    )

    draw_text(
        panel,
        f"Severity: {alert_result['severity']}",
        (155, 640),
        0.34,
        1
    )

    alert_text = alert_result["alert"]

    if len(alert_text) > 48:

        alert_text = (
            alert_text[:45] +
            "..."
        )

    draw_text(
        panel,
        f"Alert: {alert_text}",
        (20, 662),
        0.29,
        1
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    draw_text(
        panel,
        "Recommendation:",
        (20, 686),
        0.30,
        1
    )

    recommendation = (
        recommendation_result[
            "recommendation"
        ]
    )

    if len(recommendation) > 42:

        recommendation = (
            recommendation[:39] +
            "..."
        )

    draw_text(
        panel,
        recommendation,
        (20, 708),
        0.28,
        1
    )


    return cv2.hconcat(
        [
            camera_frame,
            panel
        ]
    )


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("======================================")
print("          FOCUSLENS AI")
print("======================================")
print()
print("Camera started.")
print("Look straight at the camera during calibration.")
print("Press Q to quit.")
print()


try:

    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # ====================================================
        # FPS
        # ====================================================

        fps_result["fps"] = (
            fps_counter.update()
        )


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        frame_height, frame_width = (
            frame.shape[:2]
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(
            (
                time.perf_counter()
                -
                loop_start_time
            ) * 1000
        )

        if timestamp_ms <= last_timestamp_ms:

            timestamp_ms = (
                last_timestamp_ms +
                1
            )

        last_timestamp_ms = timestamp_ms


        # ====================================================
        # SMALL PROCESSING FRAME
        # ====================================================

        processing_frame = cv2.resize(
            frame,
            (
                PROCESS_WIDTH,
                PROCESS_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        rgb_frame = cv2.cvtColor(
            processing_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # FACE DETECTION
        # ====================================================

        face_updated = False

        if (
            frame_counter %
            FACE_PROCESS_INTERVAL == 0
            or
            last_face_landmarks is None
        ):

            face_updated = True

            try:

                face_result = (
                    face_landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                if face_result.face_landmarks:

                    last_face_landmarks = (
                        face_result.face_landmarks[0]
                    )

                    last_face_detected = True

                else:

                    last_face_landmarks = None

                    last_face_detected = False


            except Exception as error:

                print(
                    "Face detection error:",
                    error
                )

                last_face_landmarks = None

                last_face_detected = False


        face_landmarks = (
            last_face_landmarks
        )


        # ====================================================
        # FACE PROCESSING
        # ====================================================

        if face_landmarks is not None:

            detection_result = (
                detection_status.update(
                    True,
                    last_pose_detected
                )
            )


            # ------------------------------------------------
            # IRIS DRAWING
            # ------------------------------------------------

            draw_iris(
                frame,
                face_landmarks
            )


            # ------------------------------------------------
            # OPTIONAL FACE MESH
            # ------------------------------------------------

            if DRAW_FULL_FACE_MESH:

                draw_face_landmarks(
                    frame,
                    face_landmarks
                )


            # ------------------------------------------------
            # FACE ANALYSIS ONLY ON NEW FACE RESULT
            # ------------------------------------------------

            if face_updated:

                try:

                    # ========================================
                    # GAZE
                    # ========================================

                    gaze_h, gaze_v = (
                        get_eye_gaze_values(
                            face_landmarks
                        )
                    )


                    # ========================================
                    # EYE DIRECTION
                    # ========================================

                    eye_direction = (
                        get_eye_direction(
                            gaze_h,
                            gaze_v
                        )
                    )


                    # ========================================
                    # CALIBRATION
                    # ========================================

                    attention_analyzer.update_calibration(
                        face_landmarks,
                        gaze_h,
                        gaze_v
                    )


                    # ========================================
                    # HEAD POSE
                    # ========================================

                    head_result = (
                        attention_analyzer.calculate_head_pose(
                            face_landmarks,
                            PROCESS_WIDTH,
                            PROCESS_HEIGHT
                        )
                    )

                    (
                        head_x,
                        head_y,
                        face_scale,
                        head_direction
                    ) = head_result


                    # ========================================
                    # ATTENTION
                    # ========================================

                    attention_score = (
                        attention_analyzer.calculate_attention_score(
                            head_direction,
                            eye_direction
                        )
                    )

                    attention_status = (
                        attention_analyzer.get_status(
                            attention_score
                        )
                    )


                    # ========================================
                    # BLINK
                    # ========================================

                    blink_result = (
                        blink_detector.update(
                            face_landmarks
                        )
                    )


                    # ========================================
                    # DROWSINESS
                    # ========================================

                    drowsiness_result = (
                        drowsiness_detector.update(
                            blink_result["eye_state"],
                            attention_status
                        )
                    )


                    # ========================================
                    # CALIBRATION STATUS
                    # ========================================

                    try:

                        calibration_progress = (
                            attention_analyzer
                            .get_calibration_progress()
                        )

                        if isinstance(
                            calibration_progress,
                            dict
                        ):

                            current_frame = (
                                calibration_progress.get(
                                    "current_frame",
                                    60
                                    if calibration_progress.get(
                                        "completed",
                                        False
                                    )
                                    else 0
                                )
                            )

                        else:

                            calibration_value = float(
                                calibration_progress
                            )

                            current_frame = int(
                                calibration_value
                                if calibration_value > 1
                                else calibration_value * 60
                            )

                        calibration_result = (
                            calibration_status.update(
                                current_frame
                            )
                        )

                    except Exception:

                        calibration_result = (
                            calibration_status.update(
                                60
                                if getattr(
                                    attention_analyzer,
                                    "calibrated",
                                    False
                                )
                                else 0
                            )
                        )


                    # ========================================
                    # DISTRACTION
                    # ========================================

                    distraction_result = (
                        distraction_tracker.update(
                            attention_status,
                            head_direction,
                            eye_direction
                        )
                    )


                except Exception as error:

                    print(
                        "Face analysis error:",
                        error
                    )

                    traceback.print_exc()


        else:

            # =================================================
            # NO FACE
            # =================================================

            attention_score = 0.0

            attention_status = "NO FACE"

            head_direction = "NO FACE"

            eye_direction = "NO FACE"

            head_x = 0.0

            head_y = 0.0

            face_scale = 0.0

            gaze_h = 0.5

            gaze_v = 0.5


            detection_result = (
                detection_status.update(
                    False,
                    last_pose_detected
                )
            )


            blink_result = {
                "blink_count":
                    blink_detector.blink_count,

                "blink_rate":
                    blink_detector.get_blink_rate(),

                "eye_state":
                    "NO FACE",

                "eye_ratio":
                    0.0,

                "closed_frames":
                    0
            }


            drowsiness_result = {
                "status": "NO FACE",
                "drowsiness_score": 0.0,
                "closed_duration": 0.0
            }


            distraction_result = (
                distraction_tracker.update(
                    "NO FACE",
                    "NO FACE",
                    "NO FACE"
                )
            )


            calibration_result = (
                calibration_status.update(
                    60
                    if getattr(
                        attention_analyzer,
                        "calibrated",
                        False
                    )
                    else 0
                )
            )


        # ====================================================
        # POSE DETECTION
        # ====================================================

        pose_updated = False

        if (
            frame_counter %
            POSE_PROCESS_INTERVAL == 0
            or
            last_pose_landmarks is None
        ):

            pose_updated = True

            try:

                pose_result = (
                    pose_landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                if pose_result.pose_landmarks:

                    last_pose_landmarks = (
                        pose_result.pose_landmarks[0]
                    )

                    last_pose_detected = True

                else:

                    last_pose_landmarks = None

                    last_pose_detected = False


            except Exception as error:

                print(
                    "Pose detection error:",
                    error
                )

                last_pose_landmarks = None

                last_pose_detected = False


        pose_landmarks = (
            last_pose_landmarks
        )


        # ====================================================
        # POSE PROCESSING
        # ====================================================

        if pose_landmarks is not None:

            detection_result = (
                detection_status.update(
                    last_face_detected,
                    True
                )
            )


            # ------------------------------------------------
            # DRAW POSE
            # ------------------------------------------------

            if DRAW_POSE:

                draw_pose_landmarks(
                    frame,
                    pose_landmarks
                )


            # ------------------------------------------------
            # POSTURE / ACTIVITY / ERGONOMICS
            # ONLY ON FRESH POSE
            # ------------------------------------------------

            if pose_updated:

                # ============================================
                # POSTURE
                # ============================================

                try:

                    posture_result = (
                        posture_analyzer.analyze(
                            pose_landmarks
                        )
                    )

                except Exception as error:

                    print(
                        "Posture error:",
                        error
                    )

                    posture_result = {
                        "posture": "ERROR",
                        "neck_angle": 0.0,
                        "shoulder_angle": 0.0,
                        "torso_angle": 0.0,
                        "stability": 0.0
                    }


                # ============================================
                # ACTIVITY
                # ============================================

                try:

                    activity_result = (
                        activity_engine.update(
                            pose_landmarks
                        )
                    )

                except Exception as error:

                    print(
                        "Activity error:",
                        error
                    )

                    activity_result = {
                        "activity": "ERROR",
                        "confidence": 0.0,
                        "frames": 0
                    }


                # ============================================
                # ERGONOMICS
                # ============================================

                try:

                    ergonomic_result = (
                        ergonomic_analyzer.analyze(
                            posture_result[
                                "neck_angle"
                            ],

                            posture_result[
                                "shoulder_angle"
                            ],

                            posture_result[
                                "torso_angle"
                            ],

                            posture_result[
                                "posture"
                            ]
                        )
                    )

                except Exception as error:

                    print(
                        "Ergonomic error:",
                        error
                    )

                    ergonomic_result = {
                        "risk": "ERROR",
                        "bad_duration": 0.0,
                        "reasons": [],
                        "recommendation": ""
                    }


        else:

            # =================================================
            # NO BODY
            # =================================================

            posture_result = {
                "posture": "NO BODY",
                "neck_angle": 0.0,
                "shoulder_angle": 0.0,
                "torso_angle": 0.0,
                "stability": 0.0
            }


            # Clear temporal activity sequence
            # when the body disappears.

            activity_result = (
                activity_engine.update(
                    None
                )
            )


            ergonomic_result = {
                "risk": "NO BODY",
                "bad_duration": 0.0,
                "reasons": [],
                "recommendation": ""
            }


            detection_result = (
                detection_status.update(
                    last_face_detected,
                    False
                )
            )


            last_pose_detected = False


        # ====================================================
        # SMART ALERTS
        # ====================================================

        try:

            alert_result = (
                smart_alerts.generate_alert(
                    attention_status,
                    head_direction,
                    eye_direction,
                    posture_result["posture"],
                    drowsiness_result["status"],
                    ergonomic_result["risk"],
                    detection_result["face_detected"],
                    detection_result["body_detected"]
                )
            )

        except Exception as error:

            print(
                "Smart alert error:",
                error
            )


        # ====================================================
        # LIVE RECOMMENDATION
        # ====================================================

        try:

            recommendation_result = (
                live_recommendation.generate(
                    attention_status,
                    posture_result["posture"],
                    activity_result["activity"],
                    ergonomic_result["risk"],
                    drowsiness_result["status"],
                    detection_result["face_detected"],
                    detection_result["body_detected"],
                    head_direction,
                    eye_direction
                )
            )

        except Exception as error:

            print(
                "Recommendation error:",
                error
            )


        # ====================================================
        # SESSION TRACKER
        # ====================================================

        try:

            session_tracker.update(
                attention_score,
                attention_status,
                posture_result["posture"],
                activity_result["activity"],
                ergonomic_result["risk"]
            )

        except Exception:

            pass


        # ====================================================
        # SESSION TIME
        # ====================================================

        elapsed_seconds = int(
            time.time() -
            start_time
        )

        minutes = (
            elapsed_seconds // 60
        )

        seconds = (
            elapsed_seconds % 60
        )

        elapsed_time = (
            f"{minutes:02d}:{seconds:02d}"
        )


        # ====================================================
        # CAMERA TITLE
        # ====================================================

        cv2.putText(
            frame,
            "FocusLens AI",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        dashboard = create_dashboard(
            frame,
            elapsed_time,
            attention_score,
            attention_status,
            head_direction,
            eye_direction,
            head_x,
            head_y,
            face_scale,
            gaze_h,
            gaze_v,
            posture_result,
            activity_result,
            ergonomic_result,
            blink_result,
            drowsiness_result,
            fps_result,
            detection_result,
            calibration_result,
            distraction_result,
            alert_result,
            recommendation_result
        )


        # ====================================================
        # FINAL DISPLAY
        # ====================================================

        dashboard = cv2.resize(
            dashboard,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        cv2.imshow(
            WINDOW_NAME,
            dashboard
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


        # ====================================================
        # FRAME COUNTER
        # ====================================================

        frame_counter += 1


# ============================================================
# KEYBOARD INTERRUPT
# ============================================================

except KeyboardInterrupt:

    print()
    print("Program stopped.")


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()

    face_landmarker.close()

    pose_landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("FocusLens AI closed.")