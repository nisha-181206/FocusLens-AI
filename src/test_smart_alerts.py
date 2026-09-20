import time

from smart_alerts import SmartAlertSystem


def print_result(title, result):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("Alert:", result["alert"])
    print("Severity:", result["severity"])
    print("Alert Count:", result["alert_count"])
    print("Alert Duration:", round(result["alert_duration"], 2))


# ============================================================
# CREATE ALERT SYSTEM
# ============================================================

alerts = SmartAlertSystem(
    distraction_limit=2.0,
    posture_limit=5.0,
    critical_distraction_limit=5.0,
    cooldown=5.0
)


# ============================================================
# TEST 1 — NORMAL CONDITION
# ============================================================

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 1 — NORMAL CONDITION",
    result
)


# ============================================================
# TEST 2 — SHORT DISTRACTION
# ============================================================

print("\nStarting short distraction test...")

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="LEFT",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 2 — DISTRACTION START",
    result
)

time.sleep(1)

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="LEFT",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 2 — AFTER 1 SECOND",
    result
)


# ============================================================
# TEST 3 — DISTRACTION WARNING
# ============================================================

print("\nWaiting for distraction threshold...")

time.sleep(1.2)

result = alerts.generate_alert(
    attention_status="DISTRACTED",
    head_direction="LEFT",
    eye_direction="LEFT",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 3 — DISTRACTION WARNING",
    result
)


# ============================================================
# TEST 4 — RETURN TO CENTER
# ============================================================

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 4 — RETURNED TO CENTER",
    result
)


# ============================================================
# TEST 5 — BAD POSTURE
# ============================================================

print("\nStarting bad posture test...")

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="SLOUCHING",
    drowsiness_status="ALERT",
    ergonomic_risk="LOW RISK",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 5 — BAD POSTURE START",
    result
)

time.sleep(5.2)

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="SLOUCHING",
    drowsiness_status="ALERT",
    ergonomic_risk="LOW RISK",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 5 — POSTURE WARNING",
    result
)


# ============================================================
# TEST 6 — DROWSINESS
# ============================================================

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="DROWSY",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=True
)

print_result(
    "TEST 6 — DROWSINESS",
    result
)


# ============================================================
# TEST 7 — FACE NOT DETECTED
# ============================================================

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=False,
    body_detected=True
)

print_result(
    "TEST 7 — FACE NOT DETECTED",
    result
)


# ============================================================
# TEST 8 — BODY NOT DETECTED
# ============================================================

result = alerts.generate_alert(
    attention_status="GOOD",
    head_direction="CENTER",
    eye_direction="CENTER",
    posture="GOOD POSTURE",
    drowsiness_status="ALERT",
    ergonomic_risk="GOOD",
    face_detected=True,
    body_detected=False
)

print_result(
    "TEST 8 — BODY NOT DETECTED",
    result
)


print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)