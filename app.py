import os

"""
╔══════════════════════════════════════════════════════════╗
║           DROWSINESS DETECTION SYSTEM                   ║
║     Powered by MediaPipe Face Landmarker + OpenCV       ║
╚══════════════════════════════════════════════════════════╝

Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
to detect drowsiness and yawning in real-time via webcam.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import threading
import math

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

EAR_THRESHOLD   = 0.22   # Below this → eyes considered closed
MAR_THRESHOLD   = 0.38   # Above this → yawning detected (lower = more sensitive)
EAR_CONSEC_FRAMES = 20   # Frames eye must stay closed to trigger alert
MAR_CONSEC_FRAMES = 25   # Frames mouth must stay open to trigger yawn alert (higher = ignore quick talking)

# Alert cooldown (seconds) – prevents alert from firing repeatedly
ALERT_COOLDOWN  = 3.0

# ─────────────────────────────────────────────
#  MediaPipe Landmark Indices (478-point model)
# ─────────────────────────────────────────────
# Left eye landmarks  (horizontal & vertical pairs)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33,  160, 158,  133, 153, 144]
# Mouth landmarks – inner lip points for precise MAR
# Corners: 61(L), 291(R)  |  Left vertical: 82(up), 87(dn)
# Centre vertical: 13(up), 14(dn)  |  Right vertical: 312(up), 317(dn)
MOUTH     = [61, 291, 82, 87, 13, 14, 312, 317]

# Iris centres (for gaze visualisation)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
COLOR_GREEN  = (57,  255, 20)
COLOR_YELLOW = (0,   220, 255)
COLOR_RED    = (0,   60,  255)
COLOR_BLUE   = (255, 165, 0)
COLOR_WHITE  = (240, 240, 240)
COLOR_DARK   = (15,  15,  25)
COLOR_CYAN   = (255, 220, 50)
COLOR_ORANGE = (0,   140, 255)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio – Soukupová & Čech (2016)."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    # vertical 1
    A = euclidean(pts[1], pts[5])
    # vertical 2
    B = euclidean(pts[2], pts[4])
    # horizontal
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def compute_mar(landmarks, mouth_indices, w, h):
    """Mouth Aspect Ratio – 3-vertical-pair formula for accuracy.

    mouth_indices layout:
      [0]=61(L corner), [1]=291(R corner),
      [2]=82(L up),     [3]=87(L dn),
      [4]=13(C up),     [5]=14(C dn),
      [6]=312(R up),    [7]=317(R dn)
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    V_left   = euclidean(pts[2], pts[3])   # left inner lip gap
    V_center = euclidean(pts[4], pts[5])   # center inner lip gap
    V_right  = euclidean(pts[6], pts[7])   # right inner lip gap
    H        = euclidean(pts[0], pts[1])   # mouth width
    return (V_left + V_center + V_right) / (3.0 * H)


def get_landmark_points(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]


# Pre-load connection modules removed – drawing handled natively with OpenCV Delaunay triangulation


def draw_face_mesh(frame, lms, w, h, status):
    """
    Render full face mesh natively matching the reference image style:
      - Semi-transparent tessellation triangles (via OpenCV Delaunay)
      - Colored contours: eyes, lips
      - Iris dots
      - Red landmark dots
    """
    # ── 1. Create Delaunay triangulation for the face mesh spiderweb ──
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    pts_xy = []
    for lm in lms:
        # Clamp points strictly within the image boundaries to prevent cv::Subdiv2D crashes
        px = max(0, min(int(lm.x * w), w - 1))
        py = max(0, min(int(lm.y * h), h - 1))
        subdiv.insert((px, py))
        pts_xy.append((px, py))
        
    overlay = frame.copy()
    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if (0 <= pt1[0] <= w) and (0 <= pt1[1] <= h) and \
           (0 <= pt2[0] <= w) and (0 <= pt2[1] <= h) and \
           (0 <= pt3[0] <= w) and (0 <= pt3[1] <= h):
            # Only draw if the triangle is reasonably small (prevents edge artifacts)
            if euclidean(pt1, pt2) < w * 0.15:
                cv2.line(overlay, pt1, pt2, (100, 100, 100), 1, cv2.LINE_AA)
                cv2.line(overlay, pt2, pt3, (100, 100, 100), 1, cv2.LINE_AA)
                cv2.line(overlay, pt3, pt1, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # ── 2. Draw Eyes Contour ───────────────────────────────────────
    eye_col = COLOR_RED if status == "DROWSY" else (COLOR_YELLOW if status == "WARNING" else COLOR_GREEN)
    cv2.polylines(frame, [np.array([pts_xy[i] for i in LEFT_EYE])], True, (255, 100, 50), 2, cv2.LINE_AA)
    cv2.polylines(frame, [np.array([pts_xy[i] for i in RIGHT_EYE])], True, (50, 200, 50), 2, cv2.LINE_AA)

    # ── 3. Draw Lips Contour ───────────────────────────────────────
    mouth_col = COLOR_ORANGE if status == "YAWNING" else (80, 80, 240)
    cv2.polylines(frame, [np.array([pts_xy[i] for i in MOUTH])], True, mouth_col, 2, cv2.LINE_AA)

    # ── 4. Iris dots ───────────────────────────────────────────────
    for idx_set in [LEFT_IRIS, RIGHT_IRIS]:
        try:
            cx = int(np.mean([pts_xy[i][0] for i in idx_set]))
            cy = int(np.mean([pts_xy[i][1] for i in idx_set]))
            cv2.circle(frame, (cx, cy), 6, (230, 130, 50), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (230, 130, 50), -1, cv2.LINE_AA)
        except Exception:
            pass

    # ── 5. Red landmark dots ───────────────────────────────────────
    for pt in pts_xy:
        cv2.circle(frame, pt, 1, (60, 60, 220), -1, cv2.LINE_AA)

    # ── 6. EAR / MAR text ──────────────────────────────────────────
    nose_x, nose_y = pts_xy[1]
    ear_val = (compute_ear(lms, LEFT_EYE, w, h) + compute_ear(lms, RIGHT_EYE, w, h)) / 2.0
    mar_val = compute_mar(lms, MOUTH, w, h)
    put_text(frame, f"EAR:{ear_val:.2f}", (nose_x - 40, nose_y - 20), 0.38, eye_col)
    put_text(frame, f"MAR:{mar_val:.2f}", (nose_x - 40, nose_y - 4), 0.38, mouth_col)


def draw_eye_contour(frame, pts, color):
    hull = cv2.convexHull(np.array(pts))
    cv2.drawContours(frame, [hull], -1, color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_rounded_rect(frame, x, y, w, h, r, color, thickness=-1, alpha=1.0):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, thickness)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, thickness)
    cv2.ellipse(overlay, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        frame[:] = overlay


def draw_progress_bar(frame, x, y, w, h, value, max_val, color, bg=(40, 40, 50)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
    fill = int(w * min(value / max_val, 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 90), 1, cv2.LINE_AA)


def put_text(frame, text, pos, scale=0.5, color=COLOR_WHITE, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  ALERT MANAGER  (thread-safe beep)
# ─────────────────────────────────────────────
class AlertManager:
    def __init__(self):
        self._last_alert = 0
        self._lock       = threading.Lock()
        self.active      = False

    def trigger(self, reason="DROWSY"):
        now = time.time()
        with self._lock:
            if now - self._last_alert >= ALERT_COOLDOWN:
                self._last_alert = now
                self.active = True
                threading.Thread(target=self._beep, args=(reason,), daemon=True).start()

    def _beep(self, reason):
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1800, 250)
                time.sleep(0.1)
        except Exception:
            print(f"\a\a\a")  # fallback terminal bell
        time.sleep(1.5)
        self.active = False


# ─────────────────────────────────────────────
#  HUD OVERLAY
# ─────────────────────────────────────────────
class HUD:
    def __init__(self, frame_w, frame_h):
        self.fw = frame_w
        self.fh = frame_h
        self._panel_w = 260
        self._panel_h = frame_h
        self._anim_t  = 0.0

    def draw(self, frame, state: dict):
        self._anim_t += 0.05
        pw = self._panel_w

        # ── Left sidebar ──────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (pw, self.fh), (12, 12, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Title
        put_text(frame, "DROWSINESS", (12, 32), 0.65, COLOR_CYAN, 2)
        put_text(frame, "MONITOR",    (12, 54), 0.65, COLOR_CYAN, 2)

        # Divider
        cv2.line(frame, (12, 64), (pw - 12, 64), (60, 60, 80), 1)

        # Status pill
        status     = state["status"]
        status_col = COLOR_GREEN if status == "ALERT" else (COLOR_YELLOW if status == "WARNING" else COLOR_RED)
        cv2.rectangle(frame, (12, 75), (pw - 12, 105), status_col, -1)
        put_text(frame, f"  {status}", (20, 97), 0.65, COLOR_DARK, 2)

        # ── Metrics ──────────────────────────────────
        y = 125
        for label, val, threshold, mode in [
            ("EAR  (Eye)",    state["ear"],  EAR_THRESHOLD, "low"),
            ("MAR  (Mouth)",  state["mar"],  MAR_THRESHOLD, "high"),
        ]:
            put_text(frame, label, (12, y), 0.45, COLOR_WHITE, 1)
            y += 18
            bar_color = (COLOR_GREEN if (val > threshold if mode == "low" else val < threshold)
                         else COLOR_RED)
            draw_progress_bar(frame, 12, y, pw - 24, 10, val, 1.0, bar_color)
            put_text(frame, f"{val:.3f}", (pw - 52, y + 9), 0.4, bar_color, 1)
            # threshold line
            tx = 12 + int((pw - 24) * threshold)
            cv2.line(frame, (tx, y), (tx, y + 10), COLOR_YELLOW, 1)
            y += 28

        # ── Counters ─────────────────────────────────
        cv2.line(frame, (12, y), (pw - 12, y), (60, 60, 80), 1); y += 12
        put_text(frame, "BLINK COUNT",    (12, y), 0.42, (180, 180, 180)); y += 18
        put_text(frame, str(state["blinks"]), (12, y), 0.70, COLOR_WHITE, 2); y += 30

        put_text(frame, "YAWN COUNT",     (12, y), 0.42, (180, 180, 180)); y += 18
        put_text(frame, str(state["yawns"]),  (12, y), 0.70, COLOR_WHITE, 2); y += 30

        put_text(frame, "DROWSY EVENTS",  (12, y), 0.42, (180, 180, 180)); y += 18
        put_text(frame, str(state["drowsy_events"]), (12, y), 0.70, COLOR_RED, 2); y += 30

        # ── Session timer ─────────────────────────────
        elapsed = int(time.time() - state["start_time"])
        h_t, rem = divmod(elapsed, 3600)
        m_t, s_t = divmod(rem, 60)
        cv2.line(frame, (12, y), (pw - 12, y), (60, 60, 80), 1); y += 12
        put_text(frame, "SESSION TIME",   (12, y), 0.42, (180, 180, 180)); y += 18
        put_text(frame, f"{h_t:02d}:{m_t:02d}:{s_t:02d}", (12, y), 0.65, COLOR_CYAN, 2); y += 30

        # ── FPS ───────────────────────────────────────
        put_text(frame, f"FPS: {state['fps']:.1f}", (12, self.fh - 20), 0.45, (120, 120, 130))

        # ── ALERT FLASH ──────────────────────────────
        if state["alert_active"]:
            pulse = abs(math.sin(self._anim_t * 5))
            ov2   = frame.copy()
            cv2.rectangle(ov2, (0, 0), (self.fw, self.fh), (0, 0, int(200 * pulse)), -1)
            cv2.addWeighted(ov2, 0.35, frame, 0.65, 0, frame)

            msg = state.get("alert_msg", "DROWSINESS DETECTED!")
            ts  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            tx  = pw + (self.fw - pw - ts[0]) // 2
            ty  = self.fh // 2
            # shadow box
            cv2.rectangle(frame, (tx - 20, ty - 35), (tx + ts[0] + 20, ty + 15), COLOR_DARK, -1)
            put_text(frame, msg, (tx, ty), 1.0, COLOR_RED, 2)

        # ── Keybind legend ────────────────────────────
        legend = [("Q", "Quit"), ("R", "Reset stats"), ("S", "Save snapshot")]
        lx = pw + 8
        ly = self.fh - 8 - len(legend) * 18
        for key, desc in legend:
            put_text(frame, f"[{key}]", (lx, ly), 0.38, COLOR_CYAN)
            put_text(frame, desc, (lx + 28, ly), 0.38, (160, 160, 170))
            ly += 16


# ─────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  DROWSINESS DETECTION SYSTEM  –  Loading…")
    print("=" * 55)

    # ── Build MediaPipe FaceLandmarker ────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(opts)

    # ── Open webcam ───────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {fw}×{fh}")

    hud   = HUD(fw, fh)
    alert = AlertManager()

    # ── State ─────────────────────────────────────────
    ear_counter    = 0
    mar_counter    = 0
    blink_count    = 0
    yawn_count     = 0
    drowsy_events  = 0
    eye_was_closed = False
    mouth_was_open = False
    start_time     = time.time()
    prev_time      = time.time()
    fps            = 0.0
    snapshot_dir   = os.path.dirname(__file__)
    frame_ts_ms    = 0

    print("  Press Q to quit | R to reset | S to save snapshot")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [!] Camera read failed – retrying…")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)          # Mirror for natural UX
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # FPS
        now   = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Run inference
        frame_ts_ms += int(1000 / 30)
        result = detector.detect_for_video(mp_img, frame_ts_ms)

        ear_val    = 0.0
        mar_val    = 0.0
        status     = "ALERT"
        alert_msg  = ""

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            ear_l = compute_ear(lms, LEFT_EYE,  fw, fh)
            ear_r = compute_ear(lms, RIGHT_EYE, fw, fh)
            ear_val = (ear_l + ear_r) / 2.0
            mar_val = compute_mar(lms, MOUTH, fw, fh)

            # ── Eye state machine ─────────────────────
            if ear_val < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter == EAR_CONSEC_FRAMES:
                    # First frame crossing the threshold → count as drowsy event
                    drowsy_events += 1
                if ear_counter >= EAR_CONSEC_FRAMES:
                    # Beep continuously WHILE eyes stay closed (respects cooldown)
                    alert.trigger("EYES CLOSED")
                    alert_msg = "⚠  EYES CLOSED – WAKE UP!"
                    eye_was_closed = True
            else:
                if eye_was_closed:
                    blink_count += 1
                ear_counter    = 0
                eye_was_closed = False

            # ── Mouth / yawn state machine ────────────
            if mar_val > MAR_THRESHOLD:
                mar_counter += 1
                if mar_counter == MAR_CONSEC_FRAMES:
                    # First frame crossing yawn threshold → count as yawn
                    yawn_count += 1
                if mar_counter >= MAR_CONSEC_FRAMES:
                    # Beep continuously WHILE mouth is open/yawning
                    alert.trigger("YAWNING")
                    alert_msg = "⚠  YAWNING DETECTED!"
                    mouth_was_open = True
            else:
                mar_counter    = 0
                mouth_was_open = False

            # ── Status label ─────────────────────────
            if ear_counter >= EAR_CONSEC_FRAMES:
                status = "DROWSY"
            elif mar_counter >= MAR_CONSEC_FRAMES:
                status = "YAWNING"
            elif ear_val < EAR_THRESHOLD + 0.04:
                status = "WARNING"
            else:
                status = "ALERT"

            # ── Draw full face mesh ───────────────────
            draw_face_mesh(frame, lms, fw, fh, status)

        else:
            # No face detected
            status = "NO FACE"
            put_text(frame, "No face detected", (hud._panel_w + 20, 40), 0.7, COLOR_YELLOW, 2)

        # ── Draw HUD ──────────────────────────────────
        state = dict(
            ear=ear_val, mar=mar_val,
            status=status,
            blinks=blink_count, yawns=yawn_count,
            drowsy_events=drowsy_events,
            start_time=start_time,
            fps=fps,
            alert_active=alert.active,
            alert_msg=alert_msg,
        )
        hud.draw(frame, state)

        cv2.imshow("Drowsiness Detection  –  press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            blink_count = yawn_count = drowsy_events = 0
            ear_counter = mar_counter = 0
            start_time  = time.time()
            print("  [+] Stats reset.")
        elif key == ord('s'):
            fname = os.path.join(snapshot_dir, f"snapshot_{int(time.time())}.png")
            cv2.imwrite(fname, frame)
            print(f"  [+] Snapshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n  Session ended. Goodbye!")


if __name__ == "__main__":
    main()
