import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import threading
import math
import streamlit as st

# ─────────────────────────────────────────────
#  STREAMLIT PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Drowsiness Monitor", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    /* Premium Gradient Light Theme (SignSpeak AI styling) */
    .stApp {
        background: linear-gradient(135deg, #f7e6f3 0%, #e3eeff 100%);
        color: #1c1c28;
    }
    
    /* Top Header Adjustments */
    .css-10trblm {
        color: #1a1a2e;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek Side Panel Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.07);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .metric-card:hover {  transform: translateY(-2px); }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }
    
    /* Status Colors */
    .status-alert { color: #f43f5e; }
    .status-warning { color: #f59e0b; }
    .status-safe { color: #10b981; }
    
    /* Professional Action Button */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        color: white;
    }
    
    /* Video Frame Container Styling */
    .video-container {
        border-radius: 20px;
        overflow: hidden;
        border: 4px solid white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS & INDICES
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
ALERT_COOLDOWN = 3.0

LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158,  133, 153, 144]
MOUTH      = [61, 291, 82, 87, 13, 14, 312, 317]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Colors for mesh drawing natively
C_GREEN  = (57, 255, 20)
C_YELLOW = (0, 220, 255)
C_RED    = (0, 60, 255)
C_ORANGE = (0, 140, 255)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def compute_ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def compute_mar(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    V_left   = euclidean(pts[2], pts[3])
    V_center = euclidean(pts[4], pts[5])
    V_right  = euclidean(pts[6], pts[7])
    H        = euclidean(pts[0], pts[1])
    return (V_left + V_center + V_right) / (3.0 * H)

def draw_face_mesh(frame, lms, w, h, status):
    """Render Delaunay triangulation mesh and critical contours."""
    # 1. Delaunay Triangulation (Spiderweb)
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    pts_xy = []
    
    for lm in lms:
        px = max(0, min(int(lm.x * w), w - 1))
        py = max(0, min(int(lm.y * h), h - 1))
        subdiv.insert((px, py))
        pts_xy.append((px, py))
        
    overlay = frame.copy()
    try:
        triangleList = subdiv.getTriangleList()
        for t in triangleList:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            
            if all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in [pt1, pt2, pt3]):
                if euclidean(pt1, pt2) < w * 0.15:
                    cv2.line(overlay, pt1, pt2, (100, 100, 100), 1, cv2.LINE_AA)
                    cv2.line(overlay, pt2, pt3, (100, 100, 100), 1, cv2.LINE_AA)
                    cv2.line(overlay, pt3, pt1, (100, 100, 100), 1, cv2.LINE_AA)
    except Exception:
        pass
        
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # 2. Eye & Mouth Contours
    eye_col = C_RED if status == "DROWSY" else (C_YELLOW if status == "WARNING" else C_GREEN)
    cv2.polylines(frame, [np.array([pts_xy[i] for i in LEFT_EYE])], True, (255, 100, 50), 2, cv2.LINE_AA)
    cv2.polylines(frame, [np.array([pts_xy[i] for i in RIGHT_EYE])], True, (50, 200, 50), 2, cv2.LINE_AA)

    mouth_col = C_ORANGE if status == "YAWNING" else (80, 80, 240)
    cv2.polylines(frame, [np.array([pts_xy[i] for i in MOUTH])], True, mouth_col, 2, cv2.LINE_AA)

    # 3. Iris & Red Landmark Dots
    for idx_set in [LEFT_IRIS, RIGHT_IRIS]:
        try:
            cx = int(np.mean([pts_xy[i][0] for i in idx_set]))
            cy = int(np.mean([pts_xy[i][1] for i in idx_set]))
            cv2.circle(frame, (cx, cy), 6, (230, 130, 50), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (230, 130, 50), -1, cv2.LINE_AA)
        except Exception:
            pass

    for pt in pts_xy:
        cv2.circle(frame, pt, 1, (60, 60, 220), -1, cv2.LINE_AA)

    # 4. Floating Text
    nose_x, nose_y = pts_xy[1]
    ear_val = (compute_ear(lms, LEFT_EYE, w, h) + compute_ear(lms, RIGHT_EYE, w, h)) / 2.0
    mar_val = compute_mar(lms, MOUTH, w, h)
    cv2.putText(frame, f"EAR:{ear_val:.2f}", (nose_x - 40, nose_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"MAR:{mar_val:.2f}", (nose_x - 40, nose_y - 2),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, mouth_col, 1, cv2.LINE_AA)

    # 5. Overlaid Action Strip (Banner) for Alerts
    if status in ["DROWSY", "YAWNING"]:
        banner_overlay = frame.copy()
        msg = "🚨 DROWSINESS DETECTED" if status == "DROWSY" else "🥱 YAWNING DETECTED"
        color = (200, 30, 30) if status == "DROWSY" else (220, 120, 0)
        
        # Draw translucent background pill
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = h - 60
        
        cv2.rectangle(banner_overlay, (tx - 30, ty - 45), (tx + text_size[0] + 30, ty + 20), (15, 15, 20), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (tx - 30, ty - 45), (tx + text_size[0] + 30, ty + 20), color, 3)
        # Text
        cv2.putText(frame, msg, (tx, ty - 5), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  ALERT MANAGER
# ─────────────────────────────────────────────
class AlertManager:
    def __init__(self):
        self._last_alert = 0
        self._lock = threading.Lock()
        
    def trigger(self, play_audio=True):
        now = time.time()
        with self._lock:
            if now - self._last_alert >= ALERT_COOLDOWN:
                self._last_alert = now
                if play_audio:
                    threading.Thread(target=self._beep, daemon=True).start()
                return True
        return False

    def _beep(self):
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1800, 250)
                time.sleep(0.1)
        except Exception:
            pass


# ─────────────────────────────────────────────
#  STREAMLIT UI LAYOUT & LOOP
# ─────────────────────────────────────────────
def app():
    # Centered Header
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.markdown("<h1 style='text-align: center; color: #1a1a2e; margin-bottom: 0;'>👁️ Break the Drowsiness</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 40px;'>Real-time AI monitoring for EAR (Eye) & MAR (Mouth) metrics.</p>", unsafe_allow_html=True)

    # Initialize Session State
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    def toggle_cam():
        st.session_state.camera_running = not st.session_state.camera_running

    # SIDEBAR CONTROLS
    st.sidebar.markdown("<h2 style='color: #4f46e5;'>⚙️ System Settings</h2>", unsafe_allow_html=True)
    st.sidebar.button(
        "� Stop Camera Focus" if st.session_state.camera_running else "📸 Activate Camera Focus",
        on_click=toggle_cam,
        key="cam_btn"
    )
    
    play_audio = st.sidebar.checkbox("🔊 Enable Alert Audio Beep", value=True)
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Threshold Adjustments")
    ear_thresh = st.sidebar.slider("EAR Threshold (Eyes Closed)", 0.15, 0.35, 0.22, 0.01)
    mar_thresh = st.sidebar.slider("MAR Threshold (Yawning)", 0.25, 0.60, 0.38, 0.01)
    
    ear_frames = st.sidebar.number_input("Frames to trigger Drowsy", 5, 60, 20)
    mar_frames = st.sidebar.number_input("Frames to trigger Yawn", 5, 60, 25)

    # MAIN LAYOUT: BIG CAMERA AND RIGHT PANEL
    main_col, side_col = st.columns([2.5, 1], gap="large")

    with side_col:
        # Side metrics placeholders
        e_status = st.empty()
        e_ear    = st.empty()
        e_mar    = st.empty()
        
        # Flex block for events
        ev1, ev2 = st.columns(2)
        e_drowsy = ev1.empty()
        e_yawns  = ev2.empty()

    def render_metrics(status, ear, mar, drowsy_ev, yawn_ev):
        s_class = "status-safe" if status == "ALERT" else ("status-warning" if status == "WARNING" else "status-alert")
        
        e_status.markdown(f'<div class="metric-card"><div class="metric-label">System State</div><div class="metric-value {s_class}">{status}</div></div>', unsafe_allow_html=True)
        e_ear.markdown(f'<div class="metric-card"><div class="metric-label">Live EAR Score</div><div class="metric-value">{ear:.2f}</div></div>', unsafe_allow_html=True)
        e_mar.markdown(f'<div class="metric-card"><div class="metric-label">Live MAR Score</div><div class="metric-value">{mar:.2f}</div></div>', unsafe_allow_html=True)
        e_drowsy.markdown(f'<div class="metric-card" style="padding:15px;"><div class="metric-label" style="font-size:0.75rem;">Drowsy<br>Events</div><div class="metric-value" style="font-size:2rem; color:#f43f5e;">{drowsy_ev}</div></div>', unsafe_allow_html=True)
        e_yawns.markdown(f'<div class="metric-card" style="padding:15px;"><div class="metric-label" style="font-size:0.75rem;">Yawn<br>Events</div><div class="metric-value" style="font-size:2rem; color:#f59e0b;">{yawn_ev}</div></div>', unsafe_allow_html=True)

    with main_col:
        frame_placeholder = st.empty()

    if not st.session_state.camera_running:
        render_metrics("STANDBY", 0.0, 0.0, 0, 0)
        frame_placeholder.markdown("""
        <div style="background-color: #1a1a2e; width: 100%; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; border-radius: 20px; border: 4px solid white; box-shadow: 0 10px 40px rgba(0,0,0,0.15); box-sizing: border-box;">
            <h2 style="color: #4f46e5; text-align:center; margin:0; line-height:1.4;">
                <span style="font-size: 3rem;">�</span><br>
                SYSTEM STANDBY<br>
                <span style="font-size: 1.1rem; color: #888; font-weight: 400;">Click 'Activate Camera Focus' to begin</span>
            </h2>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # INIT MEDIAPIPE
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    alert_mgr = AlertManager()
    
    # State
    ear_counter = 0
    mar_counter = 0
    drowsy_events = 0
    yawn_count = 0
    frame_ts_ms = 0
    
    try:
        with mp_vision.FaceLandmarker.create_from_options(options) as detector:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera failed to read frame.")
                    break
                    
                frame = cv2.flip(frame, 1)
                fw, fh = frame.shape[1], frame.shape[0]
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                frame_ts_ms += 1000 // 30
                result = detector.detect_for_video(mp_img, frame_ts_ms)
                
                ear_val, mar_val = 0.0, 0.0
                status = "SEARCHING..."
                
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    ear_val = (compute_ear(lms, LEFT_EYE, fw, fh) + compute_ear(lms, RIGHT_EYE, fw, fh)) / 2.0
                    mar_val = compute_mar(lms, MOUTH, fw, fh)
                    
                    # LOGIC: EAR
                    if ear_val < ear_thresh:
                        ear_counter += 1
                        if ear_counter == ear_frames:
                            drowsy_events += 1
                        if ear_counter >= ear_frames:
                            alert_mgr.trigger(play_audio)
                    else:
                        ear_counter = 0
                        
                    # LOGIC: MAR
                    if mar_val > mar_thresh:
                        mar_counter += 1
                        if mar_counter == mar_frames:
                            yawn_count += 1
                        if mar_counter >= mar_frames:
                            alert_mgr.trigger(play_audio)
                    else:
                        mar_counter = 0
                        
                    # Determine Status
                    if ear_counter >= ear_frames: status = "DROWSY"
                    elif mar_counter >= mar_frames: status = "YAWNING"
                    elif ear_val < ear_thresh + 0.04: status = "WARNING"
                    else: status = "ALERT"
                    
                    # Draw
                    draw_face_mesh(rgb, lms, fw, fh, status)
                
                # Render UI
                render_metrics(status, ear_val, mar_val, drowsy_events, yawn_count)
                
                # Show video (streamlit handles RGB images perfectly)
                frame_placeholder.image(rgb, use_container_width=True)

    finally:
        cap.release()

if __name__ == "__main__":
    app()
