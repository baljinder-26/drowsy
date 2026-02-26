import cv2
import numpy as np
import streamlit as st
import base64
import mediapipe as mp
from scipy.spatial import distance as dist
import time
# -- Page Configuration --
st.set_page_config(
    page_title="Drowsiness & Yawn Tracker", 
    page_icon="�", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 40px 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 40px;
        text-align: center;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .hero-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 10px 0;
        padding: 0;
        color: #ffffff;
        letter-spacing: -1px;
    }
    
    .hero-header p {
        font-size: 1.2rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }

    .metric-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
    }
    
    .metric-value {
        font-size: 4rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    
    .metric-value-span {
        font-size: 1.5rem;
        color: #475569;
        font-weight: 600;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .status-alert {
        background: linear-gradient(135deg, #ef4444, #991b1b);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.6rem;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.6);
        animation: pulse-red 1s infinite alternate;
        letter-spacing: 1px;
        border: 1px solid #f87171;
    }
    
    .status-ok {
        background: linear-gradient(135deg, #10b981, #047857);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 1px;
        border: 1px solid #34d399;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b, #b45309);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 1px;
        border: 1px solid #fbbf24;
    }
    
    @keyframes pulse-red {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        100% { transform: scale(1.02); box-shadow: 0 0 40px rgba(239, 68, 68, 0.9); }
    }
    
    /* Global st image styling */
    img[data-testid="stImage"] {
        border-radius: 20px;
        border: 2px solid #334155;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# -- MediaPipe Setup --
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
import urllib.request

@st.cache_resource
def load_face_landmarker():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        # Auto-download the specialized MediaPipe task model if missing
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

face_mesh = load_face_landmarker()

# Indicies for MediaPipe Face Mesh
# Order: [Outer, Top 1, Top 2, Inner, Bottom 2, Bottom 1]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [13, 14, 78, 308] # top inner, bottom inner, left corner, right corner

def calculate_ear(landmarks, indices, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    ear = (dist.euclidean(p[1], p[5]) + dist.euclidean(p[2], p[4])) / (2.0 * dist.euclidean(p[0], p[3]))
    return ear

def calculate_mar(landmarks, indices, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    mar = dist.euclidean(p[0], p[1]) / (dist.euclidean(p[2], p[3]) + 1e-6)
    return mar

# -- Global State --
if 'alarm_on' not in st.session_state:
    st.session_state.alarm_on = False

# Read audio file to base64 for playing in browser
def get_audio_html(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f'<audio autoplay loop><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    except Exception as e:
        return ""

# -- Page UI --
st.markdown("""
<div class="hero-header">
    <h1>🚗 Driver Safety AI Dashboard</h1>
    <p>Real-time autonomous fatigue and yawning detection powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ System Control")
st.sidebar.markdown("Toggle the camera to start AI inference.")
run_app = st.sidebar.toggle("Start Camera System", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("### Detection Sensitivity")
CLOSED_FRAME_THRESHOLD = st.sidebar.slider("Drowsiness Threshold (Frames)", min_value=3, max_value=50, value=18, help="How many consecutive frames the eyes must be closed to trigger an alert.")
YAWN_FRAME_THRESHOLD = st.sidebar.slider("Yawning Threshold (Frames)", min_value=2, max_value=20, value=5, help="How many consecutive frames the mouth must be wide open to trigger an alert.")

# Placeholder for audio
audio_placeholder = st.empty()

# --- Dashboard Layout ---
col_metrics, col_video = st.columns([1, 2.3], gap="large")

with col_metrics:
    st.markdown("<h3 style='color: #e2e8f0; font-size: 1.3rem; margin-bottom: 20px; font-weight: 600;'><span style='color: #38bdf8'>⚡</span> System Telemetry</h3>", unsafe_allow_html=True)
    drowsy_ui = st.empty()
    yawn_ui = st.empty()
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    status_ui = st.empty()

with col_video:
    st.markdown("<h3 style='color: #e2e8f0; font-size: 1.3rem; margin-bottom: 20px; font-weight: 600;'><span style='color: #38bdf8'>📷</span> Real-time Analysis Feed</h3>", unsafe_allow_html=True)
    frame_window = st.image([], use_container_width=True)

if run_app:
    if "camera" not in st.session_state:
        st.session_state.camera = cv2.VideoCapture(0)
        st.session_state.camera.set(3, 640)
        st.session_state.camera.set(4, 480)
    cap = st.session_state.camera

    closed_frames = 0
    yawn_frames = 0
    face_detected_last = False

    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Resize for speed + stability
        frame = cv2.resize(frame, (640, 480))
        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Convert numpy frame to MediaPipe Image object format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_mesh.detect(mp_image)

        eyes_found = False
        yawn_detected_current = False
        face_detected_last = False

        if results.face_landmarks:
            face_detected_last = True
            landmarks = results.face_landmarks[0]

            # Calculate Eye Aspect Ratio (EAR)
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Typically, EAR < 0.22 indicates closed eyes
            if avg_ear > 0.21: 
                eyes_found = True

            # Calculate Mouth Aspect Ratio (MAR)
            mar = calculate_mar(landmarks, LIPS, w, h)
            
            if mar > 0.45:
                yawn_detected_current = True
                cv2.putText(frame, "Yawning", (30, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            # Draw visual tracking nodes on the eyes and mouth
            for idx in LEFT_EYE + RIGHT_EYE + LIPS:
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # ── Smarter Drowsiness Logic ─────────────────────────────
        if face_detected_last:
            if not eyes_found:
                closed_frames += 1
            else:
                closed_frames = 0
                
            if yawn_detected_current:
                yawn_frames += 1
            else:
                yawn_frames = max(0, int(yawn_frames - 0.5))
        else:
            closed_frames = 0
            yawn_frames = 0

        # ── Updates stream ──────────────────────────────────────────
        # UI Updates - Custom Metrics Boxes
        drowsy_ui.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{closed_frames} <span class="metric-value-span">/ {CLOSED_FRAME_THRESHOLD}</span></div>
                <div class="metric-label">Closed Frames</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        yawn_ui.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{int(yawn_frames)} <span class="metric-value-span">/ {YAWN_FRAME_THRESHOLD}</span></div>
                <div class="metric-label">Yawn Frames</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # ── Alert Logic ─────────────────────────────────────────────
        if closed_frames >= CLOSED_FRAME_THRESHOLD or yawn_frames >= YAWN_FRAME_THRESHOLD:
            alert_type = "SLEEPING" if closed_frames >= CLOSED_FRAME_THRESHOLD else "YAWNING"
            status_ui.markdown(f'<div class="status-alert">🚨 CRITICAL: DRIVER IS {alert_type}! 🚨</div>', unsafe_allow_html=True)
            
            # Print Alert inside the Video Frame
            cv2.putText(frame, f"{alert_type} ALERT!", (50, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            # Heavy Red Border
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 15)

            if not st.session_state.alarm_on:
                st.session_state.alarm_on = True
                # Play audio via base64 encoded HTML string
                audio_html = get_audio_html("alarm.wav")
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.session_state.alarm_on = False
            audio_placeholder.empty() # Stop audio
            if face_detected_last:
                status_ui.markdown('<div class="status-ok">🟢 DRIVER ACTIVE & ALERT</div>', unsafe_allow_html=True)
            else:
                status_ui.markdown('<div class="status-warning">🟠 SIGNAL LOST: FACING AWAY</div>', unsafe_allow_html=True)
                
            # Mild Green Border
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 2)

        # Streamlit expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(rgb_frame, channels="RGB")

    # While loop breaks when run_app is False
    if "camera" in st.session_state:
        st.session_state.camera.release()
        del st.session_state.camera
    st.session_state.alarm_on = False
    audio_placeholder.empty()
else:
    if "camera" in st.session_state:
        st.session_state.camera.release()
        del st.session_state.camera
    # Beautiful Empty State using new modern dashboard styling
    st.markdown("""
        <div style='text-align: center; padding: 100px 20px; background: rgba(30,41,59,0.5); border-radius: 20px; border: 2px dashed #475569; max-width: 800px; margin: 40px auto; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
            <h2 style='color: #94a3b8; font-size: 2rem; margin-bottom: 20px;'>System Standby Mode</h2>
            <p style='color: #64748b; font-size: 1.2rem;'>Enable the <strong style='color: #38bdf8'>Start Camera System</strong> toggle in the sidebar to initialize AI inference engine.</p>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.alarm_on = False
    audio_placeholder.empty()