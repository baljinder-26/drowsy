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
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800&display=swap');
    
    /* Global App Background with Live Animation */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #0b0c10, #130f24, #0b0c10, #1f1035, #0b0c10);
        background-size: 400% 400%;
        animation: background-live 15s ease infinite;
    }
    
    @keyframes background-live {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Make header and sidebar cohesive */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(11, 12, 16, 0.7) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
    }
    
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
    
    /* Match Header Box from Screenshot */
    .hero-header {
        padding: 40px 10px 40px 10px;
        color: white;
        margin-bottom: 20px;
        text-align: left; /* Shift text alignment to match reference */
    }
    
    .hero-header h1 {
        font-family: 'Montserrat', sans-serif;
        font-size: 5.5rem;
        font-weight: 700;
        margin: 0 0 25px 0;
        padding: 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    
    .text-rainbow {
        background: linear-gradient(to right, #ff2a2a, #ff7a2a, #ffd700, #32cd32, #1de9b6, #4dd0e1, #818cf8, #e81cff, #f06292, #ff2a2a);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: text-shine 3s linear infinite;
    }

    @keyframes text-shine {
        to {
            background-position: 200% center;
        }
    }
    
    .hero-header p {
        font-size: 1.25rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
        letter-spacing: 0px;
        line-height: 1.6;
        max-width: 800px;
    }
    
    /* Interactive Animated Section Headers */
    .section-header {
        color: #e2e8f0;
        font-size: 2rem;
        margin-bottom: 25px;
        font-weight: 800;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        padding-bottom: 10px;
        border-bottom: 2px solid transparent;
        display: inline-block;
        cursor: pointer;
    }
    
    .section-header:hover {
        transform: translateX(10px) scale(1.02);
        color: #ffffff;
        text-shadow: 0 0 15px rgba(139, 92, 246, 0.6);
        border-bottom: 2px solid rgba(168, 85, 247, 0.8);
    }
    
    .section-icon {
        color: #38bdf8;
        display: inline-block;
        animation: bounce-icon 3s ease-in-out infinite;
        margin-right: 10px;
        font-size: 2.2rem;
    }
    
    @keyframes bounce-icon {
        0% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-5px) rotate(-10deg); text-shadow: 0 0 15px rgba(56, 189, 248, 0.8); }
        75% { transform: translateY(2px) rotate(10deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 15px 40px 0 rgba(139, 92, 246, 0.2);
    }
    
    .metric-value {
        font-size: 4rem;
        font-weight: 800;
        color: #d8b4fe;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    
    .metric-value-span {
        font-size: 1.5rem;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .status-alert {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(153, 27, 27, 0.2));
        backdrop-filter: blur(10px);
        color: #fca5a5;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.6rem;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
        animation: pulse-red 1.5s infinite alternate;
        letter-spacing: 1.5px;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    .status-ok {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(4, 120, 87, 0.1));
        backdrop-filter: blur(10px);
        color: #6ee7b7;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 1.5px;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(180, 83, 9, 0.1));
        backdrop-filter: blur(10px);
        color: #fcd34d;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 1.5px;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    @keyframes pulse-red {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(239, 68, 68, 0.2); }
        100% { transform: scale(1.02); box-shadow: 0 0 40px rgba(239, 68, 68, 0.5); }
    }
    
    /* Custom Sidebar Button Styling matching image */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #e81cff, #ec4899) !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(236, 72, 153, 0.4) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 15px !important;
        transition: transform 0.2s ease, box-shadow 0.2s !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(236, 72, 153, 0.6) !important;
    }
    
    /* Main Area Button Styling matching image */
    section[data-testid="stMain"] .stButton > button {
        background: white !important;
        border: none !important;
        border-left: 5px solid var(--btn-border-color, #e81cff) !important;
        border-radius: 8px !important;
        color: black !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        padding: 10px 25px !important;
        margin-top: -20px !important;
        margin-bottom: 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 15px var(--btn-border-color, #e81cff) !important;
        animation: main-btn-glow 2s infinite alternate !important;
    }
    
    @keyframes main-btn-glow {
        0% { 
            box-shadow: 0 0 8px var(--btn-border-color, rgba(232, 28, 255, 0.5)); 
            transform: scale(1) translateY(0); 
        }
        100% { 
            box-shadow: 0 0 25px var(--btn-border-color, rgba(232, 28, 255, 0.9)); 
            transform: scale(1.03) translateY(-3px); 
        }
    }

    section[data-testid="stMain"] .stButton > button:hover {
        background: #f8fafc !important;
        transform: scale(1.05) translateY(-5px) !important;
        box-shadow: 0 0 35px var(--btn-border-color, #e81cff) !important;
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
if 'app_running' not in st.session_state:
    st.session_state.app_running = False

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
    <h1><span class="text-rainbow">Awake Eyes.</span><br><span class="text-rainbow">Safe Roads.</span></h1>
    <p>The future is here. This is the space for our latest creations, innovations, and cutting-edge technologies for the greater good.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: white; font-weight: 700; font-size: 1.6rem; margin-bottom: 5px;'>Telemetry Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #94a3b8; margin-bottom: 15px;'>Engage the systems below to begin.</p>", unsafe_allow_html=True)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.button("Start Scanner", use_container_width=True):
        st.session_state.app_running = True
        st.rerun()

with col_s2:
    if st.button("Stop Scanner", use_container_width=True):
        st.session_state.app_running = False
        st.rerun()

run_app = st.session_state.app_running
st.sidebar.markdown("<h3 style='color: white; margin-top: 25px; margin-bottom: 10px;'>Sensitivity Matrix</h3>", unsafe_allow_html=True)
CLOSED_FRAME_THRESHOLD = st.sidebar.slider("Drowsy Tolerance (Frames)", min_value=3, max_value=50, value=18)
YAWN_FRAME_THRESHOLD = st.sidebar.slider("Yawn Tolerance (Frames)", min_value=2, max_value=20, value=5)

# Placeholder for audio
audio_placeholder = st.empty()

# --- Dashboard Layout ---
col_metrics, col_video = st.columns([1, 2.3], gap="large")

with col_metrics:
    # Adding a button directly below the header matching UI ref 
    if not st.session_state.app_running:
        st.markdown("<style>:root { --btn-border-color: #e81cff; }</style>", unsafe_allow_html=True)
        if st.button("Start Drowsy Detection", use_container_width=False):
            st.session_state.app_running = True
            st.rerun()
    else:
        st.markdown("<style>:root { --btn-border-color: #1de9b6; }</style>", unsafe_allow_html=True)
        if st.button("Monitor Driver Status", use_container_width=False):
            st.session_state.app_running = False
            st.rerun()
    drowsy_ui = st.empty()
    yawn_ui = st.empty()
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    status_ui = st.empty()

with col_video:
    st.markdown("<div class='section-header'><span class='section-icon'>📷</span> Real-time Analysis Feed</div>", unsafe_allow_html=True)
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