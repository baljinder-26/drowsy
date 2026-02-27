import cv2
import numpy as np
import streamlit as st
import base64
import mediapipe as mp
from scipy.spatial import distance as dist
import time
import os
import urllib.request
import threading
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# -- Page Configuration --
st.set_page_config(
    page_title="Drowsiness & Yawn Tracker", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes scrollGrid {
        0% { background-position: 0px 0px, 0px 0px, 0px 0px; }
        100% { background-position: 50px 50px, 50px 50px, 0px 0px; }
    }

    /* High-tech Live Scrolling Grid Background */
    .stApp {
        background-color: #050814 !important;
        background-image: 
            linear-gradient(rgba(0, 255, 204, 0.2) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 204, 0.2) 1px, transparent 1px),
            radial-gradient(circle at 50% 50%, rgba(121, 40, 202, 0.1) 0%, transparent 60%);
        background-size: 50px 50px, 50px 50px, 100% 100%;
        animation: scrollGrid 3s linear infinite;
        background-attachment: fixed;
    }
    
    /* Do NOT hide the header completely so the sidebar toggle arrow is still clickable when collapsed */
    .stApp > header {
        background-color: transparent !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 0rem !important;
        margin-top: -20px;
        padding-bottom: 2rem !important;
        max-width: 1300px;
    }
    
    /* Hero Header Update - Matching the image */
    .hero-header {
        padding: 50px 0px 40px 0px;
        margin-bottom: 30px;
        text-align: left;
        max-width: 650px;
    }
    
    @keyframes textShine {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    @keyframes floatText {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .hero-header h1 {
        font-size: 5rem !important;
        font-weight: 800 !important;
        margin: 0 0 20px 0 !important;
        padding: 0 !important;
        background: linear-gradient(90deg, #ffffff 0%, #00ffcc 20%, #ff007f 40%, #ffffff 60%, #ffffff 100%) !important;
        background-size: 200% auto !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        animation: textShine 4s linear infinite, floatText 4s ease-in-out infinite !important;
        letter-spacing: -2px !important;
        line-height: 1.05 !important;
    }
    
    .hero-header p {
        font-size: 1.15rem;
        color: #a0aec0;
        font-weight: 400;
        margin-bottom: 30px;
        line-height: 1.6;
    }

    .action-row {
        display: flex;
        align-items: center;
        gap: 30px;
        margin-top: 30px;
        margin-bottom: 20px;
    }

    /* "Explore" looking Button style for aesthetic */
    div.stButton > button[kind="primary"] {
        background: #ffffff !important;
        color: #090c15 !important;
        font-weight: 800 !important;
        height: 55px !important;
        font-size: 1.1rem !important;
        border-radius: 4px !important;
        box-shadow: -6px 6px 0px 0px #ff007f !important;
        transition: all 0.2s ease !important;
        border: 2px solid transparent !important;
        animation: none !important;
        background-image: none !important;
    }
    
    div.stButton > button[kind="primary"] p {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #090c15 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translate(-3px, 3px) !important;
        box-shadow: -3px 3px 0px 0px #ff007f !important;
        background: #ffffff !important;
        color: #090c15 !important;
    }
    
    .custom-btn-text {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        cursor: pointer;
    }

    /* Metric Cards - Technical Look */
    .metric-card {
        background: #0d1323;
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 6px;
        padding: 25px 20px;
        text-align: left;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: #00ffcc;
    }
    
    .metric-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .metric-value-span {
        font-size: 1.2rem;
        color: #475569;
        font-weight: 500;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #00ffcc;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    @keyframes gradientButton {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    div.stButton > button {
        background: linear-gradient(135deg, #ff007f 0%, #7928ca 50%, #00ffcc 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientButton 4s ease infinite !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        transition: transform 0.2s !important;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.4) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 255, 204, 0.6) !important;
        color: white !important;
    }

    div.stButton > button:focus {
        color: white !important;
    }

    /* Navigation Simulation */
    .top-nav {
        display: flex;
        gap: 30px;
        align-items: center;
        padding-top: 10px;
        padding-bottom: 20px;
        position: relative;
        z-index: 50;
    }
    
    .top-nav a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
        transition: color 0.2s;
    }
    
    .top-nav a:hover {
        color: #ffffff;
    }
    
    .nav-brand {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: 20px;
    }
    
    /* Customizing sidebar */
    [data-testid="stSidebar"] {
        background-color: #05080f !important;
        border-right: 1px solid rgba(0, 255, 204, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Status Styles */
    .status-alert {
        background: rgba(255, 0, 127, 0.05);
        color: #ff007f;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        font-weight: 800;
        font-size: 1.2rem;
        border: 1px solid #ff007f;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.15);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .status-ok {
        background: rgba(0, 255, 204, 0.05);
        color: #00ffcc;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        font-weight: 800;
        font-size: 1.2rem;
        border: 1px solid #00ffcc;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.1);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.05);
        color: #fbbf24;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        font-weight: 800;
        font-size: 1.2rem;
        border: 1px solid #f59e0b;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Image styling */
    img[data-testid="stImage"] {
        border-radius: 8px;
        border: 1px solid rgba(0, 255, 204, 0.3);
        box-shadow: 0 0 40px rgba(0, 0, 0, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Fake Navigation Bar
st.markdown("""
<div class="top-nav">
    <span class="nav-brand">AI Labs</span>
    <a href="#">About</a>
    <a href="#" style="color: #ffffff; font-weight: 600;">Explore</a>
    <a href="#">Learn</a>
    <a href="#">Connect</a>
</div>
""", unsafe_allow_html=True)

# -- MediaPipe Setup --
@st.cache_resource
def load_face_landmarker():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
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

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [13, 14, 78, 308]

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

def get_audio_html(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f'<audio autoplay loop><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    except Exception as e:
        return ""

# Removed speak_alert due to server-side hardware limitations on Streamlit Cloud.
# We use browser-side Speech Synthesis (JavaScript) instead.

# -- Page UI --
st.markdown("""
<div class="hero-header" style="margin-bottom: 0px;">
      <h1>Generate Alert ,<br>Enhance Protect</h1>
    <p>
        The future is here. This space showcases my latest creation —
        a deep learning–based Drowsiness Detection System built for safety
        and real-world impact.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("") # small spacer
st.sidebar.title("Telemetry Control") 
st.sidebar.markdown("Engage the systems below to begin.")

if 'run_app' not in st.session_state:
    st.session_state.run_app = False

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Start Scanner", use_container_width=True):
        st.session_state.run_app = True
with col2:
    if st.button("Stop Scanner", use_container_width=True):
        st.session_state.run_app = False

run_app = st.session_state.run_app

st.sidebar.markdown("---")
st.sidebar.markdown("### Sensitivity Matrix")
CLOSED_FRAME_THRESHOLD = st.sidebar.slider("Drowsy Tolerance (Frames)", min_value=3, max_value=50, value=18)
YAWN_FRAME_THRESHOLD = st.sidebar.slider("Yawn Tolerance (Frames)", min_value=2, max_value=20, value=5)

st.sidebar.markdown("<br><br><br><p style='text-align: center; color: #475569; font-size: 0.8rem; font-weight: 500;'>Built by Abhijay Parashar</p>", unsafe_allow_html=True)

audio_placeholder = st.empty()

# --- Dashboard Layout ---
col_video, col_metrics, col_padding = st.columns([3, 1.2, 0.2], gap="large")

# --- Dashboard Layout ---
col_video, col_metrics, col_padding = st.columns([3, 1.2, 0.2], gap="large")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

with col_video:
    st.markdown("### Live Feed Monitor")

with col_metrics:
    status_ui = st.empty()
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    drowsy_ui = st.empty()
    yawn_ui = st.empty()

# For Streamlit cloud webrtc is required. We use a callback approach below.
# Note: Streamlit-webrtc processes frames in a separate thread.
# Simple visual alerts will occur directly on the video feed.

class VideoProcessor:
    def __init__(self):
        self.closed_frames = 0
        self.yawn_frames = 0
        self.alarm_on = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (800, 600))
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        eyes_found = False
        yawn_detected_current = False
        face_detected_last = False

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_mesh.detect(mp_image)

            if results.face_landmarks:
                face_detected_last = True
                landmarks = results.face_landmarks[0]

                left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear > 0.21: 
                    eyes_found = True

                mar = calculate_mar(landmarks, LIPS, w, h)
                
                if mar > 0.45:
                    yawn_detected_current = True
                    cv2.putText(img, "YAWN DETECTED", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 127), 2)

                for idx in LEFT_EYE + RIGHT_EYE + LIPS:
                    pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                    cv2.circle(img, pt, 2, (0, 255, 204), -1)
        except Exception as e:
            # Fallback if MediaPipe fails
            pass

        if face_detected_last:
            if not eyes_found:
                self.closed_frames += 1
            else:
                self.closed_frames = 0
                
            if yawn_detected_current:
                self.yawn_frames += 1
            else:
                self.yawn_frames = max(0, int(self.yawn_frames - 0.5))
        else:
            self.closed_frames = 0
            self.yawn_frames = 0

        # Draw visual indicators
        if self.closed_frames >= CLOSED_FRAME_THRESHOLD or self.yawn_frames >= YAWN_FRAME_THRESHOLD:
            alert_type = "SLEEP DETECTED" if self.closed_frames >= CLOSED_FRAME_THRESHOLD else "FATIGUE (YAWN)"
            cv2.putText(img, f"ALERT: {alert_type}", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 127), 3)
            # Neon pink frame
            cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (127, 0, 255), 10)
        else:
            cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (204, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_video:
    if not run_app:
        # Hide the WebRTC container visually when offline so we only see the customized offline warning
        st.markdown("""
        <style>
            #drowsy-detector { display: none !important; }
            iframe[title="streamlit_webrtc.webrtc_streamer"] { display: none !important; }
        </style>
        <div style='text-align: left; padding: 60px 40px; background: #0d1323; border-radius: 6px; border: 1px dashed rgba(0, 255, 204, 0.3); max-width: 100%;'>
            <h2 style='color: #ffffff; font-size: 2rem; margin-bottom: 20px; font-weight: 800; letter-spacing: -1px;'>Scanner Offline</h2>
            <p style='color: #94a3b8; font-size: 1.1rem; font-weight: 500;'>Initialize the scanner using the sidebar toggle to begin telemetry feed.</p>
        </div>
        """, unsafe_allow_html=True)

    # Always keep the component mounted in the DOM to prevent expensive hard-reloads of WebRTC modules
    ctx = webrtc_streamer(
        key="drowsy-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True, 
            "audio": False
        },
        async_processing=True,
        desired_playing_state=run_app
    )

if run_app:
    # Continuous loop to poll the video processor and update the Streamlit UI metrics and audio/speech
    if ctx and ctx.state.playing:
        if ctx.video_processor:
            closed = ctx.video_processor.closed_frames
            yawn = ctx.video_processor.yawn_frames
            
            drowsy_ui.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{closed} <span class="metric-value-span">/ {CLOSED_FRAME_THRESHOLD}</span></div>
                    <div class="metric-label">Eye Closure</div>
                </div>
                """, unsafe_allow_html=True
            )
            
            yawn_ui.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{int(yawn)} <span class="metric-value-span">/ {YAWN_FRAME_THRESHOLD}</span></div>
                    <div class="metric-label">Yawn Events</div>
                </div>
                """, unsafe_allow_html=True
            )
            
            if closed >= CLOSED_FRAME_THRESHOLD or yawn >= YAWN_FRAME_THRESHOLD:
                alert_type = "SLEEP DETECTED" if closed >= CLOSED_FRAME_THRESHOLD else "FATIGUE (YAWN)"
                status_ui.markdown(f'<div class="status-alert">⚠️ {alert_type}</div>', unsafe_allow_html=True)
                
                if not st.session_state.alarm_on:
                    st.session_state.alarm_on = True
                    
                    # Browser Web Speech API
                    speech_text = "Drowsy detected." if closed >= CLOSED_FRAME_THRESHOLD else "Yawn detected."
                    js_speech = f"""
                    <script>
                        var msg = new SpeechSynthesisUtterance("{speech_text}");
                        var voices = window.speechSynthesis.getVoices();
                        var maleVoice = voices.find(v => 
                            v.name.toLowerCase().includes('male') || 
                            v.name.toLowerCase().includes('david') || 
                            v.name.toLowerCase().includes('mark')
                        );
                        if (maleVoice) {{
                            msg.voice = maleVoice;
                        }}
                        msg.pitch = 0.8; 
                        window.speechSynthesis.speak(msg);
                    </script>
                    """
                    
                    audio_html = get_audio_html("alarm.wav")
                    audio_placeholder.markdown(audio_html + js_speech, unsafe_allow_html=True)
            else:
                st.session_state.alarm_on = False
                audio_placeholder.empty()
                status_ui.markdown('<div class="status-ok">WEBRTC SYSTEM NOMINAL</div>', unsafe_allow_html=True)
        
        # We avoid the 'while True' loop here because it can hang Streamlit Cloud.
        # Streamlit will rerun this script automatically when state changes occur.
        # To get real-time updates for metrics, we use a slower rerun or simply rely on the video feed.
        # For professional real-time metric updates, adding st_autorefresh is recommended.
        st.button("Update Telemetry", key="refresh_but")
else:
    st.session_state.alarm_on = False
    audio_placeholder.empty()
