"""
Gestura v2.0 — Streamlit Web App (Updated)
==========================================
Run with: streamlit run gestura_app.py
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gestura v2.0 — Air Drawing Pad",
    page_icon="✍️",
    layout="wide"
)

# ── Vibrant CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Title */
    h1 {
        background: linear-gradient(90deg, #00ff88, #00ccff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-family: 'Courier New', monospace;
        font-size: 2.5em !important;
        text-shadow: 0 0 30px #00ff88;
    }

    h3 {
        color: #00ccff !important;
        font-family: monospace;
        border-bottom: 1px solid #00ff8844;
        padding-bottom: 5px;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #1a1a3e, #2d2d6e);
        color: white;
        border: 1px solid #00ff88;
        padding: 10px;
        font-size: 14px;
        font-family: monospace;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00ff88, #00ccff);
        color: black;
        transform: scale(1.02);
        box-shadow: 0 0 15px #00ff88;
    }

    /* Selectbox */
    .stSelectbox>div>div {
        background-color: #1a1a3e !important;
        color: white !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px;
    }

    /* Slider */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #00ff88, #00ccff) !important;
    }

    /* Toggle */
    .stToggle>label {
        color: white !important;
        font-family: monospace;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a3e99, #2d2d6e99);
        border: 1px solid #00ff8866;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        font-family: monospace;
        backdrop-filter: blur(10px);
    }

    .colour-box {
        background: linear-gradient(135deg, #1a1a3e, #2d2d6e);
        border: 1px solid #ff00ff66;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        font-family: monospace;
    }

    /* Status badges */
    .status-on {
        background: linear-gradient(90deg, #00ff88, #00cc66);
        color: black;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .status-off {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555;
        font-family: monospace;
        padding: 20px;
        background: linear-gradient(90deg, #00ff8811, #00ccff11, #ff00ff11);
        border-radius: 15px;
        margin-top: 20px;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00ff88, #00ccff, transparent);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Voice Recognition ─────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ── Colours ───────────────────────────────────────────────────────────────────
COLOURS = {
    "⚪ White":   (255, 255, 255),
    "🔴 Red":     (0,   0,   255),
    "🟢 Green":   (0,   255, 0  ),
    "🔵 Blue":    (255, 0,   0  ),
    "🟡 Yellow":  (0,   255, 255),
    "🟣 Purple":  (255, 0,   255),
    "🟠 Orange":  (0,   165, 255),
    "🩷 Pink":    (147, 20,  255),
    "🩵 Cyan":    (255, 255, 0  ),
    "🤍 Silver":  (192, 192, 192),
}

PINCH_THRESHOLD = 40

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ── Shape Correction ──────────────────────────────────────────────────────────
def correct_shape(canvas, stroke_points, colour_bgr, thickness):
    if len(stroke_points) < 5:
        return
    pts    = np.array(stroke_points, dtype=np.int32)
    hull   = cv2.convexHull(pts)
    peri   = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    n      = len(approx)
    x, y, w, h = cv2.boundingRect(pts)

    for i in range(1, len(stroke_points)):
        cv2.line(canvas, stroke_points[i-1], stroke_points[i],
                 (0, 0, 0), thickness + 12)

    if n <= 2 or (w < 25 or h < 25):
        cv2.line(canvas, tuple(pts[0]), tuple(pts[-1]), colour_bgr, thickness)
    elif n == 3:
        cv2.polylines(canvas, [approx], True, colour_bgr, thickness)
    elif n == 4:
        cv2.rectangle(canvas, (x, y), (x+w, y+h), colour_bgr, thickness)
    else:
        center = (x + w//2, y + h//2)
        radius = (w + h) // 4
        cv2.circle(canvas, center, radius, colour_bgr, thickness)


# ── Voice Listener ────────────────────────────────────────────────────────────
_voice_cmd  = ""
_voice_lock = threading.Lock()
_listening  = False

def _voice_thread():
    global _voice_cmd, _listening
    if not VOICE_AVAILABLE:
        return
    try:
        r = sr.Recognizer()
        with sr.Microphone() as src:
            r.adjust_for_ambient_noise(src, duration=0.3)
            audio = r.listen(src, timeout=4, phrase_time_limit=3)
            text  = r.recognize_google(audio).lower()
            with _voice_lock:
                _voice_cmd = text
    except Exception:
        pass
    _listening = False

def start_voice():
    global _listening
    if VOICE_AVAILABLE and not _listening:
        _listening = True
        threading.Thread(target=_voice_thread, daemon=True).start()

def get_voice_cmd():
    global _voice_cmd
    with _voice_lock:
        cmd = _voice_cmd
        _voice_cmd = ""
    return cmd


# ── Video Processor ───────────────────────────────────────────────────────────
class GesturaProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas         = None
        self.prev_x         = None
        self.prev_y         = None
        self.drawing        = False
        self.was_drawing    = False
        self.eraser_mode    = False
        self.shape_mode     = True
        self.colour_bgr     = (255, 255, 255)
        self.thickness      = 5
        self.current_stroke = []
        self.clear_flag     = False
        self.voice_status   = ""
        self.voice_timer    = 0
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        if VOICE_AVAILABLE:
            start_voice()

    def recv(self, frame):
        from av import VideoFrame
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # Init / clear canvas
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if self.clear_flag:
            self.canvas[:] = 0
            self.clear_flag = False

        # Voice commands
        if VOICE_AVAILABLE:
            cmd = get_voice_cmd()
            if cmd:
                self.voice_status = f"🎤 Heard: '{cmd}'"
                self.voice_timer  = 90
                colour_map = {
                    "red": (0,0,255), "green": (0,255,0),
                    "blue": (255,0,0), "white": (255,255,255),
                    "yellow": (0,255,255), "purple": (255,0,255),
                    "orange": (0,165,255), "pink": (147,20,255),
                    "cyan": (255,255,0)
                }
                for name, bgr in colour_map.items():
                    if name in cmd:
                        self.colour_bgr  = bgr
                        self.eraser_mode = False
                if "clear" in cmd:
                    self.canvas[:] = 0
                if "eraser" in cmd:
                    self.eraser_mode = not self.eraser_mode
                if "shape" in cmd:
                    self.shape_mode = not self.shape_mode
                start_voice()

            if self.voice_timer > 0:
                self.voice_timer -= 1
            else:
                self.voice_status = "🎤 Listening..."
                start_voice()

        # Hand detection
        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        self.was_drawing = self.drawing
        cur_x, cur_y = None, None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                img, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            index_tip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
            cur_x = int(index_tip.x * w)
            cur_y = int(index_tip.y * h)
            tx    = int(thumb_tip.x * w)
            ty    = int(thumb_tip.y * h)

            pinch_dist   = np.hypot(cur_x - tx, cur_y - ty)
            self.drawing = (pinch_dist >= PINCH_THRESHOLD)

            dot_col = self.colour_bgr if (self.drawing and not self.eraser_mode) else (100,100,100)
            cv2.circle(img, (cur_x, cur_y), 8, dot_col, -1)

            if self.drawing and self.prev_x is not None:
                if self.eraser_mode:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y),
                             (cur_x, cur_y), (0,0,0), 40)
                else:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y),
                             (cur_x, cur_y), self.colour_bgr, self.thickness)
                    if self.shape_mode:
                        self.current_stroke.append((cur_x, cur_y))

        # Shape correction on pen lift
        if self.was_drawing and not self.drawing:
            if self.shape_mode and len(self.current_stroke) > 5 and not self.eraser_mode:
                correct_shape(self.canvas, self.current_stroke,
                              self.colour_bgr, self.thickness)
            self.current_stroke = []

        self.prev_x = cur_x if self.drawing else None
        self.prev_y = cur_y if self.drawing else None

        # Blend canvas
        mask   = self.canvas.sum(axis=2) > 0
        output = img.copy()
        output[mask] = cv2.addWeighted(img, 0.15, self.canvas, 0.85, 0)[mask]

        # HUD
        cv2.rectangle(output, (0, 0), (w, 40), (15, 15, 15), -1)
        mode  = "ERASER 🧹" if self.eraser_mode else "Drawing ✏️"
        shape = "[Shape ON]" if self.shape_mode else "[Shape OFF]"
        pen   = "Pen DOWN" if self.drawing else "Pen UP (pinch)"
        info  = f"  {mode} | {shape} | {pen}"
        cv2.putText(output, info, (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.colour_bgr, 1, cv2.LINE_AA)

        # Voice status
        if self.voice_status:
            cv2.putText(output, self.voice_status, (6, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 180), 1, cv2.LINE_AA)

        return VideoFrame.from_ndarray(output, format="bgr24")


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# ✍️ Gestura v2.0 — Air Drawing Pad")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-family:monospace;'>"
    "Draw in the air using your index finger • Pinch to lift pen • AI Shape Correction"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

col_cam, col_controls = st.columns([3, 1])

with col_controls:
    st.markdown("### 🎨 Colour")
    selected_colour = st.selectbox("Pick Colour", list(COLOURS.keys()), label_visibility="collapsed")

    st.markdown("### 📏 Stroke Size")
    thickness = st.slider("Size", 1, 30, 5, label_visibility="collapsed")

    st.markdown("### ⚙️ Tools")
    eraser_mode = st.toggle("🧹 Eraser Mode", value=False)
    shape_mode  = st.toggle("🔷 AI Shape Correction", value=True)

    st.markdown("---")

    st.markdown("### 🗑️ Canvas")
    clear_btn = st.button("🗑️ Clear Canvas")
    save_btn  = st.button("📥 Download Sketch")

    st.markdown("---")

    # Voice status
    if VOICE_AVAILABLE:
        st.markdown("""
        <div class="info-box">
        <b>🎤 Voice Commands:</b><br>
        Say: <b>"red"</b> → red colour<br>
        Say: <b>"blue"</b> → blue colour<br>
        Say: <b>"green"</b> → green colour<br>
        Say: <b>"yellow"</b> → yellow<br>
        Say: <b>"purple"</b> → purple<br>
        Say: <b>"clear"</b> → clear canvas<br>
        Say: <b>"eraser"</b> → toggle eraser<br>
        Say: <b>"shape"</b> → toggle shapes<br>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
        ⚠️ Voice disabled<br>
        Install: pip install SpeechRecognition pyaudio
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>✋ How to Draw:</b><br>
    Open hand = Pen DOWN ✏️<br>
    Pinch fingers = Pen UP ✋<br><br>
    <b>🔷 Shapes detected:</b><br>
    Circle, Square, Triangle, Line!
    </div>
    """, unsafe_allow_html=True)

with col_cam:
    ctx = webrtc_streamer(
        key="gestura-v2",
        video_processor_factory=GesturaProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.colour_bgr  = COLOURS[selected_colour]
        ctx.video_processor.thickness   = thickness
        ctx.video_processor.eraser_mode = eraser_mode
        ctx.video_processor.shape_mode  = shape_mode

        if clear_btn:
            ctx.video_processor.clear_flag = True
            st.success("✅ Canvas Cleared!")

        if save_btn and ctx.video_processor.canvas is not None:
            canvas = ctx.video_processor.canvas
            if canvas.sum() > 0:
                img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                img.save("gestura_sketch.png")
                with open("gestura_sketch.png", "rb") as f:
                    st.download_button(
                        label="⬇️ Click to Download Sketch",
                        data=f,
                        file_name="gestura_sketch.png",
                        mime="image/png"
                    )
            else:
                st.warning("⚠️ Canvas is empty! Draw something first!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    ✍️ <b>Gestura v2.0</b> | Built with ❤️ using Python<br>
    👩‍💻 By <b>Dhivyashree V</b> | CARE College of Engineering
</div>
""", unsafe_allow_html=True)