"""
Gestura v2.0 — AI-Powered Gestura v2.0
==========================================
4 AI Innovations added:
  1. AI Shape Correction  — rough shapes become perfect automatically
  2. Handwriting to Text  — air-written text → digital text (press T)
  3. Voice Commands       — speak to control colour / clear / save / eraser
  4. Emotion Detection    — your face mood changes the UI accent colour

Controls
--------
Pinch (thumb+index < 40px) : Pen UP   (stop drawing)
Open hand                  : Pen DOWN (draw)
Press C   : Clear canvas
Press S   : Save sketch as gestura_sketch.png
Press E   : Toggle eraser
Press T   : OCR — read handwriting from canvas → text
Press X   : Toggle AI shape correction ON/OFF
Press 1-5 : Switch colour (1=White 2=Red 3=Green 4=Blue 5=Yellow)
Press +/= : Increase stroke thickness
Press -   : Decrease stroke thickness
Press Q   : Quit

Voice Commands (speak clearly while app is running)
---------------------------------------------------
"red" / "blue" / "green" / "white" / "yellow"
"clear"  → clears the canvas
"save"   → saves the sketch
"eraser" → toggles eraser mode
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading

# ── Optional imports with graceful fallback ──────────────────────────────────
print("[WARN] easyocr not installed — Handwriting to Text disabled.")
print("       Run: pip install easyocr")

try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
    print("[INFO] SpeechRecognition loaded — Voice Commands ready!")
except ImportError:
    VOICE_AVAILABLE = False
    print("[WARN] speech_recognition not installed — Voice Commands disabled.")
    print("       Run: pip install SpeechRecognition pyaudio")

try:
    from fer import FER
    EMOTION_AVAILABLE = True
    print("[INFO] FER loaded — Emotion Detection ready!")
except ImportError:
    EMOTION_AVAILABLE = False
    print("[WARN] fer not installed — Emotion Detection disabled.")
    print("       Run: pip install fer")

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Colour palette (BGR) ────────────────────────────────────────────────────
COLOURS = {
    "1": ("White",  (255, 255, 255)),
    "2": ("Red",    (0,   0,   255)),
    "3": ("Green",  (0,   255, 0  )),
    "4": ("Blue",   (255, 0,   0  )),
    "5": ("Yellow", (0,   255, 255)),
}

# ── Emotion → UI accent colour (BGR) ────────────────────────────────────────
EMOTION_COLOURS = {
    "happy":    (0,   255, 255),
    "sad":      (255, 100, 0  ),
    "angry":    (0,   0,   255),
    "surprise": (0,   165, 255),
    "fear":     (128, 0,   128),
    "disgust":  (0,   128, 0  ),
    "neutral":  (200, 200, 200),
}

# ── Parameters ───────────────────────────────────────────────────────────────
PINCH_THRESHOLD      = 40
MIN_THICKNESS        = 1
MAX_THICKNESS        = 30
ERASER_THICKNESS     = 40
SAVE_PATH            = "gestura_sketch.png"
EMOTION_CHECK_EVERY  = 30   # check emotion every N frames


# ════════════════════════════════════════════════════════════════════════════
# INNOVATION 1 — AI Shape Correction
# ════════════════════════════════════════════════════════════════════════════
def correct_shape(canvas, stroke_points, colour_bgr, thickness):
    """
    When user lifts pen, analyse the stroke and redraw as a perfect shape.
    Detects: line, triangle, rectangle, circle/ellipse.
    """
    if len(stroke_points) < 5:
        return

    pts  = np.array(stroke_points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    n = len(approx)

    x, y, w, h = cv2.boundingRect(pts)

    # Erase the rough original stroke
    for i in range(1, len(stroke_points)):
        cv2.line(canvas, stroke_points[i-1], stroke_points[i],
                 (0, 0, 0), thickness + 12)

    if n <= 2 or (w < 25 or h < 25):
        # ── LINE ─────────────────────────────────────────────────────────
        cv2.line(canvas, tuple(pts[0]), tuple(pts[-1]), colour_bgr, thickness)

    elif n == 3:
        # ── TRIANGLE ─────────────────────────────────────────────────────
        cv2.polylines(canvas, [approx], True, colour_bgr, thickness)

    elif n == 4:
        # ── RECTANGLE ────────────────────────────────────────────────────
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour_bgr, thickness)

    else:
        # ── CIRCLE ───────────────────────────────────────────────────────
        center = (x + w // 2, y + h // 2)
        radius = (w + h) // 4
        cv2.circle(canvas, center, radius, colour_bgr, thickness)


# ════════════════════════════════════════════════════════════════════════════
# INNOVATION 2 — Handwriting to Text (EasyOCR)
# ════════════════════════════════════════════════════════════════════════════
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None and OCR_AVAILABLE:
        print("[INFO] Loading EasyOCR model (first run ~30s)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

def canvas_to_text(canvas):
    """Run OCR on current canvas and return recognised text string."""
    if not OCR_AVAILABLE:
        return "Install easyocr to use this feature"
    reader = get_ocr_reader()
    if reader is None:
        return "OCR not ready"
    results = reader.readtext(canvas)
    if not results:
        return "No text detected on canvas"
    return "  |  ".join([r[1] for r in results])


# ════════════════════════════════════════════════════════════════════════════
# INNOVATION 3 — Voice Commands (SpeechRecognition)
# ════════════════════════════════════════════════════════════════════════════
_voice_command = ""
_voice_lock    = threading.Lock()
_voice_active  = False

def _voice_listener_thread():
    global _voice_command, _voice_active
    if not VOICE_AVAILABLE:
        return
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            text  = recognizer.recognize_google(audio).lower()
            with _voice_lock:
                _voice_command = text
    except Exception:
        pass
    _voice_active = False

def start_voice_listen():
    global _voice_active
    if VOICE_AVAILABLE and not _voice_active:
        _voice_active = True
        t = threading.Thread(target=_voice_listener_thread, daemon=True)
        t.start()

def get_voice_command():
    global _voice_command
    with _voice_lock:
        cmd = _voice_command
        _voice_command = ""
    return cmd


# ════════════════════════════════════════════════════════════════════════════
# INNOVATION 4 — Emotion Detection (FER)
# ════════════════════════════════════════════════════════════════════════════
_emotion_detector   = None
_last_emotion       = "neutral"
_emotion_counter    = 0

def get_emotion_detector():
    global _emotion_detector
    if _emotion_detector is None and EMOTION_AVAILABLE:
        _emotion_detector = FER(mtcnn=False)
    return _emotion_detector

def detect_emotion(frame):
    """Return dominant emotion string detected from face in frame."""
    global _last_emotion, _emotion_counter
    _emotion_counter += 1
    if _emotion_counter % EMOTION_CHECK_EVERY != 0:
        return _last_emotion
    detector = get_emotion_detector()
    if detector is None:
        return "neutral"
    try:
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            _last_emotion = max(emotions, key=emotions.get)
    except Exception:
        pass
    return _last_emotion


# ════════════════════════════════════════════════════════════════════════════
# HUD Overlay
# ════════════════════════════════════════════════════════════════════════════
def draw_ui(frame, canvas, drawing, colour_name, colour_bgr,
            thickness, eraser_mode, shape_mode, fps,
            ocr_text, emotion, voice_status):

    h, w = frame.shape[:2]

    # Blend canvas onto camera feed
    mask   = canvas.sum(axis=2) > 0
    output = frame.copy()
    output[mask] = cv2.addWeighted(frame, 0.15, canvas, 0.85, 0)[mask]

    # Accent colour driven by emotion (or current draw colour if unavailable)
    accent = EMOTION_COLOURS.get(emotion, colour_bgr) if EMOTION_AVAILABLE else colour_bgr

    # ── Top status bar ───────────────────────────────────────────────────
    cv2.rectangle(output, (0, 0), (w, 36), (20, 20, 20), -1)
    mode_str  = "ERASER" if eraser_mode else f"Colour:{colour_name}"
    shape_str = "[Shape ON]" if shape_mode else "[Shape OFF]"
    pen_str   = "Pen:DOWN" if drawing else "Pen:UP"
    mood_str  = f"Mood:{emotion.capitalize()}" if EMOTION_AVAILABLE else ""
    bar_text  = f"  {mode_str}  {shape_str}  |  {pen_str}  |  FPS:{fps:.0f}  {mood_str}"
    cv2.putText(output, bar_text, (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, accent, 1, cv2.LINE_AA)

    # Colour swatches
    for i, (_, bgr) in enumerate(COLOURS.values()):
        cx = w - 140 + i * 26
        cv2.rectangle(output, (cx, 6), (cx+20, 30), bgr, -1)

    # ── OCR result bar ───────────────────────────────────────────────────
    if ocr_text:
        cv2.rectangle(output, (0, 36), (w, 62), (25, 25, 25), -1)
        cv2.putText(output, f"TEXT: {ocr_text[:90]}", (6, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1, cv2.LINE_AA)

    # ── Voice status bar ─────────────────────────────────────────────────
    if voice_status:
        cv2.putText(output, voice_status, (6, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1, cv2.LINE_AA)

    # ── Bottom legend ────────────────────────────────────────────────────
    legend = "[C]lear [S]ave [E]raser [T]ext [X]Shape [1-5]Colour [+/-]Size [Q]uit"
    cv2.putText(output, legend, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    return output


def get_landmark_px(lm, fw, fh):
    return int(lm.x * fw), int(lm.y * fh)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from webcam.")

    frame_h, frame_w = frame.shape[:2]
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # ── State variables ──────────────────────────────────────────────────
    prev_x, prev_y = None, None
    drawing        = False
    eraser_mode    = False
    shape_mode     = True    # AI shape correction enabled by default
    colour_name    = "White"
    colour_bgr     = (255, 255, 255)
    thickness      = 5
    fps            = 0.0
    prev_time      = time.time()
    current_stroke = []      # collects points for current stroke
    ocr_text       = ""
    voice_status   = ""
    voice_timer    = 0
    emotion        = "neutral"

    # Start voice listener
    if VOICE_AVAILABLE:
        start_voice_listen()
        voice_status = "Mic: Listening..."

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    print("\nGestura v2.0 started!")
    print("Innovations active:")
    print(f"  Shape Correction : ON")
    print(f"  Handwriting→Text : {'ON' if OCR_AVAILABLE else 'OFF (install easyocr)'}")
    print(f"  Voice Commands   : {'ON' if VOICE_AVAILABLE else 'OFF (install SpeechRecognition)'}")
    print(f"  Emotion Detection: {'ON' if EMOTION_AVAILABLE else 'OFF (install fer)'}")
    print("\nPress Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame      = cv2.flip(frame, 1)
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        was_drawing = drawing

        # Emotion detection (every 30 frames)
        if EMOTION_AVAILABLE:
            emotion = detect_emotion(frame)

        results = hands.process(rgb)
        cur_x, cur_y = None, None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            index_tip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_lm.landmark[mp_hands.HandLandmark.THUMB_TIP]

            cur_x, cur_y = get_landmark_px(index_tip, frame_w, frame_h)
            tx, ty       = get_landmark_px(thumb_tip,  frame_w, frame_h)

            pinch_dist = np.hypot(cur_x - tx, cur_y - ty)
            drawing    = (pinch_dist >= PINCH_THRESHOLD)

            dot_col = colour_bgr if (drawing and not eraser_mode) else (100, 100, 100)
            cv2.circle(frame, (cur_x, cur_y), 8, dot_col, -1)

            if drawing and prev_x is not None:
                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (cur_x, cur_y),
                             (0, 0, 0), ERASER_THICKNESS)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (cur_x, cur_y),
                             colour_bgr, thickness)
                    if shape_mode:
                        current_stroke.append((cur_x, cur_y))

        # ── Pen just lifted → run shape correction ───────────────────────
        if was_drawing and not drawing:
            if shape_mode and len(current_stroke) > 5 and not eraser_mode:
                correct_shape(canvas, current_stroke, colour_bgr, thickness)
            current_stroke = []

        prev_x, prev_y = (cur_x, cur_y) if drawing else (None, None)

        # ── Voice command processing ─────────────────────────────────────
        if VOICE_AVAILABLE:
            cmd = get_voice_command()
            if cmd:
                voice_status = f"Heard: '{cmd}'"
                voice_timer  = 90
                if   "red"    in cmd: colour_name, colour_bgr = COLOURS["2"]; eraser_mode = False
                elif "green"  in cmd: colour_name, colour_bgr = COLOURS["3"]; eraser_mode = False
                elif "blue"   in cmd: colour_name, colour_bgr = COLOURS["4"]; eraser_mode = False
                elif "white"  in cmd: colour_name, colour_bgr = COLOURS["1"]; eraser_mode = False
                elif "yellow" in cmd: colour_name, colour_bgr = COLOURS["5"]; eraser_mode = False
                elif "clear"  in cmd: canvas[:] = 0; ocr_text = ""
                elif "save"   in cmd: cv2.imwrite(SAVE_PATH, canvas); print("Saved!")
                elif "eraser" in cmd: eraser_mode = not eraser_mode
                start_voice_listen()

            if voice_timer > 0:
                voice_timer -= 1
            else:
                voice_status = "Mic: Listening..."
                start_voice_listen()

        # FPS
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Compose and show frame
        output = draw_ui(frame, canvas, drawing, colour_name, colour_bgr,
                         thickness, eraser_mode, shape_mode, fps,
                         ocr_text, emotion, voice_status)

        cv2.imshow("Gestura v2.0", output)

        # ── Keyboard input ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if   key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
            ocr_text  = ""
            print("Canvas cleared.")
        elif key == ord("s"):
            cv2.imwrite(SAVE_PATH, canvas)
            print(f"Saved → {os.path.abspath(SAVE_PATH)}")
        elif key == ord("e"):
            eraser_mode = not eraser_mode
            print(f"Eraser: {'ON' if eraser_mode else 'OFF'}")
        elif key == ord("t"):
            print("[INFO] Running OCR on canvas...")
            ocr_text = canvas_to_text(canvas)
            print(f"[OCR Result] {ocr_text}")
        elif key == ord("x"):
            shape_mode = not shape_mode
            print(f"AI Shape Correction: {'ON' if shape_mode else 'OFF'}")
        elif chr(key) in COLOURS:
            colour_name, colour_bgr = COLOURS[chr(key)]
            eraser_mode = False
            print(f"Colour → {colour_name}")
        elif key in (ord("+"), ord("=")):
            thickness = min(thickness + 2, MAX_THICKNESS)
        elif key == ord("-"):
            thickness = max(thickness - 2, MIN_THICKNESS)

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Gestura v2.0 closed.")


if __name__ == "__main__":
    main()