# GesturaV2.0
AI-powered air drawing pad using hand gestures
Draw in the air.write with your hand and also control with your voice.
GesturaV2.0 is a real-time, AI-powered air drawing application that lets you draw on a digital canvas using only your hand gestures — no mouse, no stylus, no touch screen required.
Built with Python, OpenCV, and MediaPipe, it combines 4 AI innovations: shape correction, handwriting-to-text conversion, voice control, and emotion-based UI theming.

# Features

| Feature | Description |
|---|---|
| Air Drawing | Draw using hand gestures detected via webcam |
| AI Shape Correction | Rough shapes auto-correct to perfect geometry |
| Handwriting to Text | Air-written text converted to digital text using OCR |
| Voice Commands | Control the app hands-free using speech |
| Emotion Detection | UI accent colour changes based on your facial expression |

---

# Tech Stack

- **Python** — Core language
- **OpenCV** — Real-time video processing and canvas rendering
- **MediaPipe** — Hand landmark detection
- **EasyOCR** — Handwriting recognition (air-written → digital text)
- **SpeechRecognition** — Voice command processing
- **Streamlit** — Web-based UI interface

---

# Controls

| Action | Control |
|---|---|
| ✏️ Draw | Open hand |
| ⏸️ Stop Drawing | Pinch (thumb + index < 40px) |
| 🗑️ Clear Canvas | Press `C` |
| 💾 Save Sketch | Press `S` |
| 🧹 Toggle Eraser | Press `E` |
| 📝 OCR (Handwriting → Text) | Press `T` |
| 🔷 Toggle Shape Correction | Press `X` |
| 🎨 Switch Colour | Press `1–5` |
| ➕ Increase Thickness | Press `+` / `=` |
| ➖ Decrease Thickness | Press `-` |
| ❌ Quit | Press `Q` |

**Voice Commands** *(speak clearly while app is running)*
`"red"` / `"blue"` / `"green"` / `"white"` / `"yellow"` / `"clear"` / `"save"` / `"eraser"`

---

# Installation

1. **Clone the repository**
```bash
git clone https://github.com/shreevdhivya08-cloud/GesturaV2.0.git
cd GesturaV2.0
```

2. **Install dependencies**
```bash
pip install opencv-python mediapipe easyocr speechrecognition streamlit
```

3. **Run the app**
```bash
python air_drawing_pad.py
```

> ⚠️ Make sure your webcam and microphone are connected and accessible.

---

# Project Structure

```
GesturaV2.0/
├── air_drawing_pad.py       # Main application file
├── gestura_app.py           # Streamlit UI module
├── gesturawebpageimg.jpeg   # Preview image
├── Gesturamodelvideo.mp4    # Demo video
├── README.md
└── LICENSE
```

---

# Developer

**Dhivyashree V** — Student, 1st year, Department of Artificial Intelligence & Data Science,
CARE College of Engineering  

---

# License

This project is licensed under the [MIT License](LICENSE).
