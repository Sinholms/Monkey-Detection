# Monkey Expression Detection

Real-time webcam app that classifies monkey facial expressions using MediaPipe face and hand landmark geometry — no ML classification, purely geometric heuristics.

## Expressions

| Expression | Trigger |
|---|---|
| Thinking | Finger tip near mouth (< 0.10 normalized distance) |
| Happy | Mouth open (MAR > 0.20) + finger away from nose bridge |
| Shocked | Mouth open + no hand detected, or extreme mouth opening (MAR > 0.30) |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download MediaPipe task models to project root:

```bash
curl -L -o face_landmarker_v2.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Run

```bash
source venv/bin/activate
python monyet.py
```

Controls: `q` quit, `r` reset camera, `f` toggle fullscreen.

## Requirements

- Python 3.14+
- MediaPipe 0.10.35+
- OpenCV (with Qt/HighGUI)
- Webcam
