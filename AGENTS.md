# Monkey Detection — AGENTS.md

Single-file real-time webcam app (`monyet.py`) that classifies monkey facial expressions
("Thinking", "Happy", "Shocked") via MediaPipe face/hand landmark geometry — no ML classification.

## Run

```bash
source venv/bin/activate
python monyet.py
```

## Critical — MediaPipe Tasks API (not `mp.solutions`)

MediaPipe ≥0.10.35 on Python 3.14 **removed** `mp.solutions`. Do NOT use `mp.solutions.face_mesh`, `mp.solutions.hands`, or `mp.solutions.drawing_utils`. The app uses `mediapipe.tasks.python.vision`:

- `vision.FaceLandmarker` (not FaceMesh)
- `vision.HandLandmarker` (not Hands)
- `vision.RunningMode.IMAGE` (not LIVE_STREAM)
- Inference: `detect(mp_image)` not `process(rgb)`
- Models: `face_landmarker_v2.task` (3.7 MB) + `hand_landmarker.task` (7.8 MB) at project root

If you need to rewrite the inference layer, all three (`__init__` model loading, frame processing, hand index constants) must be replaced together.

## Wayland Display Fix

Under Wayland, `cv2.imshow` crashes unless `QT_QPA_PLATFORM=xcb` is set **before** `import cv2`. Already in monyet.py. Preserve this.

## Expression Logic (geometry-only)

All thresholds are normalized 0-1 landmark distances:

| Expression | Rule | Thresholds |
|---|---|---|
| Thinking | Index or thumb tip within 0.10 of mouth center (landmark 13) | dist < 0.10 |
| Happy | MAR > 0.20 **and** finger > 0.20 from nose bridge (landmark 1) | MAR > 0.20, dist > 0.20 |
| Shocked | MAR > 0.20 + no hand detected, **or** MAR > 0.30 | MAR > 0.20 / 0.30 |

- Smoothing: 8-frame deque, require ≥2 non-Neutral matches out of last 5 to activate
- Hand landmarks: `INDEX_FINGER_TIP = 8`, `THUMB_TIP = 4`
- Face landmarks: 478-point model (FaceLandmarker v2)

## Files

| File | Purpose |
|---|---|
| `monyet.py` | Entire application (526 lines) |
| `face_landmarker_v2.task` | Face landmark model (not committed) |
| `hand_landmarker.task` | Hand landmark model (not committed) |
| `Monkey_*.jpg` | Overlay images shown on match |
| `requirements.txt` | opencv-python, opencv-contrib-python, mediapipe, numpy |
| `venv/` | Python 3.14 virtualenv |

## Git

- Single commit: `d1aa8c3` "first commit"
- monyet.py has been rewritten for Tasks API (uncommitted — do not restore the original `mp.solutions` version)
- `.task` model files + `.omo/` are untracked, `.gitignore`-friendly

## Known Issues

- `QFontDatabase: Cannot find font directory` warnings: cosmetic, Qt stopped shipping fonts. `cv2.putText` renders fine via Hershey/fontconfig.
- No tests, no CI. Manual verification: run `python monyet.py` and verify face/hand detection with webcam.
- Camera warmup requires 10 successful frames. Slow cameras may fail the warmup loop.
