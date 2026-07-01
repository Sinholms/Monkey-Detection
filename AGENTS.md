# Monkey Expression Matcher — Agent Guide

## Entry points

- `main.py` — thin wrapper around `runpy.run_module("monyet")`. Not where the logic lives.
- `monyet.py` — the real inference app (MonkeyExpressionMatcher class). Loads model, accesses camera, runs main loop.
- `train.py` — training script. Standalone, no shared CLI with monyet.py.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/                     # produces data/fer2013.csv

# Train with default FER2013 labels (7-class, ~66%)
python train.py --data data/fer2013.csv --device cuda

# Train with FERPlus cleaner labels (7-class, ~75%)
python train.py --data data/fer2013.csv --device cuda --ferplus data/fer2013new.csv

# Train with 3-class expression labels (Shocked, Happy, Neutral)
python train.py --data data/fer2013.csv --device cuda --label-mode expression

# Run inference
python monyet.py                               # or: python main.py
python monyet.py --debug                       # show probs + face bbox
python monyet.py --debug --confidence-threshold 0.50  # default threshold
```

**Key defaults**: `confidence_threshold=0.50` (argparse default). Lower to 0.30 for more matches (more FP risk).

## Key conventions & quirks

### Lazy imports in monyet.py
`monyet.py` defers all imports inside `load_runtime_dependencies()`. This is intentional — cv2/mediapipe/torch are loaded at runtime, not at module level. If modifying monyet.py, always call `load_runtime_dependencies()` before using globals like `cv2`, `np`, `torch`.

### Label modes (train.py)
Two modes controlled by `--label-mode`:
- `emotion` (**default**, 7-class): FER2013 labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). Mapped to Shocked/Happy/None at inference via `EXPRESSION_MAP`.
- `expression` (3-class): Shocked(0), Happy(1), Neutral(2). Trains model on pre-mapped labels.

**Threshold behavior**: In `emotion` mode (default), only `--confidence-threshold` applies (default 0.50). Per-class thresholds (`--shocked-threshold`, `--happy-threshold`) only apply in `expression` mode.

### Inference pipeline
1. MediaPipe Face Mesh (or OpenCV Haar cascade fallback) for face detection
2. ResNet50 predicts expression from face crop
3. **Priority**: Thinking (hand gesture) > ML prediction
4. Temporal smoothing: 8-frame deque majority vote (≥2 matching frames to display, 4 consecutive neutral to clear)
5. Monkey image overlaid on top-right of frame

### MediaPipe on Python 3.13
On Python 3.13+ with MediaPipe 0.10.35, `mp.solutions` is **not exposed**. The app falls back to OpenCV Haar cascade (`haarcascade_frontalface_default.xml`). Thinking gesture detection is disabled. This is silent — check `face_backend` attribute.

### Camera init (Linux v4l2-ctl wake-up)
`initialize_camera()` tries indices 0..2, picks the first working camera. Platform-aware: `cv2.CAP_DSHOW` (Windows), `CAP_ANY` (macOS), `CAP_ANY` (Linux with v4l2-ctl wake).

**Linux camera sleep bug**: Some integrated cameras (Chicony, etc.) enter a low-power state during the ~5s CUDA model init, causing `VIDIOC_REQBUFS` to fail with `errno=19 (No such device)`. The fix in `_open_camera()` runs `v4l2-ctl -d /dev/videoN --set-fmt-video=width=640,height=480,pixelformat=MJPG --stream-mmap --stream-count=1 --stream-to=/dev/null` to stream one frame before OpenCV opens the device. This wakes the sensor.

Resolution defaults to 1280x720; init uses 640x480, drains 10 frames, then attempts target resolution.

### Model / checkpoint format
Saved by `train.py` as a dict: `state_dict`, `num_classes`, `class_names`, `label_mode`, `image_size`, `epoch`, `train_acc`, `val_acc`, `test_acc`, `test_loss`, `config`. `model.py`'s `load_checkpoint()` uses `weights_only=True`.

### Training outputs
`train.py` produces: `best_model.pth`, `metrics.csv`, `training_config.json`, `classification_report_validation.txt`, `classification_report_test.txt`, `training_curves.png`, `confusion_matrix.png`, `confusion_matrix_test.png`.

### Headless matplotlib
`train.py` sets `os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"` and `matplotlib.use("Agg")` before any plotting import. Do not remove.

### Architecture
- ResNet50 (ImageNet pre-trained), layers 1–3 frozen, layer 4 + FC head fine-tuned
- FC head: Dropout(0.5) → Linear(2048, num_classes)
- Optimizer: AdamW with differential LR (1e-3 head, 1e-4 layer4)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Loss: Weighted CrossEntropyLoss with label smoothing (0.05)
- AMP on CUDA, early stopping (patience=12)

### FERPlus integration
FERPlus (Microsoft) provides cleaner labels for FER2013 images via 10 crowd-sourced annotators per image. CSV at `data/fer2013new.csv`. Pass `--ferplus data/fer2013new.csv` to `train.py` to use it. Labels are loaded by `load_ferplus_labels()` in `dataset.py` via majority-vote over 7 emotion columns. The dataset class applies FERPlus labels by masking the same Usage split column as the original FER2013 — the `mask` must align both DataFrames by row position.

### Class mappings (dataset.py)
- `EXPRESSION_MAP` (7 emotion → 3 expression): Angry/Disgust/Fear/Sad/Surprise → Shocked; Happy → Happy; Neutral → None
- `EMOTION_TO_EXPRESSION_CLASS`: integer label mapping for `convert_labels()`
- `EXPRESSION_CLASS_TO_MONKEY`: 0 → Shocked, 1 → Happy, 2 → None

### Dataset details
FER2013 CSV must have columns `emotion`, `pixels`, `Usage`. Images are 48×48 grayscale pixels as space-separated strings. The dataset class converts to RGB, resizes to 224×224, and applies ImageNet normalization. Training augmentations: random horizontal flip, ±15° rotation, brightness/contrast jitter, ±10% translation.

### No tests, no CI, no linter/formatter
This repo has no test suite, no CI configuration, and no linter/formatter config. Do not assume pytest, ruff, black, mypy, or any standard Python tooling is set up. `requirements.txt` lists runtime deps only.

### Pre-trained checkpoint
`best_model.pth` is a 7-class emotion model trained with **FERPlus labels** (75.37% test accuracy). `best_model_7class_backup.pth` is the old FER2013 model (66.3%) — obsolete.

### Single commit history
Only one commit exists (`d1aa8c3`). No branches.
