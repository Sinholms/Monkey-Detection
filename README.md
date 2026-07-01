# Monkey Expression Matcher

Real-time webcam app that detects your facial expressions using a ResNet50 model trained on FER2013, then matches them with monkey images.

## How it works

1. **Face detection** — MediaPipe Face Mesh is used when available; otherwise OpenCV Haar cascade is used
2. **Expression classification** — ResNet50 predicts 7 FER2013 emotion classes, then maps them to `Shocked`, `Happy`, or `Neutral` at display time
3. **Monkey mapping** — `Happy` → `Monkey_Happy.jpg`, `Shocked` → `Monkey_Shocked.jpg`, `Neutral` → no match
4. **Thinking gesture** — enabled only when the installed MediaPipe package exposes `mp.solutions`
5. **Temporal smoothing** — 8-frame majority vote reduces flicker

Current trained checkpoint:

- `best_model.pth`: 7-class emotion model
- Validation accuracy: ~66% (7-class), ~81% when mapped to 3 expression targets
- Test accuracy: ~66% (7-class), ~81% when mapped to 3 expression targets

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Download dataset

```bash
mkdir -p data

# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Option B: Manual download from Kaggle, extract to data/fer2013.csv
```

## Train

```bash
python train.py --data data/fer2013.csv --device cuda
```

Outputs:

- `best_model.pth`
- `metrics.csv`
- `training_config.json`
- `classification_report_validation.txt`
- `classification_report_test.txt`
- `training_curves.png`
- `confusion_matrix.png`
- `confusion_matrix_test.png`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 30 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate (FC head) |
| `--data` | `data/fer2013.csv` | Path to FER2013 CSV |
|| `--label-mode` | `emotion` | `emotion` (default) trains 7 FER2013 classes; `expression` trains 3 app targets |
| `--device` | `auto` | Use `cuda` to fail fast if GPU is unavailable |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--label-smoothing` | `0.05` | CrossEntropy label smoothing |
|| `--early-stopping-patience` | `12` | Stop after validation plateau |

## Run

```bash
python main.py
# or
python monyet.py
```

Debug mode shows model probabilities and face crop box:

```bash
python monyet.py --debug
```

If predictions feel wrong, run with `--debug` to see the raw 7-class probabilities and the mapped expression decision. The confidence threshold controls how confidently the model must predict before a match is shown:
If predictions feel wrong, run with `--debug` to see the raw 7-class probabilities. Raise the threshold to reduce false positives:

```bash
python monyet.py --debug --confidence-threshold 0.55
# Or lower it to detect more expressions:
python monyet.py --debug --confidence-threshold 0.40
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset camera |
| `f` | Toggle fullscreen |
| `c` | Clear current smoothed match |
| `s` | Save a debug frame when `--debug` is enabled |

## Model details

- **Architecture**: ResNet50 (ImageNet pre-trained), layers 1–3 frozen, layer 4 + FC head fine-tuned
- **FC head**: Dropout(0.5) → Linear(2048, classes)
- **Default labels**: 7 FER2013 emotions (`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`)
  mapped to 3 expressions at inference time (Angry/Disgust/Fear/Sad/Surprise → Shocked, Happy → Happy, Neutral → no match)
- **Optimizer**: AdamW with differential LR (1e-3 head, 1e-4 layer4)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Loss**: Weighted CrossEntropyLoss with label smoothing
- **Training controls**: AMP on CUDA, metrics CSV, classification reports, confusion matrices, early stopping
## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- OpenCV >= 4.8
- MediaPipe >= 0.10
- CUDA optional (auto-fallback to CPU)

On Python 3.13 with MediaPipe 0.10.35, `mp.solutions` is not exposed. The app automatically falls back to OpenCV face detection in that case, and hand-based Thinking gesture detection is disabled.
