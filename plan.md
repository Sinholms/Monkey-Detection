# Product Requirements Document (PRD)
# Monkey Expression Matcher — ML Edition

## 1. Overview

### 1.1 Product Name
Monkey Expression Matcher (ML Edition)

### 1.2 Product Description
Aplikasi real-time berbasis webcam yang mendeteksi ekspresi wajah pengguna menggunakan model deep learning (ResNet50) yang di-train pada dataset FER2013, kemudian mencocokkannya dengan gambar monyet berekspresi. Aplikasi ini juga menggabungkan deteksi hand gesture (MediaPipe) untuk ekspresi "Thinking".

### 1.3 Problem Statement
Versi sebelumnya menggunakan pendekatan rule-based dengan MediaPipe landmarks (mouth aspect ratio, finger proximity). Pendekatan ini memiliki keterbatasan:
- Heuristik sederhana tidak robust terhadap variasi pencahayaan, sudut wajah, dan perbedaan individu
- Tidak bisa membedakan ekspresi kompleks (marah vs takut vs terkejut)
- Bergantung pada threshold manual yang tidak adaptif

### 1.4 Solution
Mengganti rule-based matching dengan model CNN (ResNet50) yang di-train pada dataset FER2013 (35,887 gambar, 7 kelas emosi), menghasilkan klasifikasi ekspresi yang lebih akurat, robust, dan generalizable.

---

## 2. Goals & Objectives

| Goal | Metric | Target |
|------|--------|--------|
| Akurasi klasifikasi ekspresi | Validation accuracy | > 55% (baseline), mendekati 73% (SOTA FER2013) |
| Real-time performance | FPS inference | >= 15 FPS (CPU), >= 25 FPS (GPU) |
| Robustness | Bekerja di berbagai kondisi | Pencahayaan varied, berbagai ethnicity, sudut wajah frontal |
| User experience | Latency prediksi | < 100ms per frame untuk inference model |

---

## 3. Architecture

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Webcam Input                          │
│                   (1280x720 @ 30fps)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              MediaPipe Face Mesh                         │
│         (Face Detection + Bounding Box)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌───────────────────┐
│  Face Crop        │   │  MediaPipe Hands  │
│  (bbox + pad 25%) │   │  (Hand Detection) │
└────────┬──────────┘   └────────┬──────────┘
         │                       │
         ▼                       ▼
┌───────────────────┐   ┌───────────────────┐
│  ResNet50 Model   │   │  Finger-to-Mouth  │
│  (FER2013 7-cls)  │   │  Distance Check   │
│  Preprocess:      │   │  (Rule-based)     │
│  - Resize 224x224 │   └────────┬──────────┘
│  - Normalize      │            │
│  - ToTensor       │            │
└────────┬──────────┘            │
         │                       │
         ▼                       │
┌───────────────────┐            │
│  Expression Map   │            │
│  7 classes → 3    │            │
│  monkey expr      │            │
└────────┬──────────┘            │
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Priority Logic:      │
         │  1. Thinking (hand)   │
         │  2. ML prediction     │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Temporal Smoothing   │
         │  (deque, 8 frames)    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Display Output       │
         │  (Monkey image + UI)  │
         └───────────────────────┘
```

### 3.2 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | >= 3.9 |
| ML Framework | PyTorch | >= 2.0 |
| Computer Vision | torchvision | >= 0.15 |
| Face Detection | MediaPipe Face Mesh | >= 0.10 |
| Hand Detection | MediaPipe Hands | >= 0.10 |
| Image Processing | OpenCV | >= 4.8 |
| Image Handling | Pillow | >= 10.0 |
| Data Processing | pandas, numpy | Latest |
| Metrics | scikit-learn | >= 1.3 |
| Visualization | matplotlib | >= 3.7 |

### 3.3 Model Architecture

```
ResNet50 (pre-trained ImageNet)
├── Conv layers (frozen: conv1 → layer3)
├── layer4 (fine-tuned)
└── FC Head (replaced):
    ├── Dropout(0.5)
    └── Linear(2048 → 7)
```

---

## 4. Data

### 4.1 Dataset: FER2013

| Property | Value |
|----------|-------|
| Source | Kaggle (msambare/fer2013) |
| Total images | 35,887 |
| Image format | Grayscale, 48x48 pixels |
| Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| Train split | ~28,709 images |
| Val split (PublicTest) | ~3,589 images |
| Test split (PrivateTest) | ~3,589 images |
| Known issues | Class imbalance (Disgust & Fear underrepresented) |

### 4.2 Class Mapping (FER2013 → Monkey Expression)

| FER2013 Label | Emotion | Monkey Expression | Monkey Image |
|---------------|---------|-------------------|--------------|
| 0 | Angry | Shocked | `Monkey_Shocked.jpg` |
| 1 | Disgust | Shocked | `Monkey_Shocked.jpg` |
| 2 | Fear | Shocked | `Monkey_Shocked.jpg` |
| 3 | Happy | Happy | `Monkey_Happy.jpg` |
| 4 | Sad | Shocked | `Monkey_Shocked.jpg` |
| 5 | Surprise | Shocked | `Monkey_Shocked.jpg` |
| 6 | Neutral | None (no match) | — |
| — | — | Thinking (gesture) | `Monkey_Thinking.jpg` |

> Catatan: `Monkey_Scare.jpg` tersedia di folder namun tidak digunakan pada versi ini untuk kesederhanaan.

### 4.3 Data Preprocessing

| Step | Training | Validation/Test |
|------|----------|-----------------|
| Grayscale → RGB | Repeat channel 3x | Repeat channel 3x |
| Resize | 224 x 224 | 224 x 224 |
| Horizontal Flip | Random (50%) | No |
| Rotation | Random ±15° | No |
| Color Jitter | Brightness ±0.3, Contrast ±0.3 | No |
| Affine | Translate ±10% | No |
| Normalize | ImageNet mean/std | ImageNet mean/std |

---

## 5. Functional Requirements

### 5.1 Training (`train.py`)

| ID | Requirement | Priority |
|----|-------------|----------|
| TR-01 | Load FER2013 dari CSV file | Must |
| TR-02 | Split data berdasarkan kolom Usage (Training/PublicTest/PrivateTest) | Must |
| TR-03 | Apply augmentasi pada training set | Must |
| TR-04 | Compute class weights untuk menangani imbalance | Must |
| TR-05 | Train ResNet50 dengan differential learning rates | Must |
| TR-06 | ReduceLROnPlateau scheduler (patience=3, factor=0.5) | Must |
| TR-07 | Save best model berdasarkan validation accuracy | Must |
| TR-08 | Plot training curves (loss & accuracy) | Should |
| TR-09 | Plot confusion matrix pada validation set | Should |
| TR-10 | Evaluate pada test set (PrivateTest) di akhir training | Should |
| TR-11 | Support CLI arguments (epochs, batch-size, lr, data path) | Must |

### 5.2 Inference (`monyet.py`)

| ID | Requirement | Priority |
|----|-------------|----------|
| IN-01 | Load trained model dari checkpoint (.pth) | Must |
| IN-02 | Auto-detect device (CUDA / CPU) | Must |
| IN-03 | Face detection via MediaPipe Face Mesh | Must |
| IN-04 | Face cropping dari landmarks bounding box + 25% padding | Must |
| IN-05 | ResNet50 inference pada face crop | Must |
| IN-06 | Hand detection via MediaPipe Hands | Must |
| IN-07 | Finger-to-mouth distance check untuk "Thinking" gesture | Must |
| IN-08 | Priority logic: Thinking (gesture) > ML prediction | Must |
| IN-09 | Temporal smoothing (8-frame deque, majority vote) | Must |
| IN-10 | Display monkey image overlay saat match terdeteksi | Must |
| IN-11 | Display confidence score dan emotion label | Must |
| IN-12 | FPS counter | Must |
| IN-13 | Face/Hand count status bar | Should |
| IN-14 | Keyboard controls: q (quit), r (reset camera), f (fullscreen) | Must |
| IN-15 | Camera auto-reconnect jika frame capture gagal | Should |
| IN-16 | Platform-aware camera init (Windows DSHOW / Linux V4L2) | Must |

### 5.3 Expression Logic

```
Priority 1: finger_near_mouth == True  → "Thinking" (confidence: 0.95)
Priority 2: ResNet50 predicts class 3  → "Happy"
Priority 3: ResNet50 predicts class 0/1/2/4/5 → "Shocked"
Priority 4: ResNet50 predicts class 6  → None (neutral, no display)
```

Temporal smoothing:
- History buffer: 8 frames
- Match ditampilkan jika >= 2 frame dalam history memiliki prediksi yang sama
- Match di-clear jika >= 4 frame consecutive = Neutral

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target (CPU) | Target (GPU) |
|--------|-------------|-------------|
| Inference time per frame | < 100ms | < 40ms |
| FPS (end-to-end) | >= 15 | >= 25 |
| Model size on disk | ~98MB | ~98MB |
| RAM usage | < 2GB | < 3GB (VRAM) |
| Camera init time | < 5 seconds | < 5 seconds |

### 6.2 Compatibility

| Platform | Support |
|----------|---------|
| Linux | Full |
| Windows | Full (DSHOW camera backend) |
| macOS | Partial (untested camera backend) |
| Python | 3.9+ |
| CUDA | Optional (auto-fallback ke CPU) |

### 6.3 Reliability

- Graceful fallback jika model file tidak ditemukan (error message + exit)
- Camera reconnection logic jika stream terputus
- Placeholder image jika monkey asset tidak ditemukan
- KeyboardInterrupt handling untuk clean shutdown

---

## 7. File Structure

```
Monkey-Detection/
├── monyet.py              # Main inference application
├── train.py               # Training script
├── model.py               # ResNet50 model definition + loader
├── dataset.py             # FER2013 Dataset class + transforms + mappings
├── requirements.txt       # Python dependencies
├── best_model.pth         # Trained model checkpoint (generated)
├── training_curves.png    # Training metrics plot (generated)
├── confusion_matrix.png   # Validation confusion matrix (generated)
├── Monkey_Happy.jpg       # Happy monkey asset
├── Monkey_Shocked.jpg     # Shocked monkey asset
├── Monkey_Thinking.jpg    # Thinking monkey asset
├── Monkey_Scare.jpg       # Unused asset (reserved for future)
├── data/
│   └── fer2013.csv        # FER2013 dataset (manual download)
└── plan.md                # This document
```

---

## 8. Usage Guide

### 8.1 Setup

```bash
# Clone repository
cd Monkey-Detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Download Dataset

```bash
mkdir -p data

# Option A: Kaggle CLI
pip install kaggle
# Place ~/.kaggle/kaggle.json (get from kaggle.com/settings)
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Option B: Manual download from Kaggle → extract to data/fer2013.csv
```

### 8.3 Train

```bash
# Default: 25 epochs, batch_size 64, lr 1e-3
python train.py --data data/fer2013.csv

# Custom hyperparameters
python train.py --data data/fer2013.csv --epochs 30 --batch-size 128 --lr 0.001

# Outputs:
#   best_model.pth       — best model checkpoint
#   training_curves.png  — loss & accuracy plots
#   confusion_matrix.png — validation confusion matrix
```

### 8.4 Run

```bash
python monyet.py
# Controls: q=quit, r=reset camera, f=fullscreen
```

---

## 9. Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Epochs | 25 | Balance antara convergence dan overfitting |
| Batch size | 64 | Stable gradient, fits in GPU memory |
| Learning rate (FC head) | 1e-3 | Higher for new random-init head |
| Learning rate (layer4) | 1e-4 | Lower for fine-tuning pre-trained |
| Optimizer | Adam | Adaptive, good default for transfer learning |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) | Decay on plateau |
| Loss | CrossEntropyLoss (weighted) | Handle class imbalance |
| Dropout | 0.5 | Regularization on FC head |
| Frozen layers | conv1 → layer3 | Preserve ImageNet features |
| Fine-tuned layers | layer4 + FC head | Adapt to facial expressions |

---

## 10. Known Limitations & Future Work

### 10.1 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| FER2013 low resolution (48x48) | Model kurang detail halus | Resize + augmentasi |
| Class imbalance | Disgust/Fear accuracy rendah | Weighted CrossEntropyLoss |
| ResNet50 berat di CPU | FPS < 15 di CPU lama | Bisa swap ke MobileNetV2 |
| Hanya support 1 face | Multi-person tidak didukung | MediaPipe limitation |
| Thinking hanya dari gesture | Tidak ada facial "thinking" class | By design (rule-based) |
| Monkey_Scare.jpg tidak terpakai | Asset idle | Future: 4-class mapping |

### 10.2 Future Enhancements

1. **4-class mapping**: Remap FER2013 classes → Happy, Shocked, Scared, Neutral (gunakan `Monkey_Scare.jpg`)
2. **MobileNetV2 option**: Model lebih ringan (~14MB) untuk CPU-only deployment
3. **Full fine-tuning**: Unfreeze semua layers setelah initial training untuk akurasi lebih tinggi
4. **ONNX export**: Export model ke ONNX untuk inference lebih cepat via ONNXRuntime
5. **Real-time training**: Data collection mode — collect user face images untuk fine-tune personal model
6. **Multi-face support**: Deteksi dan klasifikasi multiple faces simultaneously
7. **Confidence-based fallback**: Jika ML confidence < threshold, tampilkan "uncertain" alih-alih salah match
8. **Web UI**: Streamlit/Gradio interface sebagai alternatif dari OpenCV window

---

## 11. Success Criteria

| Criteria | How to Verify |
|----------|---------------|
| Model training converges | Val accuracy > 55%, loss decreases consistently |
| Real-time inference | FPS counter menunjukkan >= 15 di terminal |
| Correct expression mapping | Senyum → Happy, Terkejut → Shocked, Jari di mulut → Thinking |
| Stable runtime | Tidak crash dalam 10 menit continuous use |
| Camera recovery | Auto-reconnect saat USB camera dicabut-colok |
