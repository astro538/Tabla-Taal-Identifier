# 🥁 Tabla Taal & Bol Identifier

> Deep learning system for identifying **Tabla Taals** (rhythmic patterns) and **Bols** (strokes) from audio recordings, with a production-ready FastAPI backend and web demo.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training Pipeline (Deep Dive)](#training-pipeline-deep-dive)
  - [Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [Data Augmentation Strategy](#2-data-augmentation-strategy)
  - [Feature Extraction](#3-feature-extraction)
  - [Model Architectures](#4-model-architectures)
  - [Training Engine](#5-training-engine)
  - [Validation & Cross-Validation](#6-validation--cross-validation)
  - [Ensemble Method](#7-ensemble-method)
- [Inference Pipeline](#inference-pipeline)
  - [Single Prediction](#single-bol--taal-prediction)
  - [Combined Approach (Sequence)](#combined-approach-sequence-prediction)
- [API Reference](#api-reference)
- [Web Demo](#web-demo)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project builds an end-to-end deep learning pipeline to classify Indian Tabla music into:

- **8 Bols** (individual strokes): Dha, Dhin, Ga, Ka, T, Ta, Tin, Tit
- **8 Taals** (rhythmic patterns): Addha Trital, Bhajani, Dadra, Deepchandi, Ektal, Jhaptal, Rupak, Trital

The system uses **4 custom neural network architectures** trained with **5-fold stratified cross-validation**, extensive data augmentation, and modern training techniques. A production **FastAPI** backend serves predictions with a **combined approach** that merges template-based Bol mapping with model classification for frame-by-frame Bol detection in full Taal recordings.

---

## Key Results

| Task | Best Model | Accuracy | F1 Score | Comparison vs Tarang Report |
|------|-----------|:--------:|:--------:|:---------------------------:|
| **Bol** (Stroke) | Ensemble (4 models) | **99.38%** | **99.37%** | +8.38% (vs 91%) |
| **Taal** (Rhythm) | CRNN | **100.00%** | **100.00%** | +3.00% (vs 97%) |

### Per-Model Bol Accuracy
| ResidualCNN | CRNN | AttentionCRNN | ASTLite | Ensemble |
|:-----------:|:----:|:------------:|:-------:|:--------:|
| 97.88% | 97.13% | 96.63% | 90.88% | **99.38%** |

### Per-Model Taal Accuracy
| ResidualCNN | CRNN | AttentionCRNN | ASTLite | Ensemble |
|:-----------:|:----:|:------------:|:-------:|:--------:|
| 99.88% | **100.00%** | **100.00%** | 98.75% | **100.00%** |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
│                                                                  │
│  Raw Audio (.wav)                                                │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐    ┌──────────────────┐                     │
│  │ Load & Resample │───▶│   Trim Silence   │                     │
│  │  (22050 Hz)     │    │   (top_db=25)    │                     │
│  └─────────────────┘    └──────────────────┘                     │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────┐            │
│  │          DATA AUGMENTATION (Training Only)       │            │
│  │                                                  │            │
│  │  Waveform Level (×5 copies per original):        │            │
│  │   • Gaussian Noise (70% chance, σ=0.002-0.008)   │            │
│  │   • Random Gain (60% chance, ±6 dB)              │            │
│  │   • Time Shift (50% chance, ±15%)                │            │
│  │                                                  │            │
│  │  Spectrogram Level (on-the-fly, every epoch):    │            │
│  │   • SpecAugment: 2 freq masks (20 bands)         │            │
│  │   • SpecAugment: 2 time masks (30 frames)        │            │
│  │                                                  │            │
│  │  Batch Level (every batch):                      │            │
│  │   • Mixup (α=0.3, Beta distribution)             │            │
│  └──────────────────────────────────────────────────┘            │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────┐            │
│  │           MEL SPECTROGRAM EXTRACTION             │            │
│  │                                                  │            │
│  │  n_fft=2048, hop_length=512, n_mels=128          │            │
│  │  Power → dB scale → Zero-mean, Unit-variance     │            │
│  │  Output: (1, 128, T) — like a grayscale image    │            │
│  └──────────────────────────────────────────────────┘            │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────┐            │
│  │              4 MODEL ARCHITECTURES               │            │
│  │                                                  │            │
│  │  1. ResidualCNN  — ResNet-style skip connections │            │
│  │  2. CRNN         — CNN + Bidirectional LSTM      │            │
│  │  3. AttentionCRNN— CRNN + Multi-Head Attention   │            │
│  │  4. ASTLite      — Vision Transformer for Audio  │            │
│  └──────────────────────────────────────────────────┘            │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────┐            │
│  │         5-FOLD CROSS-VALIDATION                  │            │
│  │                                                  │            │
│  │  Stratified split → Train on 80%, Validate 20%   │            │
│  │  × 5 folds = every sample validated exactly once │            │
│  │  Best fold checkpoint saved per model            │            │
│  └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                         │
│                                                                  │
│  Audio File Upload                                               │
│       │                                                          │
│       ├──────── Taal Detection (CRNN model)                      │
│       │              │                                           │
│       │              ▼                                           │
│       │         Template Lookup                                  │
│       │         (Dadra → Dha Dhin Na Dha Tin Na)                 │
│       │                                                          │
│       ├──────── Onset Detection (librosa)                        │
│       │              │                                           │
│       │              ▼                                           │
│       │         Segment Extraction                               │
│       │              │                                           │
│       │              ▼                                           │
│       │         Per-Segment Bol Classification                   │
│       │              (ResidualCNN model)                         │
│       │                                                          │
│       └──────── Cross-Validation                                 │
│                  Template Bol vs Model Bol                       │
│                  → Combined confidence score                     │
│                  → Source attribution (both/template/model)      │
│                                                                  │
│  Output: Timestamped Bol sequence synced with audio playback     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
TAL-Identifier/
│
├── src/                          # Core ML source code
│   ├── config.py                 # All hyperparameters & settings
│   ├── dataset.py                # Data loading, preprocessing, augmentation
│   ├── models.py                 # 4 neural network architectures
│   ├── trainer.py                # Training loop, Mixup, early stopping, LR scheduling
│   ├── inference.py              # Production inference engine + combined prediction
│   └── utils.py                  # Visualization & results plotting
│
├── api/                          # FastAPI REST backend
│   ├── main.py                   # API endpoints, CORS, file handling
│   └── static/
│       └── index.html            # Web demo frontend
│
├── train.py                      # Training script (entry point)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── results/                      # Training outputs
│   ├── bol/
│   │   ├── models/               # Generated locally during training; not included in the tracked file list
│   │   ├── results.json          # Detailed metrics, confusion matrices
│   │   ├── results.csv           # Summary table
│   │   ├── training_curves.png   # Loss & accuracy per epoch
│   │   ├── confusion_matrices.png
│   │   ├── model_comparison.png
│   │   └── per_class_f1.png
│   ├── taal/                     # Same structure as bol/
│   └── final_comparison.png
│
└── Dataset/                     # Expected local data directory; recordings are not included in this GitHub tree
    
        ├── Bols/                 # 8 classes × 100 files each
        │   ├── dha/  dhin/  ga/  ka/  t/  ta/  tin/  tit/
        └── Taals/                # 8 classes × 100 files each
            ├── addhatrital/  bhajani/  dadra/  deepchandi/
            └── ektal/  jhaptal/  rupak/  trital/
```

**Code Stats**: ~3,200 lines of Python across 8 source files.

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
git clone https://github.com/astro538/Tabla-Taal-Identifier.git
cd Tabla-Taal-Identifier

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchaudio` | Deep learning framework |
| `librosa` | Audio loading, mel spectrograms, onset detection |
| `scikit-learn` | Stratified K-Fold, metrics |
| `numpy`, `pandas` | Data processing |
| `matplotlib`, `seaborn` | Result visualization |
| `fastapi`, `uvicorn` | REST API server |
| `python-multipart` | File upload handling |

---

## Quick Start

### Run the API Server

```bash
python3 -m api.main
```

Open in browser:
- **Web Demo**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs

### Train Models (from scratch)

```bash
# Train Bol classifier
python3 train.py --task bol

# Train Taal classifier
python3 train.py --task taal
```

### Test with curl

```bash
# Single Bol/Taal prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@your_audio.wav" -F "task=both"

# Full Bol sequence extraction (combined approach)
curl -X POST http://localhost:8000/api/v1/predict/sequence \
  -F "file=@your_taal_recording.wav"
```

---

## Dataset

**Dataset availability:** the training dataset directory shown below is not present in the tracked repository tree. A fresh clone therefore does not include the 1,600 audio files. Supply the dataset locally before training, and replace this note with its actual source and license if the recordings can be publicly shared.

| Property | Bols | Taals |
|----------|------|-------|
| Classes | 8 (Dha, Dhin, Ga, Ka, T, Ta, Tin, Tit) | 8 (Dadra, Trital, Rupak, Ektal, ...) |
| Files per class | 100 | 100 |
| Total files | 800 | 800 |
| Duration range | 1.6s – 3.0s | 2.2s – 13.2s |
| Sample rate | 22,050 Hz (mono) | 22,050 Hz (mono) |
| Format | WAV (16-bit PCM) | WAV (16-bit PCM) |

### Dataset Organization

```
Dataset/
├── Bols/
│   ├── dha/          # 100 isolated "Dha" stroke recordings
│   │   ├── dha01.wav
│   │   ├── dha02.wav
│   │   └── ...
│   ├── dhin/         # 100 "Dhin" recordings
│   └── ...
└── Taals/
    ├── dadra/        # 100 Dadra taal cycle recordings
    │   ├── dadra01.wav
    │   └── ...
    └── ...
```

---

## Training Pipeline (Deep Dive)

This is the most critical part of the system. With only **100 samples per class**, every technique is designed to maximize learning from limited data.

### 1. Data Loading & Preprocessing

**File**: `src/dataset.py` → `load_and_preprocess_audio()`

Every audio file goes through this pipeline:

```
Raw WAV file
    │
    ▼
1. Load & Resample → 22,050 Hz, mono (via librosa)
    │
    ▼
2. Trim Silence → Remove leading/trailing silence (threshold: 25 dB)
    │
    ▼
3. Normalize → Divide by max |amplitude| → range [-1, 1]
    │
    ▼
4. Pad/Truncate → Fixed length (Bol: 3.0s = 66,150 samples
                                 Taal: 8.0s = 176,400 samples)
```

**Why fixed length?** Neural networks require fixed-size inputs. Shorter audio is zero-padded; longer audio is truncated. The durations (3s for Bols, 8s for Taals) were chosen to capture the full extent of each class while keeping computation manageable.

### 2. Data Augmentation Strategy

**The core challenge**: 100 files per class is very small for deep learning. We use **3 layers of augmentation** to create virtually unlimited training variety.

#### Layer 1: Waveform Augmentation (×5 offline copies)

**File**: `src/dataset.py` → `augment_waveform()`

For each original training file, we create **5 augmented copies** at dataset initialization:

| Augmentation | Probability | Parameters | Purpose |
|-------------|:-----------:|-----------|---------|
| **Gaussian Noise** | 70% | σ ∈ [0.002, 0.008] | Simulate microphone noise, different recording conditions |
| **Random Gain** | 60% | ±6 dB | Handle volume variations |
| **Circular Time Shift** | 50% | ±15% of length | Simulate different onset positions |

**Effect**: 80 training files/class → **480 training files/class** (6× multiplier).

#### Layer 2: SpecAugment (on-the-fly, every epoch)

**File**: `src/dataset.py` → `apply_spec_augment()`

Applied after mel spectrogram extraction, **different random masks every epoch**:

```
Original Mel Spectrogram:         After SpecAugment:
┌────────────────────────┐       ┌────────────────────────┐
│████████████████████████│       │████████████████████████│
│████████████████████████│       │████████████████████████│
│████████████████████████│       │                        │  ← Freq mask
│████████████████████████│       │                        │  ← Freq mask
│████████████████████████│       │████████████████████████│
│████████████████████████│       │██████████    ██████████│  ← Time mask
│████████████████████████│       │██████████    ██████████│
└────────────────────────┘       └────────────────────────┘
```

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| Frequency masks | 2 | Number of horizontal bands to mask |
| Freq mask width | 20 | Max mel bands to mask per mask |
| Time masks | 2 | Number of vertical bands to mask |
| Time mask width | 30 | Max time frames to mask per mask |

**Effect**: Every sample looks **different every epoch**, even though the underlying audio is the same. This forces the model to learn robust features rather than memorizing specific spectrogram patterns.

#### Layer 3: Mixup (per batch, during training)

**File**: `src/trainer.py` → `mixup_data()`

Blends pairs of training samples and their labels:

```
Sample A: Mel spec of "Dha"  (label = [0,0,1,0,0,0,0,0])
Sample B: Mel spec of "Tin"  (label = [0,0,0,0,0,0,1,0])

λ = 0.7 (sampled from Beta(0.3, 0.3))

Mixed:   0.7 × A + 0.3 × B  (label = [0,0,0.7,0,0,0,0.3,0])
```

**Effect**: Creates synthetic intermediate samples, dramatically improving generalization. Especially powerful for small datasets.

#### Combined Effect

```
Per Fold:
  Original training files:        80 per class  ×  8 classes  =    640 samples
  After waveform augmentation:   480 per class  ×  8 classes  =  3,840 samples
  With SpecAugment per epoch:    Each of 3,840 has unique masks = different every epoch
  With Mixup per batch:          Random pairs blended = virtually unlimited variety

  → Model NEVER sees the same exact input twice across epochs
```

### 3. Feature Extraction

**File**: `src/dataset.py` → `extract_mel_spectrogram()`

Audio is converted to mel spectrogram — a time-frequency representation that captures both spectral content (pitch, timbre) and temporal structure (rhythm, onset patterns).

```python
# Parameters (matching training exactly)
sample_rate  = 22050     # Hz
n_fft       = 2048       # FFT window size (~93ms)
hop_length  = 512        # Step between frames (~23ms)
n_mels      = 128        # Frequency resolution (mel bands)
```

**Processing**:
1. **Mel filterbank** → 128 triangular filters, spaced on mel scale (emphasizes human-perceivable frequencies)
2. **Power to dB** → Logarithmic compression (models human loudness perception)
3. **Normalize** → Zero mean, unit variance per sample (stable training)

**Output shape**: `(1, 128, T)` where T = time frames
- Bol: `(1, 128, 130)` — 3.0s at 512 hop
- Taal: `(1, 128, 345)` — 8.0s at 512 hop

### 4. Model Architectures

**File**: `src/models.py`

Four architectures were designed and compared, each bringing different strengths:

#### Model 1: ResidualCNN

```
Input (1, 128, T)
  ▼
Conv(1→32, 7×7, stride=2) → BN → GELU → MaxPool
  ▼
ResBlock(32→32) × 2           ← Skip connections prevent degradation
  ▼
ResBlock(32→64, stride=2) × 2
  ▼
ResBlock(64→128, stride=2) × 2
  ▼
ResBlock(128→256, stride=2) × 2
  ▼
AdaptiveAvgPool(1,1) → FC(256) → FC(128) → FC(8)
```

**Strengths**: Strong spatial feature extraction, skip connections enable deep learning without vanishing gradients. Best single model for Bol classification (97.88%).

#### Model 2: CRNN (Convolutional Recurrent Neural Network)

```
Input (1, 128, T)
  ▼
ConvBlock(1→32) → ConvBlock(32→64) → ConvBlock(64→128) → ConvBlock(128→128)
  ▼
GlobalAvgPool (frequency axis)     ← Collapses freq to get time sequence
  ▼
BiLSTM (128→256, 2 layers)        ← Captures temporal dependencies
  ▼
LayerNorm → MeanPool(time) → FC(128) → FC(8)
```

**Strengths**: LSTM captures long-range temporal patterns in both forward and backward directions. Best for Taal (100% accuracy) because Taals are defined by their temporal beat patterns.

#### Model 3: AttentionCRNN

```
CRNN backbone
  ▼
Multi-Head Self-Attention (4 heads, d=256)
  ▼
Feed-Forward Network + Residual
  ▼
Learnable Attention Pooling        ← Learns which time steps matter most
  ▼
FC(128) → FC(8)
```

**Strengths**: Attention mechanism allows the model to focus on the most discriminative moments. Also achieves 100% on Taal.

#### Model 4: ASTLite (Audio Spectrogram Transformer)

```
Input (1, 128, T)
  ▼
Patch Embedding (16×16 patches via Conv2d)
  ▼
+ CLS Token + Positional Embeddings
  ▼
Transformer Encoder × 4 (d=192, 4 heads, pre-norm)
  ▼
CLS token → FC(192) → FC(8)
```

**Strengths**: Inspired by Vision Transformer (ViT) and AST. Treats spectrogram as image patches. Good global context but requires more data to excel.

### 5. Training Engine

**File**: `src/trainer.py`

Each model is trained with these components:

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Optimizer** | AdamW (lr=1e-3, wd=1e-4) | Weight decay regularization |
| **Loss** | CrossEntropy + Label Smoothing (0.1) | Prevents overconfident predictions |
| **LR Schedule** | Cosine Annealing + 5-epoch Linear Warmup | Stable start, smooth decay |
| **Gradient Clipping** | max_norm=1.0 | Stable LSTM training |
| **Early Stopping** | Patience=12 on val_acc | Prevents overfitting |
| **Mixup** | α=0.3 (Beta distribution) | Data-efficient regularization |

#### Label Smoothing

Instead of hard targets `[0, 0, 1, 0, ...]`, uses soft targets: `[0.0125, 0.0125, 0.9125, 0.0125, ...]` (with smoothing=0.1). This prevents the model from becoming overconfident and improves generalization.

#### Cosine Warmup LR Schedule

```
Learning Rate
    │
1e-3│          ╲
    │         ╱  ╲
    │        ╱    ╲
    │       ╱      ╲
    │      ╱        ╲
    │     ╱          ╲
1e-6│ ──╱              ╲──────
    └────────────────────────── Epochs
     warmup    cosine decay
     (5 ep)    (remaining)
```

### 6. Validation & Cross-Validation

**File**: `train.py`

#### 5-Fold Stratified Cross-Validation

```
100 files per class
        │
Fold 1: Train [21-100]     → Validate [1-20]      (80/20 split)
Fold 2: Train [1-20,41-100] → Validate [21-40]
Fold 3: Train [1-40,61-100] → Validate [41-60]
Fold 4: Train [1-60,81-100] → Validate [61-80]
Fold 5: Train [1-80]        → Validate [81-100]
```

**Why Stratified?** `StratifiedKFold` ensures each fold has exactly the same class distribution — critical with balanced datasets to prevent any fold from having uneven class representation.

**Why 5 folds?** Every single file gets validated **exactly once**. The reported accuracy is the average across all 5 folds, which is much more reliable than a single train/test split.

#### Validation Protocol

- **Training set**: Augmented (waveform copies + SpecAugment + Mixup)
- **Validation set**: **ZERO augmentation** — raw original files only
- **No weight updates** during validation (`torch.no_grad()`)
- **Best checkpoint** saved when val_acc improves

#### Metrics Computed

| Metric | Description |
|--------|-------------|
| Accuracy | Overall percentage correct |
| Precision | Of predicted X, how many actually X? (weighted avg) |
| Recall | Of actual X, how many predicted X? (weighted avg) |
| F1 Score | Harmonic mean of precision & recall |
| Confusion Matrix | N×N grid showing all class predictions |
| Per-class Report | Above metrics for each individual class |

### 7. Ensemble Method

After training all 4 models, we compute an ensemble prediction by averaging their softmax probabilities:

```python
ensemble_probs = (ResidualCNN_probs + CRNN_probs + AttentionCRNN_probs + ASTLite_probs) / 4
final_prediction = argmax(ensemble_probs)
```

**Effect**: Ensemble achieves 99.38% Bol accuracy (vs 97.88% for best single model), because different architectures make different mistakes that cancel out.

---

## Inference Pipeline

**File**: `src/inference.py` → `TablaPredictor`

### Single Bol / Taal Prediction

For isolated audio clips (single Bol stroke or Taal cycle):

1. Load audio (WAV/MP3/OGG/FLAC/M4A)
2. Preprocess (trim, normalize, pad/truncate) — **identical to training**
3. Extract mel spectrogram — **identical to training**
4. Run through model → softmax → top-k predictions

### Combined Approach (Sequence Prediction)

For full Taal recordings, extracts **every Bol** in the audio:

```
Audio File
    │
    ├─── Step 1: Taal Classification (CRNN model)
    │              → "Dadra" (70.7% confidence)
    │
    ├─── Step 2: Template Lookup
    │              → Dadra theka: Dha Dhin Na | Dha Tin Na
    │
    ├─── Step 3: Onset Detection (librosa)
    │              → [0.05s, 0.42s, 0.84s, 1.25s, 1.46s, 1.70s, ...]
    │
    ├─── Step 4: Per-Segment Bol Classification (ResidualCNN)
    │              → Each onset segment classified independently
    │
    └─── Step 5: Cross-Validation
                   │
                   ├── Template says "Dha", Model says "Dha" → source: "both" ✓
                   ├── Template says "Dhin", Model says "Ka" → source: "template" (trust Taal)
                   └── Template says "Na" (not in model vocab) → source: "template"
```

The **cross-validation logic**:
- If both agree → `source: "both"` — highest confidence
- If they disagree but Taal confidence > 60% → `source: "template"` — trust the known pattern
- If template Bol isn't in model vocabulary (Na, Ge, Ti, Dhi) → `source: "template"` — only template knows these
- If Taal confidence is low → `source: "model"` — let the classifier decide

### Supported Taal Thekas (Bol Patterns)

| Taal | Beats | Theka |
|------|:-----:|-------|
| **Dadra** | 6 | Dha Dhin Na \| Dha Tin Na |
| **Trital** | 16 | Dha Dhin Dhin Dha \| Dha Dhin Dhin Dha \| Dha Tin Tin Ta \| Ta Dhin Dhin Dha |
| **Rupak** | 7 | Tin Tin Na \| Dhin Na \| Dhin Na |
| **Ektal** | 12 | Dhin Dhin Dha Dha \| Tin Tin Na Na \| Dhin Dhin Dha Dha |
| **Jhaptal** | 10 | Dhi Na \| Dhi Dhi Na \| Ti Na \| Dhi Dhi Na |
| **Bhajani** | 8 | Dha Ge Na Ti \| Na Ka Dhi Na |
| **Deepchandi** | 14 | Dha Dhin - Dha \| Dha Dhin - Ta \| Tin - Dha Dha \| Dhin - |
| **Addha Trital** | 14 | Dha Dhin \| Dha Dha Tin Ta \| Ta Tin Dha Dha \| Dhin Dha Dha Tin |

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /api/v1/health`

Health check.

```json
{"status": "healthy", "models_loaded": true, "device": "mps"}
```

### `POST /api/v1/predict`

Predict Taal and/or Bol from audio file.

**Parameters**:
- `file` (form-data): Audio file
- `task` (query): `"bol"`, `"taal"`, or `"both"` (default)

**Response**:
```json
{
  "status": "success",
  "task": "both",
  "bol": {"label": "dha", "display_name": "Dha", "confidence": 0.905, "top_k": [...]},
  "taal": {"label": "dadra", "display_name": "Dadra", "confidence": 0.707, "top_k": [...]},
  "audio_info": {"duration_seconds": 2.57, "sample_rate": 22050},
  "processing_time_ms": 108.0
}
```

### `POST /api/v1/predict/sequence`

**Combined approach** — extract full Bol sequence from a Taal recording.

**Parameters**:
- `file` (form-data): Audio file of Tabla playing

**Response**:
```json
{
  "status": "success",
  "taal": {"label": "dadra", "display_name": "Dadra", "confidence": 0.707, ...},
  "theka": "Dha Dhin Na | Dha Tin Na",
  "bol_sequence": [
    {
      "time": 0.046,
      "end_time": 0.418,
      "bol": "Dha",
      "confidence": 0.81,
      "source": "both",
      "template_bol": "Dha",
      "model_bol": "Dha",
      "beat_position": 1,
      "cycle": 1
    },
    ...
  ],
  "bol_count": 8,
  "cycles_detected": 1,
  "pattern_length": 6,
  "processing_time_ms": 3061.9
}
```

### `GET /api/v1/models`

Get loaded model information.

### `GET /api/v1/classes`

Get available Bol/Taal class names.

---

## Web Demo

The web interface at `http://localhost:8000` provides:

- **Drag & drop** audio file upload
- **Audio playback** with controls
- **Taal detection** with confidence bars and theka display
- **Live Bol timeline** — highlights active Bol as audio plays
- **Source attribution** — color-coded: 🟢 Both agree, 🔵 Template, 🟡 Model
- **Cycle display** — shows which cycle each Bol belongs to
- **Click-to-seek** — click timeline to jump to any position

---

## Results

### Per-Fold Consistency (Bol — ResidualCNN)

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|:------:|:------:|:------:|:------:|:------:|:----:|:---:|
| 97.5% | 99.4% | 96.9% | 98.1% | 97.5% | 97.88% | 0.85% |

The low standard deviation (0.85%) indicates similar performance across the five recorded folds. This alone does **not** prove that data leakage is absent; for recordings from repeated sessions or performers, use participant- or session-independent evaluation where appropriate.

### Where the Bol Model Struggles

The confusion matrix shows realistic errors between acoustically similar Bols:
- **Dha ↔ Dhin**: Both are bass + treble strokes (Dha = open, Dhin = pressed)
- **T ↔ Ta**: Both are treble-only strokes (T = light, Ta = sharp)
- **Ga ↔ Ka**: Both are bass strokes on the smaller drum

These confusions align with how human learners also struggle to distinguish these pairs.

---

## Future Work

### Phase 1: Improved Bol Extraction from Taals
- Fine-tune Bol model on segments extracted from Taal recordings (not just isolated Bols)
- Add Bols not currently in vocabulary: Na, Ge, Ti, Dhi, Tirakita

### Phase 2: Real-time Detection
- WebSocket endpoint for streaming audio
- Sliding window with overlap for continuous detection
- Tempo estimation from onset intervals

### Phase 3: Mobile App
- Convert models to ONNX / TorchScript / CoreML
- React Native or Flutter frontend
- On-device inference for offline use

### Phase 4: Robustness
- Collect data from multiple Tabla players
- Test across different microphones and environments
- Add noise augmentation for live performance scenarios

---

## License

This project is for educational and research purposes.

## Acknowledgments

- Training recordings: the dataset location and license still need to be documented before redistributing them.
- Inspired by the Tarang Report on Tabla classification
- Model architectures inspired by ResNet, CRNN, AST research papers
