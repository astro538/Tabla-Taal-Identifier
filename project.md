# Tabla Taal & Bol Identifier — Project Documentation

## 1. Project Overview

**Goal**: Build an end-to-end deep learning system to classify Indian Tabla music into:
- **8 Bols** (individual strokes): Dha, Dhin, Ga, Ka, T, Ta, Tin, Tit
- **8 Taals** (rhythmic patterns): Addha Trital, Bhajani, Dadra, Deepchandi, Ektal, Jhaptal, Rupak, Trital

**Tech Stack**: Python, PyTorch, librosa, scikit-learn, FastAPI, React Native (Expo)

**Key Achievement**: 99.38% Bol accuracy (Ensemble), 100% Taal accuracy (CRNN) — surpassing the prior Tarang Report baseline (91% Bol, 97% Taal).

---

## 2. Dataset

- **Source**: Kaggle — [Tabla Taala Dataset Visualizations](https://www.kaggle.com/code/pranav6670/tabla-taala-dataset-visualizations)
- **Total files**: 1,600 WAV files (800 Bols + 800 Taals)
- **Classes**: 8 per task, **perfectly balanced** — 100 files per class
- **Format**: WAV 16-bit PCM, mono channel
- **Sample rate**: 22,050 Hz
- **Duration range**: Bols 1.6–3.0s, Taals 2.2–13.2s
- **Organization**: `Dataset/Bols/<class_name>/*.wav` and `Dataset/Taals/<class_name>/*.wav`

### Bol Classes (Individual Strokes)
| Bol | Type | Description |
|---|---|---|
| Dha | Bass + Treble (open) | Open bass stroke on bayan + open treble on dayan |
| Dhin | Bass + Treble (pressed) | Pressed bass + index finger press on dayan |
| Ga | Bass only | Resonant strike on bayan (left drum) |
| Ka | Bass only (muted) | Non-resonant strike on bayan |
| T | Treble (light) | Light tap on dayan rim |
| Ta | Treble (sharp) | Sharp strike on dayan edge |
| Tin | Treble (ring) | Ring finger strike on dayan |
| Tit | Treble (damped) | Damped stroke on dayan |

### Taal Classes (Rhythmic Patterns)
| Taal | Beats | Vibhag | Theka (Bol Pattern) |
|---|:---:|---|---|
| Dadra | 6 | 3-3 | Dha Dhin Na \| Dha Tin Na |
| Trital | 16 | 4-4-4-4 | Dha Dhin Dhin Dha \| Dha Dhin Dhin Dha \| Dha Tin Tin Ta \| Ta Dhin Dhin Dha |
| Rupak | 7 | 3-2-2 | Tin Tin Na \| Dhin Na \| Dhin Na |
| Ektal | 12 | 4-4-4 | Dhin Dhin Dha Dha \| Tin Tin Na Na \| Dhin Dhin Dha Dha |
| Jhaptal | 10 | 2-3-2-3 | Dhi Na \| Dhi Dhi Na \| Ti Na \| Dhi Dhi Na |
| Bhajani | 8 | 4-4 | Dha Ge Na Ti \| Na Ka Dhi Na |
| Deepchandi | 14 | 3-4-3-4 | Dha Dhin - Dha \| Dha Dhin - Ta \| Tin - Dha Dha \| Dhin - |
| Addha Trital | 14 | 2-4-4-4 | Dha Dhin \| Dha Dha Tin Ta \| Ta Tin Dha Dha \| Dhin Dha Dha Tin |

### Dataset Challenge
With only **100 samples per class**, this is a **small-data deep learning problem**. Standard approaches would overfit. Our 3-layer augmentation strategy (Section 4) addresses this by creating virtually unlimited training variety from limited originals.

---

## 3. Training Pipeline Architecture

```
Raw Audio (.wav)
    ↓
1. Load & Resample (22050 Hz, mono) — librosa
    ↓
2. Trim Silence (threshold: 25 dB) — librosa.effects.trim
    ↓
3. Normalize Amplitude (divide by max |amplitude| → [-1, 1])
    ↓
4. Pad/Truncate to fixed length (Bol: 3.0s=66150 samples, Taal: 8.0s=176400 samples)
    ↓
5. Data Augmentation (3 layers — training only)
    ↓
6. Mel Spectrogram Extraction (n_fft=2048, hop=512, n_mels=128)
    ↓
7. Power → dB → Zero-mean, Unit-variance normalization
    ↓
8. Output tensor shape: (1, 128, T) where T=130 (Bol) or 345 (Taal)
    ↓
9. Feed into 4 model architectures
    ↓
10. 5-Fold Stratified Cross-Validation
    ↓
11. Ensemble via weighted majority voting
```

### End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END SYSTEM                                │
│                                                                         │
│  ┌───────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│  │   TRAINING     │     │    INFERENCE      │     │   APPLICATION    │   │
│  │   PIPELINE     │────▶│    ENGINE         │────▶│   LAYER          │   │
│  │                │     │                   │     │                  │   │
│  │ • 4 Models     │     │ • TablaPredictor  │     │ • FastAPI        │   │
│  │ • 5-Fold CV    │     │ • Best fold ckpt  │     │ • Web UI         │   │
│  │ • Augmentation │     │ • Combined pred   │     │ • Mobile App     │   │
│  │ • ~3200 LOC    │     │ • Onset detection │     │ • REST API       │   │
│  └───────────────┘     └──────────────────┘     └──────────────────┘   │
│         │                       │                        │              │
│    train.py              src/inference.py           api/main.py        │
│    src/models.py                                    mobile/             │
│    src/trainer.py                                                       │
│    src/dataset.py                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Augmentation Pipeline (Block Diagram)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    3-LAYER AUGMENTATION STRATEGY                      │
│                                                                      │
│  LAYER 1: Waveform Augmentation (offline, at dataset init)           │
│  ┌──────────┐                                                        │
│  │ Original │──┬──→ Copy 1: noise(σ=0.005) + gain(+3dB)             │
│  │  Audio   │  ├──→ Copy 2: shift(+10%) + noise(σ=0.003)            │
│  │ (1 file) │  ├──→ Copy 3: gain(-4dB)                              │
│  │          │  ├──→ Copy 4: noise(σ=0.007) + shift(-8%)             │
│  │          │  └──→ Copy 5: gain(+2dB) + noise(σ=0.004)            │
│  └──────────┘                                                        │
│  Result: 1 original + 5 augmented = 6 samples per file               │
│  Per fold: 80 files × 6 = 480 per class × 8 classes = 3,840 total   │
│                                                                      │
│  LAYER 2: SpecAugment (on-the-fly, every __getitem__ call)           │
│  ┌────────────────────────┐     ┌────────────────────────┐           │
│  │ ████████████████████████│     │ ████████████████████████│           │
│  │ ████████████████████████│     │ ████████████████████████│           │
│  │ ████████████████████████│ ──▶ │                        │ ← freq   │
│  │ ████████████████████████│     │ ██████████    ██████████│ ← time   │
│  │ ████████████████████████│     │ ████████████████████████│           │
│  └────────────────────────┘     └────────────────────────┘           │
│  Different random masks every epoch → same data looks different       │
│                                                                      │
│  LAYER 3: Mixup (per batch, in training loop)                        │
│  Sample A (λ=0.7) ──┐                                                │
│                      ├──→ Mixed = 0.7·A + 0.3·B                      │
│  Sample B (1-λ=0.3) ─┘    Label = 0.7·[Dha] + 0.3·[Tin]            │
│                                                                      │
│  COMBINED: Model NEVER sees identical input twice across epochs       │
└──────────────────────────────────────────────────────────────────────┘
```

### Training Loop Block Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP (per model, per fold)             │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │ DataLoader   │    │   Mixup     │    │   Forward   │              │
│  │ (batch=32)   │───▶│  (α=0.3)   │───▶│    Pass     │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│                                                │                     │
│                                                ▼                     │
│                                       ┌─────────────┐               │
│  ┌─────────────┐    ┌─────────────┐  │ CrossEntropy │               │
│  │  Update LR   │    │  Backward   │◀─│ + LabelSmooth│               │
│  │  (Cosine     │◀──│    Pass     │  │   (0.1)     │               │
│  │   Warmup)    │    │ + GradClip  │  └─────────────┘               │
│  └──────┬──────┘    └─────────────┘                                 │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Validate    │    │ Save Best   │    │   Early     │              │
│  │ (no augment, │───▶│  Checkpoint │───▶│  Stopping   │              │
│  │  no_grad)    │    │ (if improved)│   │ (patience=12)│              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                      │
│  Repeat for max 80 epochs or until early stopping triggers           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Augmentation (3 Layers)

### Layer 1: Waveform Augmentation (offline, ×5 copies per original)
| Augmentation | Probability | Parameters | Purpose |
|---|:---:|---|---|
| Gaussian Noise | 70% | σ ∈ [0.002, 0.008] | Simulate mic noise |
| Random Gain | 60% | ±6 dB | Handle volume variations |
| Circular Time Shift | 50% | ±15% of length | Different onset positions |

**Effect**: 80 train files/class → 480 files/class (6× multiplier)

### Layer 2: SpecAugment (on-the-fly, every epoch)
- 2 frequency masks (max 20 mel bands each)
- 2 time masks (max 30 frames each)
- Applied after mel spectrogram extraction, different masks every epoch

### Layer 3: Mixup (per batch)
- α=0.3, Beta distribution sampling
- Blends pairs of samples and their soft labels
- `mixed = λ·A + (1-λ)·B`, `label = λ·label_A + (1-λ)·label_B`

**Combined effect**: Model never sees the same exact input twice across epochs.

---

## 5. Feature Extraction — Mel Spectrogram

| Parameter | Value |
|---|---|
| Sample Rate | 22,050 Hz |
| FFT Window (n_fft) | 2048 (~93ms) |
| Hop Length | 512 (~23ms) |
| Mel Bands (n_mels) | 128 |
| Power | 2.0 |
| dB Conversion | librosa.power_to_db (ref=max) |
| Normalization | Zero mean, unit variance per sample |
| Output Shape (Bol) | (1, 128, 130) |
| Output Shape (Taal) | (1, 128, 345) |

---

## 6. Model Architectures (4 models)

### Model 1: ResidualCNN
ResNet-inspired CNN with skip connections. Best single model for Bol classification.

```
┌─────────────────────────────────────────────────────────────┐
│                    ResidualCNN Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (B, 1, 128, T)      ← Mel spectrogram              │
│       ↓                                                     │
│  ┌──────────────────────────────────┐                       │
│  │ Conv2d(1→32, 7×7, stride=2)     │                       │
│  │ BatchNorm2d(32) → GELU          │  → (B, 32, 64, T/2)   │
│  │ MaxPool2d(3, stride=2, pad=1)   │  → (B, 32, 32, T/4)   │
│  └──────────────────────────────────┘                       │
│       ↓                                                     │
│  ┌──────────────────────────────────┐                       │
│  │ ResBlock(32→32) × 2             │                       │
│  │  ┌─────────────────────┐        │                       │
│  │  │ Conv(3×3) → BN → GELU       │                       │
│  │  │ Dropout2d(0.1)      │──┐     │                       │
│  │  │ Conv(3×3) → BN       │  │add │  → (B, 32, 32, T/4)   │
│  │  │       ↓              │  │     │                       │
│  │  │     + shortcut ←─────┘  │     │                       │
│  │  │     GELU                │     │                       │
│  │  └─────────────────────┘        │                       │
│  └──────────────────────────────────┘                       │
│       ↓                                                     │
│  ResBlock(32→64, stride=2) × 2     → (B, 64, 16, T/8)      │
│       ↓                                                     │
│  ResBlock(64→128, stride=2) × 2    → (B, 128, 8, T/16)     │
│       ↓                                                     │
│  ResBlock(128→256, stride=2) × 2   → (B, 256, 4, T/32)     │
│       ↓                                                     │
│  AdaptiveAvgPool2d(1, 1)           → (B, 256, 1, 1)        │
│       ↓                                                     │
│  Flatten                           → (B, 256)               │
│       ↓                                                     │
│  FC(256→256) → GELU → Dropout(0.3)                         │
│  FC(256→128) → GELU → Dropout(0.15)                        │
│  FC(128→8)                         → (B, 8) logits         │
│                                                             │
│  Strengths: Skip connections prevent vanishing gradients.   │
│  Best for Bol (97.88%) — spatial feature extraction.        │
└─────────────────────────────────────────────────────────────┘
```

### Model 2: CRNN (CNN + Bidirectional LSTM)
CNN extracts spectral features, BiLSTM captures temporal dependencies.

```
┌─────────────────────────────────────────────────────────────┐
│                     CRNN Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (B, 1, 128, T)                                      │
│       ↓                                                     │
│  ┌─── CNN Feature Extractor ───────────────────────┐        │
│  │ ConvBlock(1→32):   Conv(3×3)→BN→GELU→Pool(2,2)→Drop    │
│  │                                   → (B, 32, 64, T/2)    │
│  │ ConvBlock(32→64):  Conv(3×3)→BN→GELU→Pool(2,2)→Drop    │
│  │                                   → (B, 64, 32, T/4)    │
│  │ ConvBlock(64→128): Conv(3×3)→BN→GELU→Pool(2,2)→Drop    │
│  │                                   → (B, 128, 16, T/8)   │
│  │ ConvBlock(128→128):Conv(3×3)→BN→GELU→Pool(2,1)→Drop    │
│  │                                   → (B, 128, 8, T/8)    │
│  └─────────────────────────────────────────────────┘        │
│       ↓                                                     │
│  GlobalAvgPool(dim=2) — collapse frequency  → (B, 128, T/8)│
│       ↓                                                     │
│  Permute(0,2,1)                             → (B, T/8, 128)│
│       ↓                                                     │
│  ┌─── Bidirectional LSTM ──────────────────────────┐        │
│  │ BiLSTM(input=128, hidden=128, layers=2)         │        │
│  │ Forward LSTM ──→  128-dim                       │        │
│  │ Backward LSTM ←── 128-dim                       │        │
│  │ Concat: 128+128 = 256-dim    → (B, T/8, 256)   │        │
│  └─────────────────────────────────────────────────┘        │
│       ↓                                                     │
│  LayerNorm(256)                                             │
│       ↓                                                     │
│  MeanPool(dim=1) — avg over time   → (B, 256)              │
│       ↓                                                     │
│  FC(256→128) → GELU → Dropout(0.3)                         │
│  FC(128→8)                         → (B, 8) logits         │
│                                                             │
│  Strengths: BiLSTM captures temporal beat patterns.         │
│  Best for Taal (100%) — temporal dependency modeling.       │
└─────────────────────────────────────────────────────────────┘
```

### Model 3: AttentionCRNN
Extends CRNN with Multi-Head Self-Attention and learnable pooling.

```
┌─────────────────────────────────────────────────────────────┐
│                  AttentionCRNN Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (B, 1, 128, T)                                      │
│       ↓                                                     │
│  [Same CNN backbone as CRNN]       → (B, 128, 8, T/8)      │
│       ↓                                                     │
│  GlobalAvgPool(freq) → Permute     → (B, T/8, 128)         │
│       ↓                                                     │
│  BiLSTM(128→256, 2 layers)        → (B, T/8, 256)          │
│       ↓                                                     │
│  ┌─── Multi-Head Self-Attention ───────────────────┐        │
│  │  4 attention heads, d_model=256                 │        │
│  │                                                 │        │
│  │  Q = K = V = LSTM output                        │        │
│  │  Attention(Q,K,V) = softmax(QKᵀ/√d)·V          │        │
│  │  4 heads computed in parallel, then concat      │        │
│  │  + Residual connection + LayerNorm              │        │
│  │                                   → (B, T/8, 256)       │
│  └─────────────────────────────────────────────────┘        │
│       ↓                                                     │
│  ┌─── Feed-Forward Network ────────────────────────┐        │
│  │  FC(256→512) → GELU → Drop(0.1) → FC(512→256)  │        │
│  │  + Residual connection + LayerNorm              │        │
│  │                                   → (B, T/8, 256)       │
│  └─────────────────────────────────────────────────┘        │
│       ↓                                                     │
│  ┌─── Learnable Attention Pooling ─────────────────┐        │
│  │  α = softmax(FC(256→1))   — learn per-timestep  │        │
│  │  output = Σ(α_t · h_t)   — weighted sum         │        │
│  │                                   → (B, 256)    │        │
│  └─────────────────────────────────────────────────┘        │
│       ↓                                                     │
│  FC(256→128) → GELU → Dropout(0.3)                         │
│  FC(128→8)                         → (B, 8) logits         │
│                                                             │
│  Strengths: Attention focuses on discriminative moments.    │
│  Taal: 100%, Bol: 96.63%                                   │
└─────────────────────────────────────────────────────────────┘
```

### Model 4: ASTLite (Audio Spectrogram Transformer)
Treats mel spectrogram as image patches, processes with Transformer.

```
┌─────────────────────────────────────────────────────────────┐
│                    ASTLite Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (B, 1, 128, T)                                      │
│       ↓                                                     │
│  ┌─── Patch Embedding ────────────────────────────┐         │
│  │ Conv2d(1→192, kernel=16, stride=16)            │         │
│  │ Splits spectrogram into 16×16 non-overlapping  │         │
│  │ patches, each projected to d=192               │         │
│  │ Flatten spatial → (B, N_patches, 192)          │         │
│  └────────────────────────────────────────────────┘         │
│       ↓                                                     │
│  Prepend [CLS] token (learnable)   → (B, 1+N, 192)         │
│  + Positional Embeddings (learnable, max 300+1)             │
│  + Dropout(0.2)                                             │
│       ↓                                                     │
│  ┌─── Transformer Encoder × 4 ────────────────────┐        │
│  │  Pre-Norm (LayerNorm before attention)          │        │
│  │                                                 │        │
│  │  Multi-Head Attention (4 heads, d=192)          │        │
│  │  + Residual + Dropout                           │        │
│  │       ↓                                         │        │
│  │  LayerNorm                                      │        │
│  │  FFN: FC(192→768) → GELU → Drop → FC(768→192)  │        │
│  │  + Residual + Dropout                           │        │
│  │                                   → (B, 1+N, 192)       │
│  └─────────────────────────────────────────────────┘        │
│       ↓                                                     │
│  Final LayerNorm                                            │
│  Extract [CLS] token output        → (B, 192)              │
│       ↓                                                     │
│  FC(192→192) → GELU → Dropout(0.2)                         │
│  FC(192→8)                         → (B, 8) logits         │
│                                                             │
│  Init: truncated normal (σ=0.02), Kaiming for Conv2d       │
│  Needs more data to excel. Bol: 90.88%, Taal: 98.75%       │
└─────────────────────────────────────────────────────────────┘
```

### Model Comparison Summary
| Model | Params | Bol Acc | Taal Acc | Strength |
|---|---|:---:|:---:|---|
| ResidualCNN | ~1.2M | 97.88% | 99.88% | Spatial features, skip connections |
| CRNN | ~0.8M | 97.13% | **100%** | Temporal dependencies via BiLSTM |
| AttentionCRNN | ~1.1M | 96.63% | **100%** | Focused on discriminative timesteps |
| ASTLite | ~0.6M | 90.88% | 98.75% | Global context, data-hungry |
| **Ensemble** | all | **99.38%** | **100%** | Error cancellation across architectures |

---

## 7. Training Configuration

| Component | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Loss | CrossEntropyLoss + Label Smoothing (0.1) |
| LR Schedule | Cosine Annealing + 5-epoch Linear Warmup (min_lr=1e-6) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | patience=12, monitoring val_acc |
| Batch Size | 32 |
| Max Epochs | 80 |
| Mixup | α=0.3 |
| Cross-Validation | 5-fold Stratified K-Fold |
| Seed | 42 |
| Validation | Zero augmentation, torch.no_grad() |

### Label Smoothing
Hard `[0,0,1,0,...]` → Soft `[0.0125, 0.0125, 0.9125, 0.0125, ...]` (smoothing=0.1)

### LR Schedule
```
warmup (5 epochs): linear ramp 0 → 1e-3
cosine decay (remaining): 1e-3 → 1e-6
```

---

## 8. Cross-Validation & Ensemble

### 5-Fold Stratified CV
- Each fold: 80% train, 20% val
- Stratified ensures equal class distribution per fold
- Every sample validated exactly once across all folds
- Best checkpoint saved per fold per model
- Total model checkpoints: 4 models × 5 folds = 20

### Ensemble (Weighted Majority Voting)
- Collect predictions from all 4 models' best folds
- Weight each model by its accuracy
- Per-sample: accumulate weighted votes, pick argmax
- Bol Ensemble: 99.38% (vs 97.88% best single)
- Taal Ensemble: 100%

---

## 9. Results

### Bol Classification
| Model | Accuracy | F1 Score |
|---|:---:|:---:|
| ResidualCNN | 97.88% | — |
| CRNN | 97.13% | — |
| AttentionCRNN | 96.63% | — |
| ASTLite | 90.88% | — |
| **Ensemble** | **99.38%** | **99.37%** |

### Taal Classification
| Model | Accuracy | F1 Score |
|---|:---:|:---:|
| ResidualCNN | 99.88% | — |
| CRNN | **100.00%** | — |
| AttentionCRNN | **100.00%** | — |
| ASTLite | 98.75% | — |
| **Ensemble** | **100.00%** | **100.00%** |

### Fold Consistency (Bol — ResidualCNN)
Fold 1: 97.5% | Fold 2: 99.4% | Fold 3: 96.9% | Fold 4: 98.1% | Fold 5: 97.5% | **Mean: 97.88% ± 0.85%**

### Common Bol Confusions
- Dha ↔ Dhin (both bass+treble strokes)
- T ↔ Ta (both treble-only strokes)
- Ga ↔ Ka (both bass strokes)

---

## 10. Inference Pipeline

### Single Prediction Flow
```
┌────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────┐
│ Audio File  │───▶│ Load+Resample│───▶│ Trim+Normalize │───▶│ Pad/Trunc│
│ (any format)│    │ (22050 Hz)   │    │ (silence, amp) │    │ (3s/8s)  │
└────────────┘    └──────────────┘    └────────────────┘    └────┬─────┘
                                                                 │
      ┌──────────────────────────────────────────────────────────┘
      ▼
┌────────────────┐    ┌──────────────┐    ┌───────────────┐
│ Mel Spectrogram│───▶│ Model Forward│───▶│ Softmax → Top-k│
│ (128×T, dB,   │    │ (no_grad)    │    │ predictions    │
│  normalized)   │    └──────────────┘    └───────────────┘
└────────────────┘
```

### Combined Approach (Sequence Prediction) — Block Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│              COMBINED INFERENCE PIPELINE (5 Steps)                    │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │  Audio Input  │                                                    │
│  └──────┬───────┘                                                    │
│         │                                                            │
│    ┌────┴────────────────────────┐                                   │
│    │                             │                                    │
│    ▼                             ▼                                    │
│  ┌────────────────┐    ┌─────────────────┐                           │
│  │ STEP 1: Taal   │    │ STEP 3: Onset   │                           │
│  │ Classification │    │ Detection       │                           │
│  │ (CRNN model)   │    │ (librosa)       │                           │
│  │                │    │                 │                           │
│  │ Full audio →   │    │ Detect stroke   │                           │
│  │ mel spec →     │    │ boundaries via  │                           │
│  │ CRNN → softmax │    │ spectral flux   │                           │
│  │                │    │ + backtracking  │                           │
│  │ Output: "Dadra"│    │                 │                           │
│  │ (conf: 70.7%)  │    │ Output: [0.05s, │                           │
│  └───────┬────────┘    │ 0.42s, 0.84s,..]│                           │
│          │             └────────┬────────┘                           │
│          │                      │                                    │
│          ▼                      ▼                                    │
│  ┌────────────────┐    ┌─────────────────┐                           │
│  │ STEP 2:Template│    │ STEP 4: Per-Seg │                           │
│  │ Lookup         │    │ Bol Classify    │                           │
│  │                │    │ (ResidualCNN)   │                           │
│  │ Dadra theka:   │    │                 │                           │
│  │ [Dha,Dhin,Na,  │    │ Each onset →    │                           │
│  │  Dha,Tin,Na]   │    │ extract segment │                           │
│  └───────┬────────┘    │ → mel spec →    │                           │
│          │             │ ResidualCNN     │                           │
│          │             └────────┬────────┘                           │
│          │                      │                                    │
│          └──────────┬───────────┘                                    │
│                     ▼                                                │
│          ┌──────────────────────┐                                    │
│          │ STEP 5: Cross-       │                                    │
│          │ Validation           │                                    │
│          │                      │                                    │
│          │ Template  vs  Model  │                                    │
│          │    Bol         Bol   │                                    │
│          │                      │                                    │
│          │ Decision Tree:       │                                    │
│          │ ├─ Agree → "both"    │                                    │
│          │ ├─ Disagree:         │                                    │
│          │ │  ├─ Taal>60% →     │                                    │
│          │ │  │  "template"     │                                    │
│          │ │  └─ else → "model" │                                    │
│          │ └─ Not in vocab →    │                                    │
│          │    "template"        │                                    │
│          └──────────┬───────────┘                                    │
│                     ▼                                                │
│          ┌──────────────────────┐                                    │
│          │ OUTPUT:              │                                    │
│          │ Timestamped Bol      │                                    │
│          │ sequence with:       │                                    │
│          │ • time, end_time     │                                    │
│          │ • bol name           │                                    │
│          │ • confidence         │                                    │
│          │ • source attribution │                                    │
│          │ • beat position      │                                    │
│          │ • cycle number       │                                    │
│          └──────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Supported Taal Thekas
| Taal | Beats | Theka |
|---|:---:|---|
| Dadra | 6 | Dha Dhin Na \| Dha Tin Na |
| Trital | 16 | Dha Dhin Dhin Dha \| Dha Dhin Dhin Dha \| Dha Tin Tin Ta \| Ta Dhin Dhin Dha |
| Rupak | 7 | Tin Tin Na \| Dhin Na \| Dhin Na |
| Ektal | 12 | Dhin Dhin Dha Dha \| Tin Tin Na Na \| Dhin Dhin Dha Dha |
| Jhaptal | 10 | Dhi Na \| Dhi Dhi Na \| Ti Na \| Dhi Dhi Na |
| Bhajani | 8 | Dha Ge Na Ti \| Na Ka Dhi Na |
| Deepchandi | 14 | Dha Dhin - Dha \| Dha Dhin - Ta \| Tin - Dha Dha \| Dhin - |
| Addha Trital | 14 | Dha Dhin \| Dha Dha Tin Ta \| Ta Tin Dha Dha \| Dhin Dha Dha Tin |

---

## 11. API Backend (FastAPI)

- **Framework**: FastAPI with uvicorn
- **CORS**: Enabled for all origins (mobile + web)
- **File size limit**: 50 MB
- **Supported formats**: WAV, MP3, OGG, FLAC, M4A, AAC, WMA
- **Auto-docs**: Swagger at `/docs`, ReDoc at `/redoc`

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check + model status |
| GET | `/api/v1/models` | Loaded model info |
| GET | `/api/v1/classes` | Available Bol/Taal classes |
| POST | `/api/v1/predict` | Single Bol/Taal prediction |
| POST | `/api/v1/predict/sequence` | Full combined Bol sequence extraction |

### Inference Models Used
- **Bol**: ResidualCNN (best fold checkpoint)
- **Taal**: CRNN (best fold checkpoint)
- Auto-detects device: CUDA → MPS (Apple Silicon) → CPU

---

## 12. Web Interface

- **Single-page HTML** (~17KB) served at `http://localhost:8000`
- No build step — pure HTML/CSS/JS, served via FastAPI's StaticFiles

### Web Interface Features
| Feature | Description |
|---|---|
| Audio Upload | Drag & drop or click to select (WAV, MP3, OGG, FLAC, M4A) |
| Audio Playback | Built-in HTML5 audio player with full controls |
| Taal Detection | Displays detected Taal with confidence bar + theka pattern |
| Live Bol Timeline | Each detected Bol shown as a card, highlights in real-time during playback |
| Source Attribution | Color-coded: 🟢 Both agree, 🔵 Template only, 🟡 Model only |
| Cycle Display | Shows which Taal cycle each Bol belongs to (cycle 1, 2, 3...) |
| Click-to-Seek | Click any Bol card to jump audio playback to that timestamp |
| Processing Info | Shows processing time, Bol count, audio duration |

### Web Client-Server Flow
```
┌──────────────┐        ┌───────────────┐        ┌──────────────┐
│   Browser     │        │   FastAPI      │        │  ML Models   │
│   (index.html)│        │   (main.py)    │        │  (inference) │
│               │        │                │        │              │
│ 1. User drops │        │                │        │              │
│    audio file ├───────▶│ 2. Receive     │        │              │
│               │ POST   │    multipart   ├───────▶│ 3. Predict   │
│               │/predict│    form-data   │        │    combined  │
│               │/sequence│               │        │    sequence  │
│               │        │                │◀───────┤ 4. Return    │
│ 6. Render     │◀───────┤ 5. JSON resp   │        │    results   │
│    timeline   │        │                │        │              │
│    + playback │        │                │        │              │
└──────────────┘        └───────────────┘        └──────────────┘
```

---

## 13. Mobile App (React Native / Expo)

### Tech Stack
- **Framework**: React Native 0.81.5 with Expo SDK 54
- **Language**: TypeScript
- **Navigation**: React Navigation (bottom tabs + native stack)
- **UI**: LinearGradient, expo-blur (glassmorphism), expo-haptics
- **Audio**: expo-av (recording + playback)
- **File Picker**: expo-document-picker
- **Storage**: @react-native-async-storage (history, settings)
- **Animations**: react-native-reanimated

### Mobile App Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MOBILE APP (Expo/React Native)             │
│                                                              │
│  ┌─── Navigation (React Navigation) ────────────────────┐   │
│  │                                                       │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ HomeScreen  │  │AnalysisScreen│  │HistoryScreen │  │   │
│  │  │             │  │              │  │              │  │   │
│  │  │ • Record 🎙️ │  │ • Audio      │  │ • Saved      │  │   │
│  │  │   (max 30s) │  │   Player     │  │   results    │  │   │
│  │  │ • Upload 📁 │  │ • TaalCard   │  │ • Tap to     │  │   │
│  │  │ • How It    │  │ • BolTimeline│  │   re-view    │  │   │
│  │  │   Works     │  │ • Live sync  │  │              │  │   │
│  │  └─────┬──────┘  └──────┬───────┘  └──────────────┘  │   │
│  │        │                │                             │   │
│  │        │    navigate    │   API call                   │   │
│  │        └────────────────┘                             │   │
│  │                         │                             │   │
│  └─────────────────────────┼─────────────────────────────┘   │
│                            │                                 │
│  ┌─── Services Layer ──────┼─────────────────────────────┐   │
│  │                         ▼                             │   │
│  │  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │  api.ts       │  │ storage.ts   │                   │   │
│  │  │              │  │              │                   │   │
│  │  │ • predictSeq │  │ • saveHistory│                   │   │
│  │  │ • healthCheck│  │ • loadHistory│                   │   │
│  │  │ • modelInfo  │  │ • clearAll   │                   │   │
│  │  └──────┬───────┘  └──────────────┘                   │   │
│  └─────────┼─────────────────────────────────────────────┘   │
│            │                                                 │
│            │  HTTP (multipart/form-data)                      │
│            ▼                                                 │
│  ┌──────────────────┐                                        │
│  │  FastAPI Backend   │  (configurable URL in Settings)       │
│  │  :8000             │                                       │
│  └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘
```

### Screens
| Screen | Purpose |
|---|---|
| HomeScreen | Record audio (max 30s) or upload file, "How It Works" guide |
| AnalysisScreen | Audio player + analyze button → TaalCard + BolTimeline with live sync |
| HistoryScreen | Past analysis results from AsyncStorage |
| SettingsScreen | Configure API base URL, dark/light theme toggle |

### Reusable Components
| Component | Purpose |
|---|---|
| GlassCard | Glassmorphism card wrapper (semi-transparent, blurred bg) |
| TaalCard | Displays detected Taal name, confidence %, theka, beats/vibhag info |
| BolTimeline | Live Bol sequence with current-time highlighting, click-to-seek, source color-coding |

### Key Mobile Features
- **Audio Recording**: expo-av with HIGH_QUALITY preset, max 30s, timer display
- **Audio Playback**: 80ms position update interval for smooth timeline sync
- **Haptic Feedback**: Impact/notification haptics on record, stop, analyze, seek
- **Dark/Light Theme**: System-wide ThemeContext with full color palette
- **History Persistence**: All analysis results auto-saved to AsyncStorage
- **Configurable Backend**: API URL configurable in Settings (default: `http://10.0.2.2:8000`)

---

## 14. Project Structure

```
TAL-Identifier/
├── src/                          # Core ML code (~3200 lines)
│   ├── config.py                 # All hyperparameters (dataclasses)
│   ├── dataset.py                # Loading, preprocessing, augmentation
│   ├── models.py                 # 4 architectures + registry
│   ├── trainer.py                # Training loop, Mixup, early stopping, LR
│   ├── inference.py              # Production inference + combined prediction
│   └── utils.py                  # Visualization & plotting
├── api/                          # FastAPI backend
│   ├── main.py                   # REST endpoints, CORS, file handling
│   └── static/index.html         # Web demo
├── mobile/                       # React Native (Expo) app
│   ├── src/screens/              # 4 screens
│   ├── src/components/           # 3 reusable components
│   ├── src/services/             # API client + storage
│   └── src/theme/                # Dark/light theming
├── train.py                      # Training entry point (CLI)
├── results/                      # Generated outputs
│   ├── bol/models/               # 20 checkpoints (4×5 folds)
│   ├── taal/models/              # 20 checkpoints
│   └── *.png, *.json, *.csv      # Plots + metrics
├── Tabla-Taal-and-Bol-Identification/Dataset/  # Audio dataset
├── requirements.txt              # Python deps
└── Tarang_Report.pdf             # Prior work reference
```

---

## 15. Dependencies

### Python
torch, torchaudio, librosa, scikit-learn, numpy, pandas, matplotlib, seaborn, tqdm, soundfile, fastapi, uvicorn, python-multipart

### Mobile (npm)
expo (SDK 54), react-native (0.81.5), expo-av, expo-blur, expo-haptics, expo-document-picker, expo-linear-gradient, react-navigation, react-native-reanimated, async-storage, typescript

---

## 16. Key Technical Decisions

1. **Why Mel Spectrograms?** — Captures both spectral (timbre) and temporal (rhythm) info; mirrors human auditory perception via mel scale
2. **Why 4 architectures?** — CNN for spatial features, CRNN for temporal, Attention for focus, Transformer for global context
3. **Why 3-layer augmentation?** — Only 100 samples/class; need maximum diversity without collecting more data
4. **Why Combined Inference?** — Template provides domain knowledge (Taal structure), model provides acoustic analysis; cross-validation gives best of both
5. **Why CRNN for Taal?** — Taals are defined by temporal beat patterns; BiLSTM captures long-range dependencies
6. **Why ResidualCNN for Bol?** — Individual strokes are characterized by spectral features; CNN excels at spatial pattern recognition
7. **Why Label Smoothing?** — Prevents overconfident predictions on small dataset; improves generalization
8. **Why Cosine Warmup?** — Stable start (warmup prevents large early updates), smooth decay avoids sharp LR drops
