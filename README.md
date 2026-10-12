# Tacotron2 + HiFi-GAN: End-to-End Neural Text-to-Speech (TTS)

A modular, production-ready PyTorch implementation of an end-to-end Text-To-Speech pipeline combining **Tacotron2** (Autoregressive Mel-Spectrogram Generation) and **HiFi-GAN** (High-Fidelity Neural Vocoder).

---

## 🏛️ System Architecture

```text
[Input Text] 
     │
     ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Tacotron2 (src/tacotron2)                     │
│  - Character Embedding + Conv1D + BiLSTM Encoder       │
│  - Location-Sensitive Attention Mechanism             │
│  - 2-Layer LSTM Decoder with PreNet Dropout            │
│  - 5-Layer Conv1D PostNet (Residual Refinement)        │
└────────────────────────────────────────────────────────┘
     │
     ▼  [Mel Spectrogram (80 bins, 22.05 kHz)]
┌────────────────────────────────────────────────────────┐
│ Stage 2: HiFi-GAN Vocoder (src/hifigan)                │
│  - Transposed Convolutions (Upsample 256x: 8x8x2x2)    │
│  - Multi-Receptive Field Fusion (MRF ResBlocks)        │
│  - Multi-Period Discriminator (MPD)                    │
│  - Multi-Scale Discriminator (MSD)                     │
└────────────────────────────────────────────────────────┘
     │
     ▼
[Output Audio Waveform (.wav)]
```

---

## 📁 Project Structure

```text
Tacotron2/
├── pyproject.toml              # Modern package configuration (pip install -e .)
├── requirements.txt            # Unified project dependencies
├── README.md                   # Complete system documentation
│
├── src/                        # Core reusable Python library
│   ├── common/                 # Shared utilities
│   │   ├── audio.py            # AudioMelConversions, STFT, DB & norm scaling
│   │   └── tokenizer.py        # Character-level vocabulary & encoding
│   ├── tacotron2/              # Stage 1: Text to Mel-spectrogram
│   │   ├── model.py            # Tacotron2 model architecture & Tacotron2Config
│   │   └── dataset.py          # LJSpeech TTSDataset, TTSCollator, BatchSampler
│   └── hifigan/                # Stage 2: Mel-spectrogram to Audio
│       ├── model.py            # Generator, MPD, MSD architectures & HIFIGANConfig
│       ├── dataset.py          # MelDataset (pre-training & fine-tuning modes)
│       └── loss.py             # Feature loss, generator loss, discriminator loss
│
├── scripts/                    # Command-line training & execution tools
│   ├── prep_splits.py          # Prepare LJSpeech train/val metadata splits
│   ├── train_tacotron2.py      # Distributed training for Tacotron2
│   ├── train_hifigan.py        # Distributed training / fine-tuning for HiFi-GAN
│   └── save_taco_mels.py       # Cache teacher-forced Tacotron2 mels for vocoder fine-tuning
│
├── configs/                    # Ready-to-run execution scripts (.sh & .ps1)
│   ├── train_tacotron2.sh / .ps1
│   ├── train_hifigan.sh / .ps1
│   └── finetune_hifigan.sh / .ps1
│
├── notebooks/                  # Interactive Jupyter notebooks for inference
│   ├── tacotron2_inference.ipynb
│   └── hifigan_inference.ipynb
│
└── data/                       # Dataset manifests & cached spectrograms (local)
    ├── train_metadata.csv
    └── test_metadata.csv
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup

Using `uv` (recommended, high speed):
```bash
uv pip install -e .
```

Or standard `pip`:
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🎙️ Data Preparation

Download the [LJSpeech-1.1 Dataset](https://keithito.com/LJ-Speech-Dataset/) (~2.6 GB, 24 hours of single-speaker audio):

```bash
# Generate train and test metadata splits
python scripts/prep_splits.py \
    --path_to_ljspeech /path/to/LJSpeech-1.1 \
    --path_to_save data/ \
    --seed 42 \
    --test_split_pct 0.01
```

> **Note on Spectrogram Scaling**: Mel-spectrograms are converted to decibels (-100 dB to 0 dB) and normalized between `[-4.0, 4.0]`. This zero-centers the feature distributions and stabilizes training dynamics.

---

## 🚀 Training Workflow

### Stage 1: Train Tacotron2
Trains the sequence-to-sequence model to learn the alignment between text and Mel-spectrograms:

```bash
# On Linux / Mac:
bash configs/train_tacotron2.sh

# On Windows (PowerShell):
./configs/train_tacotron2.ps1
```

### Stage 2: Pretrain HiFi-GAN
Pretrains the neural vocoder on ground truth audio Mel-spectrograms:

```bash
# On Linux / Mac:
bash configs/train_hifigan.sh

# On Windows (PowerShell):
./configs/train_hifigan.ps1
```

### Stage 3: Fine-tune HiFi-GAN (Domain Adaptation)
To achieve natural, artifact-free speech, fine-tune HiFi-GAN directly on Tacotron2-generated spectrograms:

1. **Cache Tacotron2 Mel-spectrograms:**
   ```bash
   python scripts/save_taco_mels.py \
       --path_to_manifest data/train_metadata.csv data/test_metadata.csv \
       --path_to_save data/taco_gen_mels \
       --taco_weights work_dir/tacotron2/checkpoint_250/model.safetensors
   ```

2. **Fine-tune HiFi-GAN:**
   ```bash
   bash configs/finetune_hifigan.sh
   ```

---

## 📊 Quality Evaluation (UTMOS)

| Vocoder Method | Quality Description | Mean Opinion Score (UTMOS) |
| :--- | :--- | :---: |
| **Griffin-Lim** | Fast iterative signal reconstruction (baseline) | 1.88 |
| **HiFi-GAN (Pretrained)** | Deep neural upsampling | 3.37 |
| **HiFi-GAN (Fine-tuned)** | Domain-adapted on Tacotron2 outputs | **3.74** |

---

## 🔍 Interactive Inference

Open either of the interactive notebooks in `notebooks/`:
- `notebooks/tacotron2_inference.ipynb`: Explore attention alignment heatmaps and Griffin-Lim audio output.
- `notebooks/hifigan_inference.ipynb`: Run full end-to-end Text-to-Speech synthesis with HiFi-GAN.