# Evaluating CAPTCHA Distortions with Deep Learning: Can we beat the CAPTCHA?

This project's goal is to investigate the robustness of modern deep learning Optical Character Recognition (OCR) systems againts synthetic CAPTCHA-like distortions.
Using a **ResNet CNN + Transformer** architecture trained with CTC loss, we evaluate how different distortion types and severity levels impact the machine's text recognition performance, and compare these results to 
A high-performance CAPTCHA recognition system using a **ResNet CNN + Transformer** architecture, trained end-to-end with CTC loss on 113,000+ images.

## Research Objectives

This project aims to answer the following questions:

- Which CAPTCHA-style distortions most effectively reduce OCR performance?
- Which distortions impact humans more than machines (Qualitative Assessemnt with class if time allows)?
- Hows does the model accuracy degrade as distortion severity increases?
- Can modern OCR architectures reliably solve heavily distorted synthetic CAPTCHAs?

## Distortion Benchmarking

Synthetic CAPTCHA images are generated with controlled distortions, including:

- Rotation
- Gaussian blur
- Noise injection
- Character overlap
- Perpective warping
- Background clutter (ex. lines, dots etc.)
- Occlusion lines (foreground)
- Font variation (consider use cases for old German font?)


## Authors

| Name | Role |
|------|------|
| **Benedikt** | Model Architecture, Training Pipeline |
| **Luka** | Data Processing, Inference & Evaluation |

---

## Project Structure

A CAPTCHA recognition system using a **ResNet CNN + Transformer** architecture trained with CTC loss.

Deep_Larning_Captcha/
│
├── train.py          # Training script
├── inference.py      # Inference script (single image & batch)
├── README.md         # This file
└── .gitignore        # Excludes model weights & cache


Chars74K (Character-level backbone for pretraining)
    ↓
CNN backbone pretraining
    ↓ 
Synthetic CAPTCHA sequence training (use this for controlled distortions)
    ↓
Distortion robustness evaluation
    ↓
Real CAPTCHA generalization testing

## Architecture
Input Image (160 × 48 px)
│
▼
┌───────────────────┐
│  ResNet CNN       │  4 ResBlocks (stride-2)
│  Backbone         │  32 → 64 → 128 → 256 channels
└───────────────────┘
│
▼
┌───────────────────┐
│  Linear           │  Feature projection → d_model=256
│  Projection       │
└───────────────────┘
│
▼
┌───────────────────┐
│  Positional       │  Sinusoidal encoding
│  Encoding         │
└───────────────────┘
│
▼
┌───────────────────┐
│  Transformer      │  4 Layers, 8 Heads, Pre-LN
│  Encoder          │  dim_feedforward = 1024
└───────────────────┘
│
▼
┌───────────────────┐
│  CTC Head         │  62 classes + blank
└───────────────────┘
│
▼
Greedy / Beam Search Decoding
│
▼
Predicted Tex

---

## Model Details

| Parameter | Value |
|-----------|-------|
| CNN Backbone | ResNet (4 ResBlocks) |
| Sequence Model | Transformer Encoder |
| d_model | 256 |
| Attention Heads | 8 |
| Transformer Layers | 4 |
| Feedforward Dim | 1024 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Loss Function | CTC Loss |
| Charset | a–z, A–Z, 0–9 (62 classes) |
| Blank Index | 62 |
| Image Size | 160 × 48 px |
| Batch Size | 128 |
| Epochs | 20 |
| AMP | float16 mixed precision |
| Gradient Clipping | max norm 5.0 |

---

## Dataset

**[CAPTCHA Dataset – Kaggle](https://www.kaggle.com/datasets/parsasam/captcha-dataset)**

| Split | Size |
|-------|------|
| Train | 80% (~90,400 images) |
| Validation | 10% (~11,300 images) |
| Test | 10% (~11,300 images) |
| **Total** | **~113,000 images** |

---

## Results

| Metric | Greedy Decoding | Beam Search (width=5) |
|--------|-----------------|-----------------------|
| Word Accuracy | ~XX% | ~XX% |
| Char Accuracy | ~XX% | ~XX% |

> Fill in your results after training.

---

### Evaluation Metrics

- Character Error Rate (CER)
- Sequence Accuracy
- Accuracy vs Distortion Severity
- Humna vs Machine Accuracy
- Inference Confidence Scores

## Setup & Usage

### 1. Install Dependencies

```bash
pip install torch torchvision pillow tqdm matplotlib
```

### 2. Download Dataset

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('parsasam/captcha-dataset')"
```

### 3. Start Training

```bash
python train.py
```

Training outputs:
- Live plot (Loss, Accuracy, Learning Rate)
- Best model saved as `best_captcha_model.pth`
- Per-epoch console output with example predictions

### 4. Run Inference

```bash
python inference.py
```

Two modes:
- **Mode 1** — Single image with visualization
- **Mode 2** — Batch prediction on an entire folder

---

## CTC Decoding

**Greedy Decoding**
Fast — picks the most likely character at each timestep. Ideal for batch evaluation.

**Beam Search (width=5)**
Explores multiple paths simultaneously. Slightly more accurate, used for single-image inference.

---

## Data Augmentation

| Augmentation | Parameter |
|---|---|
| ColorJitter Brightness | ±0.2 |
| ColorJitter Contrast | ±0.2 |
| ColorJitter Saturation | ±0.1 |

---

## Performance

- Entire dataset is loaded into RAM before training for maximum speed
- AMP (Automatic Mixed Precision) reduces memory usage and speeds up training
- `pin_memory=True` + `non_blocking=True` for fast CPU→GPU transfers
- `torch.backends.cudnn.benchmark = True` for optimized CUDA kernels

---

## .gitignore

## Ethical Use

This project is intended for educational and research purposes only. 
The CAPTCHA images are synthetically generated and are not intended to bypass real-world security systems.

## License

This project is for educational purposes only.

---
