# 🔐 Evaluating CAPTCHA Distortions with Deep Learning

> **Can modern OCR systems beat the CAPTCHA?**

This project investigates the robustness of deep learning–based Optical Character Recognition (OCR) against synthetic CAPTCHA-style distortions. Using a **ResNet CNN + Transformer** architecture trained end-to-end with CTC loss on 113,000+ images, we evaluate how different distortion types and severity levels impact machine text recognition performance.

---

## 📋 Table of Contents

- [Research Objectives](#research-objectives)
- [Distortion Benchmarking](#distortion-benchmarking)
- [Project Structure](#project-structure)
- [Versions Overview](#versions-overview)
- [Available Models](#available-models)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [CTC Decoding](#ctc-decoding)
- [Performance Optimizations](#performance-optimizations)
- [Data Augmentation](#data-augmentation)
- [Ethical Use](#ethical-use)
- [Authors](#authors)
- [License](#license)

---

## 🎯 Research Objectives

This project aims to answer the following questions:

- Which CAPTCHA-style distortions most effectively reduce OCR accuracy?
- Which distortions impact humans more than machines? *(qualitative assessment, if time allows)*
- How does model accuracy degrade as distortion severity increases?
- Can modern OCR architectures reliably solve heavily distorted synthetic CAPTCHAs?

---

## 🌀 Distortion Benchmarking

Synthetic CAPTCHA images are generated with controlled distortions, including:

- Rotation
- Gaussian blur
- Noise injection
- Character overlap
- Perspective warping
- Background clutter (lines, dots, etc.)
- Occlusion lines (foreground)
- Font variation *(including potential use cases for old German fonts)*

---

## 📁 Project Structure

```
Deep_Learning_Captcha/
│
├── .github/
│   └── workflows/
│       └── ci.yml                             # CI/CD pipeline
│
├── src/
│   ├── Deep_Learning_Captcha.py               # Training script (GPU / high-performance)
│   ├── Read_Captcha_Traind_Modell.py          # Inference script
│   ├── classify.py                            # Classification (standard version)
│   ├── classify_optimized.py                  # Classification (RAM-optimized, for powerful PCs)
│   └── generate_synthetic_captcha/
│       └── distortions/
│           ├── __init__.py
│           ├── levels.py                      # Level 1/2/3 parameter definitions
│           ├── pipeline.py                    # Reusable DistortionPipeline class
│           ├── generate_distorted_dataset.py  # Batch CLI generator
│           ├── preview_distortions.py         # Visual preview generator
│           └── README.md
│
├── models/
│   ├── Deep_2_1.pth                           # CAPTCHA model v2.1 (ResNet + Transformer)
│   ├── ...                                    # Further pre-trained models (coming soon)
│   └── README_models.md                       # Description of all available models
│
├── plots/
│   └── training_results_epoch20_acc93.png     # Training results visualization
│
├── tests/
│   └── Lin_Modell_new.png                     # Test image
│
├── README.md
├── LICENSE
├── requirements.txt                           # GPU version dependencies
├── requirements_cpu.txt                       # CPU version dependencies (coming soon)
└── environment.yml                            # Conda environment (GPU)
```

---

## 🖥️ Versions Overview

| Version | Hardware | Description | Status |
|---------|----------|-------------|--------|
| **GPU / High-performance** | Powerful PC + NVIDIA GPU | Full model, fast training & inference | ✅ Available |
| **CPU / Lightweight** | Any PC, no GPU needed | Optimized for weak hardware | 🔜 Coming soon |

---

## 🧠 Available Models

| Model | Task | Architecture | Accuracy | Dataset | Status |
|-------|------|-------------|----------|---------|--------|
| `Deep_2_1.pth` | CAPTCHA recognition | ResNet + Transformer | 93% Word / 99% Char | [CAPTCHA Dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset) | ✅ Available |
| Handwriting model | Handwriting recognition | TBD | TBD | TBD | 🔜 Coming soon |
| Lightweight model | CAPTCHA (weak PC) | TBD | TBD | TBD | 🔜 Coming soon |

> All models can be used with the classification scripts (`classify.py` / `classify_optimized.py`)

---

## 📊 Model Architecture (ResNet + Transformer)

| Step | Layer | Details |
|------|-------|---------|
| 1 | **Input** | 160 × 48 px grayscale image |
| 2 | **ResNet CNN backbone** | 4 ResBlocks (stride-2) · 32 → 64 → 128 → 256 channels |
| 3 | **Linear projection** | Feature map → d_model = 256 |
| 4 | **Positional encoding** | Sinusoidal · adds position info to sequence |
| 5 | **Transformer encoder** | 4 layers · 8 attention heads · Pre-LN · FFN dim 1024 |
| 6 | **CTC head** | 62 classes (a–z, A–Z, 0–9) + blank token |
| 7 | **Decoding** | Greedy (fast) or Beam Search width=5 (accurate) |
| 8 | **Output** | Predicted text string |

### Model Hyperparameters

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

## 🔄 Training Pipeline

```
Chars74K (character-level backbone pretraining)
    ↓
CNN backbone pretraining
    ↓
Synthetic CAPTCHA sequence training (controlled distortions)
    ↓
Distortion robustness evaluation
    ↓
Real CAPTCHA generalization testing
```

---

## 📦 Dataset

**[CAPTCHA Dataset – Kaggle](https://www.kaggle.com/datasets/parsasam/captcha-dataset)**

| Split | Size |
|-------|------|
| Train | 80% (~90,400 images) |
| Validation | 10% (~11,300 images) |
| Test | 10% (~11,300 images) |
| **Total** | **~113,000 images** |

---

## 📈 Results

| Metric | Greedy Decoding | Beam Search (width=5) |
|--------|-----------------|-----------------------|
| Word Accuracy | 93% | TBD |
| Char Accuracy | 99% | TBD |

![Training Results](plots/training_results_epoch20_acc93.png)

### Evaluation Metrics

- Character Error Rate (CER)
- Sequence Accuracy
- Accuracy vs. Distortion Severity
- Human vs. Machine Accuracy
- Inference Confidence Scores

---

## ⚙️ Setup & Usage

### 1. Install Dependencies

**With GPU (NVIDIA CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Without GPU (CPU only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**With Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate torch_gpu
```

### 2. Download Dataset

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('parsasam/captcha-dataset')"
```

### 3. Training

```bash
python src/Deep_Learning_Captcha.py
```

Training outputs:
- Live plot (Loss, Accuracy, Learning Rate)
- Best model saved as `models/best_captcha_model.pth`
- Per-epoch console output with example predictions

### 4. Inference / Classification

**Standard version** (works on any PC):
```bash
python src/classify.py
```

**Optimized version** (loads all data into RAM — for powerful PCs):
```bash
python src/classify_optimized.py
```

Both scripts support two modes:
- **Mode 1** — Single image with visualization
- **Mode 2** — Batch prediction on an entire folder

---

## 🔤 CTC Decoding

**Greedy Decoding**
Fast — picks the most likely character at each timestep. Ideal for batch evaluation.

**Beam Search (width=5)**
Explores multiple paths simultaneously. Slightly more accurate; used for single-image inference.

---

## ⚡ Performance Optimizations

| Optimization | Description |
|---|---|
| AMP | float16 mixed precision — reduces memory, speeds up training |
| RAM preloading | Entire dataset loaded into RAM before training (optimized version) |
| pin_memory | Fast CPU→GPU data transfers |
| cudnn.benchmark | Optimized CUDA kernels for the current hardware |

---

## 💾 Data Augmentation

| Augmentation | Parameter |
|---|---|
| ColorJitter Brightness | ±0.2 |
| ColorJitter Contrast | ±0.2 |
| ColorJitter Saturation | ±0.1 |

---

## ⚖️ Ethical Use

This project is intended for **educational and research purposes only**. All CAPTCHA images used are synthetically generated and are not intended to bypass real-world security systems.

---

## 👥 Authors

| Name | Role |
|------|------|
| **Benedikt** | Model Architecture, Training Pipeline |
| **Luka** | Data Processing, Inference & Evaluation |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
