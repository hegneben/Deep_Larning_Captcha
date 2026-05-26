# Evaluating CAPTCHA Distortions with Deep Learning

> **Can modern OCR systems beat the CAPTCHA?**

This project investigates the robustness of deep learning–based Optical Character Recognition (OCR) against synthetic CAPTCHA-style distortions. Using a **ResNet CNN + Transformer** architecture trained end-to-end with CTC loss on 113,000+ real CAPTCHA images, we systematically evaluate how different distortion types and severity levels impact machine text recognition — and compare these findings to the broader OCR research landscape.

---

## Table of Contents

- [Research Objectives](#research-objectives)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [CTC Decoding](#ctc-decoding)
- [Performance Optimizations](#performance-optimizations)
- [Data Augmentation](#data-augmentation)
- [Further Work](#further-work)
- [Ethical Use](#ethical-use)
- [Authors](#authors)
- [License](#license)

---

## Research Objectives

This project answers four core research questions:

1. Which CAPTCHA-style distortions most effectively reduce OCR accuracy when applied in isolation?
2. Do combined distortions overwhelm the model faster than individual ones applied progressively?
3. Which single distortion type is most destructive in isolation?
4. How does accuracy degrade as CAPTCHA-like complexity increases step by step?

---

## Key Findings

| Finding | Result |
|---|---|
| **Architecture upgrade** | Transformer adds +11.8pp word accuracy over CNN baseline (75% → 86.82%), verified on 11,307 held-out test images |
| **Combined distortions** | Applying all distortions simultaneously at mild level drops word accuracy to **0%** — far worse than any single distortion alone |
| **Most destructive isolation** | Occlusion (0% at 5 blocks) > Blur (3% at radius 2.0) > Lines (13.5%) > Rotation (36%) > Noise (~57%, essentially irrelevant) |
| **Realistic CAPTCHA threats** | Font variation (−16.7pp) and character spacing (−16.2pp) each cause equal damage to the model while remaining imperceptible burdens to human users |
| **Domain gap** | Model trained on Gregwar/Captcha-style images achieves only 52% word accuracy on clean Arial synthetic images — revealing strong renderer-specific overfitting |

---

## Project Structure

```
Deep_Larning_Captcha/
│
├── .github/
│   └── workflows/
│       └── ci.yml                              # CI/CD pipeline
│
├── src/
│   ├── Deep_Learning_Captcha.py                # Main training script (ResNet + Transformer)
│   ├── captcha_recognition_Linear_Model.py     # CNN baseline training script
│   ├── deployment/
│   │   ├── Read_Captcha_Traind_Modell.py       # Inference script (CAPTCHA model)
│   │   └── read_handwriting_Traind_Modell.py   # Inference script (handwriting model)
│   └── generate_synthetic_captcha/
│       ├── distortions/
│       │   ├── __init__.py
│       │   ├── levels.py                       # Level 1/2/3 parameter definitions
│       │   ├── pipeline.py                     # Reusable DistortionPipeline class
│       │   ├── generate_distorted_dataset.py   # Batch CLI generator
│       │   └── preview_distortions.py          # Visual preview generator
│       └── experimental_ladder/
│           ├── ladder_config.py                # 7-level ladder configuration
│           ├── ladder_renderer.py              # Per-character rendering engine
│           └── generate_ladder_dataset.py      # Ladder dataset generator
│
├── models/
│   ├── new_Models/
│   │   ├── model_Transformer_new.pth               # ResNet+Transformer (best, ~86.82% word acc)
│   │   ├── model_Transformer_new_78acc.pth          # ResNet+Transformer (earlier run, ~78% word acc)
│   │   └── model_Transformer_new_78acc_state_dict.pth  # State dict version (used for inference)
│   ├── Lin_Model_CAPTCHA_new.pth               # CNN baseline model (~75% word acc)
│   └── README_models.md
│
├── data/
│   └── synthetic/
│       ├── clean/                              # 1,000 clean Arial synthetic CAPTCHAs + labels.csv
│       ├── distorted/                          # 3,000 distorted images (levels 1–3) + metadata.csv
│       └── experimental_ladder/               # 7,000 ladder images (7 levels × 1,000) + metadata.csv
│
├── results/
│   ├── distortion_benchmark_results.csv        # E2: combined distortion benchmark results
│   ├── ladder_results.csv                      # E3: experimental ladder results
│   └── per_distortion_results.csv             # E4: isolation study results
│
├── plots/
│   ├── Lin_Modell_new.png                      # CNN baseline training dashboard
│   ├── matplotlib_TCaptcha_Transformer_new.png # Transformer v2 training dashboard (86.82%)
│   └── matplotlib_Deep_Learning_Capcha_1.png   # Transformer v1 training dashboard (78%)
│
├── evaluate_model.py                           # Model verification on Kaggle test split
├── evaluate_distortion_benchmark.py            # E2 evaluation script
├── evaluate_ladder.py                          # E3 evaluation script
├── evaluate_per_distortion.py                  # E4 evaluation script
├── generate_visual_dataset.py                  # Visual demo image generator
├── README.md
├── LICENSE
├── requirements.txt                            # GPU version dependencies
├── requirements_cpu.txt                        # CPU version dependencies
└── environment.yml                             # Conda environment (torch-env)
```

---

## Model Architecture (ResNet + Transformer)

| Step | Component | Details |
|------|-----------|---------|
| 1 | **Input** | 160 × 48 px RGB image |
| 2 | **ResNet CNN backbone** | 4 ResBlocks (stride-2) · 32 → 64 → 128 → 256 channels |
| 3 | **Linear projection** | Flattens each column (256×6) → d_model = 256 |
| 4 | **Positional encoding** | Sinusoidal · adds unique position fingerprint to each column |
| 5 | **Transformer encoder** | 4 layers · 8 heads · bidirectional attention · Pre-LN · FFN=1024 |
| 6 | **CTC head** | 62 classes (a–z, A–Z, 0–9) + blank token (index 62) |
| 7 | **Decoding** | Greedy (training/batch eval) or Beam Search width=5 (single inference) |
| 8 | **Output** | Predicted 5-character string |

### Architectural Context

This model follows the CRNN → Transformer evolution in OCR research:

| Era | Architecture | Analogue |
|---|---|---|
| 2006 | Tesseract (hand-crafted segmentation + SVM) | Pre-deep learning baseline |
| 2015 | CRNN: CNN + BiLSTM + CTC (Shi et al.) | Our CNN baseline |
| 2024 | **This project**: CNN + Transformer + CTC | One step from state-of-the-art |
| 2021 | TrOCR: ViT + GPT decoder + pretraining (Microsoft) | Next step for further work |

### Model Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 256 | Balances capacity vs. overfitting risk on 113K images |
| Attention heads | 8 | Standard for d_model=256 (32 dims per head) |
| Transformer layers | 4 | Sufficient depth without overfitting small dataset |
| Feedforward dim | 1024 | 4× d_model — standard Transformer ratio |
| Optimizer | AdamW (lr=1e-3, decay=1e-4) | Weight decay prevents overfitting |
| Scheduler | OneCycleLR · 10% warmup | Stable convergence with fast early learning |
| Loss | CTC Loss · blank index=62 | No segmentation required |
| Batch size | 128 | GPU memory / gradient stability tradeoff |
| Epochs | 20 | Convergence observed by epoch 15–17 |
| Precision | AMP float16 + grad clip=5.0 | Memory efficiency + training stability |
| Image size | 160 × 48 px RGB | Preserves character aspect ratio |
| Charset | a–z, A–Z, 0–9 (62 classes) | Case-sensitive — critical fix from v1 |

### Model Versions

| Model | Word Acc | Char Acc | Key difference |
|-------|----------|----------|----------------|
| CNN baseline (`Lin_Model_CAPTCHA_new.pth`) | ~75% | ~90% | No Transformer — FC head only |
| Transformer v1 (`model_Transformer_new_78acc.pth`) | ~78% | ~91% | Case bug — `.lower()` normalisation applied |
| **Transformer v2** (`model_Transformer_new.pth`) | **86.82%** | **95.72%** | Case-sensitive · verified on 11,307 test images |

> All accuracy figures for Transformer v2 are verified by running inference on the held-out Kaggle test split (11,307 images).

---

## Training Pipeline

```
Kaggle CAPTCHA Dataset (113,000 images)
    ↓
Split: 80% train / 10% val / 10% test
    ↓  (split BEFORE augmentation — no data leakage)
ColorJitter augmentation (train only)
    ↓
End-to-end training: ResNet CNN → Projection → Transformer → CTC
    ↓
Best checkpoint saved (monitored on validation word accuracy)
    ↓
Distortion robustness evaluation (4 experiments)
```

> **Note on data leakage fix:** An earlier version of the pipeline applied augmentation before splitting. This was identified and corrected — the final model was trained with augmentation applied only to the training split.

---

## Dataset

**Training: [CAPTCHA Dataset – Kaggle (parsasam)](https://www.kaggle.com/datasets/parsasam/captcha-dataset)**

Generated using the [Gregwar/Captcha](https://github.com/Gregwar/Captcha) PHP library, which applies sinusoidal wave warping, per-character colour randomisation, background noise letters, and JPEG compression at quality 80. This renderer-specific style is a key source of the domain gap observed in our synthetic evaluation.

| Split | Size |
|-------|------|
| Train | ~90,400 images |
| Validation | ~11,300 images |
| Test | ~11,300 images |
| **Total** | **~113,000 images** |

**Synthetic test sets (generated in-project):**

| Dataset | Images | Purpose |
|---------|--------|---------|
| Clean synthetic (Arial) | 1,000 | Domain gap baseline |
| Distorted benchmark (L1/L2/L3) | 3,000 | Combined distortion evaluation |
| Experimental ladder (L0–L6) | 7,000 | Additive distortion ladder |
| Per-distortion isolation | 5,000 | Single-distortion ablation (5 types × 5 levels × 200 images) |

---

## Experimental Design

Four experiments were conducted, each answering a distinct research question:

### E1 — Model Benchmarking
CNN baseline vs. Transformer v1 vs. Transformer v2, evaluated on the Kaggle held-out test split. Establishes the benefit of the Transformer upgrade and the case-sensitivity fix.

### E2 — Combined Distortion Benchmark
1,000 clean synthetic Arial images distorted at three combined severity levels:

| Level | Name | Key parameters |
|-------|------|---------------|
| L0 | Clean | No distortion — domain gap baseline |
| L1 | Mild | ±4° rotation · blur 0.45 · 60 dots · 2 lines |
| L2 | Moderate | ±9° rotation · blur 1.1 · 180 dots · 5 lines · 1 occlusion |
| L3 | Severe | ±16° rotation · blur 1.8 · 360 dots · 9 lines · 3 occlusions |

### E3 — Experimental Ladder
7,000 images (1,000 per level) built from scratch with distortions added one at a time:

| Level | Cumulative distortions | Fonts used |
|-------|----------------------|------------|
| L0 | Clean centered text | Arial only |
| L1 | + Random character spacing | Arial only |
| L2 | + Vertical per-character jitter | Arial only |
| L3 | + Font variation | 31 diverse fonts |
| L4 | + Character rotation ±12° | 31 diverse fonts |
| L5 | + Mild noise + 1 occlusion line + blur | 31 diverse fonts |
| L6 | + Heavy noise + 3 lines + stronger blur | 31 diverse fonts |

### E4 — Per-Distortion Isolation
5 distortion types tested independently across 5 intensity levels, 200 images per level:

| Distortion | Intensity levels tested |
|-----------|------------------------|
| Blur | radius: 0.3, 0.7, 1.1, 1.5, 2.0 |
| Noise | std: 5, 10, 18, 28, 40 |
| Rotation | degrees: ±3, ±6, ±10, ±14, ±18 |
| Lines | count: 1, 3, 5, 7, 10 |
| Occlusion | blocks: 1, 2, 3, 4, 5 |

---

## Results

### Experiment 1 — Model Benchmarking (Kaggle test split, 11,307 images)

| Model | Word Accuracy | Char Accuracy |
|-------|--------------|---------------|
| CNN baseline | ~75% | ~90% |
| Transformer v1 (case bug) | ~78% | ~91% |
| **Transformer v2 (verified)** | **86.82%** | **95.72%** |

> CNN baseline accuracy read from training plot. Transformer v2 verified by direct inference on held-out test split.

### Experiment 2 — Combined Distortion Benchmark

| Level | Word Accuracy | Char Accuracy | Interpretation |
|-------|--------------|---------------|----------------|
| L0 Clean | 52.10% | 87.02% | Domain gap — model learned Gregwar style, not Arial |
| L1 Mild | 0.00% | 0.27% | Complete collapse — combined distortions overwhelm simultaneously |
| L2 Moderate | 0.00% | 0.07% | No recovery beyond mild |
| L3 Severe | 0.00% | 0.17% | Marginal char-level signal only |

### Experiment 3 — Experimental Ladder

| Level | Description | Word Accuracy | Char Accuracy | Drop vs previous |
|-------|-------------|--------------|---------------|-----------------|
| L0 | Clean | 54.30% | 87.62% | — |
| L1 | + Spacing | 38.10% | 75.46% | −16.2pp ⚠️ |
| L2 | + Jitter | 35.60% | 75.12% | −2.5pp |
| L3 | + Font variation | 18.90% | 58.96% | −16.7pp ⚠️ |
| L4 | + Rotation | 15.10% | 54.78% | −3.8pp |
| L5 | + Mild noise/lines | 13.40% | 52.52% | −1.7pp |
| L6 | + Heavy noise/lines | 8.20% | 49.08% | −5.2pp |

> Spacing and font variation each cause ~16pp drops — equal in damage and together accounting for the majority of degradation. Noise alone causes essentially no damage.

### Experiment 4 — Per-Distortion Isolation (word accuracy at max intensity)

| Rank | Distortion | Max intensity | Word Accuracy | Behaviour |
|------|-----------|---------------|---------------|-----------|
| 1 | **Occlusion** | 5 blocks | **0.0%** | Catastrophic from block 1 — information destroyed |
| 2 | **Blur** | radius 2.0 | **3.0%** | Sharp cliff between radius 1.5→2.0 |
| 3 | **Lines** | 10 lines | **13.5%** | Steady linear degradation |
| 4 | **Rotation** | ±18° | **36.0%** | Gradual — some training tolerance |
| 5 | **Noise** | std 40 | **~57%** | Essentially no effect — irrelevant defense |

---

## Setup & Usage

### 1. Clone and set up environment

```bash
git clone https://github.com/hegneben/Deep_Larning_Captcha.git
cd Deep_Larning_Captcha

# With Conda (recommended)
conda env create -f environment.yml
conda activate torch-env
```

### 2. Download dataset

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('parsasam/captcha-dataset')"
```

### 3. Training

```bash
python src/Deep_Learning_Captcha.py
```

### 4. Inference

```bash
# Single image or batch folder
python src/deployment/Read_Captcha_Traind_Modell.py
```

### 5. Verify model accuracy

```bash
# Runs inference on the held-out Kaggle test split
python evaluate_model.py \
  --model "models/new_Models/model_Transformer_new_78acc_state_dict.pth" \
  --dataset "path/to/kaggle/captcha/dataset"
```

### 6. Run distortion experiments

```bash
# E2: Combined distortion benchmark
python evaluate_distortion_benchmark.py \
  --model "models/new_Models/model_Transformer_new_78acc_state_dict.pth" \
  --data "data/synthetic"

# E3: Experimental ladder
python evaluate_ladder.py \
  --model "models/new_Models/model_Transformer_new_78acc_state_dict.pth" \
  --ladder "data/synthetic/experimental_ladder"

# E4: Per-distortion isolation
python evaluate_per_distortion.py \
  --model "models/new_Models/model_Transformer_new_78acc_state_dict.pth" \
  --clean "data/synthetic/clean" \
  --output "data/synthetic/per_distortion"
```

### 7. Generate visual demo images

```bash
# Outputs to Desktop/CAPTCHA_Visual_Demo/
python generate_visual_dataset.py \
  --clean "data/synthetic/clean"
```

---

## CTC Decoding

**Greedy Decoding**
Fast — picks the most likely character at each timestep independently. Used during training and batch evaluation.

**Beam Search (width=5)**
Explores 5 candidate paths simultaneously and picks the globally best sequence. Used for single-image inference. Marginally more accurate than greedy on ambiguous inputs.

**Why CTC?**
CTC eliminates the need for character-level segmentation — the model never needs to know exactly where each character is in the image. It learns the alignment between visual features and character outputs automatically during training. This is both its strength and a source of vulnerability: random character spacing (E3, L1) disrupts the temporal alignment CTC implicitly relies on, causing the largest single accuracy drop observed in the ladder experiment.

---

## Performance Optimizations

| Optimization | Description |
|---|---|
| AMP float16 | Mixed precision — reduces GPU memory, speeds up training |
| RAM preloading | Entire dataset loaded into RAM before training |
| pin_memory | Fast CPU→GPU data transfer |
| cudnn.benchmark | Optimised CUDA kernels for current hardware |
| Gradient clipping | max norm 5.0 — prevents loss spikes during OneCycleLR peak |

---

## Data Augmentation

Applied to training split only. Split performed before augmentation to prevent data leakage.

| Augmentation | Parameter | Purpose |
|---|---|---|
| ColorJitter Brightness | ±0.2 | Robustness to lighting variation |
| ColorJitter Contrast | ±0.2 | Robustness to contrast variation |
| ColorJitter Saturation | ±0.1 | Robustness to colour saturation shifts |

> Note: No geometric augmentation (rotation, blur, warp) was applied during training. This is a significant contributor to the model's poor distortion robustness on the synthetic test sets — adding distortion augmentation during training is the highest-priority improvement identified by this project.

---

## Further Work

Based on experimental findings, the following upgrades are prioritised by expected impact:

| Priority | Upgrade | Addresses | Expected impact |
|----------|---------|-----------|----------------|
| 1 | **Fine-tune on synthetic data** | Domain gap (52% clean) | Clean accuracy 52% → ~85% |
| 2 | **Pretrain on MJSynth/SynthText** | Font invariance (−16.7pp) | Font drop −16.7pp → ~−3pp |
| 3 | **Distortion augmentation in training** | Blur/rotation/noise robustness | Moderate improvement across E4 |
| 4 | **Replace CTC with attention decoder** | Spacing vulnerability (−16.2pp) | Spacing drop largely eliminated |
| 5 | **Replace ResNet with CLIP encoder** | All distributional shift issues | Transformative — font + domain gap solved |
| 6 | **Generative/inferential architecture** | Occlusion (0% at 5 blocks) | Partial recovery — hard open problem |

> Upgrades 1–3 are achievable with the existing architecture. Upgrade 4 requires replacing the CTC head with an attention decoder (see PARSeq, 2022). Upgrade 5 uses a CLIP vision encoder as a pretrained backbone (see TrOCR, Microsoft 2021). Upgrade 6 requires a fundamentally different model class (see George et al., Science 2017).

---

## Ethical Use

This project is intended for **educational and research purposes only**. All CAPTCHA images used in distortion evaluation are synthetically generated. The project does not attempt to bypass real-world deployed CAPTCHA systems. The training dataset consists of publicly available synthetic CAPTCHAs from Kaggle.

---

## Authors

| Name | Role |
|------|------|
| **Benedikt** | Model architecture, training pipeline, hyperparameter tuning |
| **Luka** | Data processing, synthetic dataset generation, distortion evaluation pipeline |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.