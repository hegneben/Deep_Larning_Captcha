# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:54:15 2026

@author: Luka Ilisevic
"""

"""
CAPTCHA Model Evaluation Script
================================
Verifies the 78% Transformer model on the Kaggle CAPTCHA dataset.

No GPU required — runs on CPU, takes ~10-20 minutes for the full test split.

Usage:
    python evaluate_model.py --model path/to/model_Transformer_new_78acc_state_dict.pth
                             --dataset path/to/captcha-dataset

Optional flags:
    --samples 500       Only evaluate on N random samples (faster, ~2-3 min)
    --split 0.1         Fraction of dataset used as test set (default: 0.1 = 10%)
    --seed 42           Random seed for reproducible train/test split
    --save-errors       Save a CSV of incorrectly predicted samples
"""

import os
import math
import string
import random
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ─── MODEL ARCHITECTURE ──────────────────────────────────────────────────────
# Must match exactly what was used during training.

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CRNN_ResTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32,  64,  stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
        )
        self.proj    = nn.Linear(256 * 6, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes + 1)

    def forward(self, x):
        feat = self.cnn(x)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, W, C * H)
        feat = self.proj(feat)
        feat = self.pos_enc(feat)
        feat = self.transformer(feat)
        return self.fc(feat).log_softmax(2)

# Alias for checkpoint compatibility
CaptchaResTransformer = CRNN_ResTransformer


# ─── CHARSET ─────────────────────────────────────────────────────────────────

CHARSET   = string.ascii_lowercase + string.ascii_uppercase + string.digits
char2idx  = {c: i for i, c in enumerate(CHARSET)}
idx2char  = {i: c for c, i in char2idx.items()}
BLANK_IDX = len(CHARSET)   # 62
NUM_CLASSES = len(CHARSET) # 62


# ─── PREPROCESSING ───────────────────────────────────────────────────────────

IMAGE_WIDTH, IMAGE_HEIGHT = 160, 48

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize to 160×48, convert to RGB, normalise."""
    img = img.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return transform_eval(img)


# ─── CTC DECODING ────────────────────────────────────────────────────────────

def greedy_decode(log_probs: torch.Tensor) -> str:
    """log_probs: (T, C)"""
    indices = log_probs.argmax(1).tolist()
    result, prev = [], BLANK_IDX
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(idx2char[idx])
        prev = idx
    return "".join(result)


# ─── DATASET HELPERS ─────────────────────────────────────────────────────────

SUPPORTED = {".png", ".jpg", ".jpeg"}

def collect_files(dataset_dir: Path) -> list[Path]:
    """Return all image files under dataset_dir in sorted order."""
    files = sorted(
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise FileNotFoundError(
            f"No images found in {dataset_dir}\n"
            "Make sure --dataset points to the folder containing the .png files."
        )
    return files

def label_from_filename(path: Path) -> str:
    """
    Kaggle dataset filenames ARE the label, e.g. 'Ab3f.png' -> 'Ab3f'
    Also handles 'prefix_Ab3f.png' -> 'Ab3f'
    """
    stem = path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem

def is_valid_label(label: str) -> bool:
    return all(c in char2idx for c in label)


# ─── MAIN EVALUATION ─────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cpu")  # CPU-only, no GPU needed
    print(f"\n{'='*55}")
    print("  CAPTCHA Model Evaluation (CPU)")
    print(f"{'='*55}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : CPU")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    model = CRNN_ResTransformer(num_classes=NUM_CLASSES).to(device)

    # Handle both full checkpoint dicts and plain state dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Checkpoint type : full dict (epoch={checkpoint.get('epoch', '?')}, "
              f"acc={checkpoint.get('best_val_word_accuracy', checkpoint.get('word_acc', '?'))})")
    else:
        state_dict = checkpoint
        print("  Checkpoint type : state_dict only")

    # Rename keys if saved under old training script names
    rename_map = {
        "cnn_backbone.":    "cnn.",
        "feature_projection.": "proj.",
        "positional_encoder.positional_encoding": "pos_enc.pe",
        "transformer_encoder.": "transformer.",
        "ctc_classifier.":  "fc.",
        "identity_downsample.": "downsample.",
    }
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in rename_map.items():
            nk = nk.replace(old, new)
        new_state[nk] = v

    model.load_state_dict(new_state)
    model.eval()
    print("  Model loaded ✔")

    # ── Collect files ─────────────────────────────────────────────────────────
    print("\n[2/4] Scanning dataset...")
    all_files = collect_files(Path(args.dataset))
    all_files = [f for f in all_files if is_valid_label(label_from_filename(f))]
    print(f"  Total valid images : {len(all_files):,}")

    # Reproduce the same 80/10/10 split used during training
    random.seed(args.seed)
    random.shuffle(all_files)
    n_train = int(0.8 * len(all_files))
    n_val   = int(0.1 * len(all_files))
    test_files = all_files[n_train + n_val:]
    print(f"  Test split (10%)   : {len(test_files):,} images")

    # Optionally subsample for speed
    if args.samples and args.samples < len(test_files):
        test_files = random.sample(test_files, args.samples)
        print(f"  Subsampled to      : {args.samples} images  (--samples flag)")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Running inference on {len(test_files):,} images...")
    print("  (This takes ~10-20 min on CPU for the full test split)\n")

    correct_words = 0
    correct_chars = 0
    total_chars   = 0
    errors        = []   # (filename, true_label, predicted)

    with torch.no_grad():
        for path in tqdm(test_files, desc="Evaluating", unit="img"):
            true_label = label_from_filename(path)

            try:
                img    = Image.open(path)
                tensor = preprocess(img).unsqueeze(0).to(device)  # (1,3,48,160)
                output = model(tensor)                              # (1,W,C)
                log_probs = output[0].permute(1, 0)                # (W,C) -> greedy per timestep
                # Actually greedy_decode expects (T,C):
                log_probs_T = output.squeeze(0)                    # (W, C+1)
                pred = greedy_decode(log_probs_T)
            except Exception as e:
                errors.append((path.name, true_label, f"ERROR: {e}"))
                continue

            # Word accuracy (exact match, case-sensitive)
            if pred == true_label:
                correct_words += 1
            else:
                errors.append((path.name, true_label, pred))

            # Character accuracy (position-aligned)
            for pc, tc in zip(pred, true_label):
                if pc == tc:
                    correct_chars += 1
            total_chars += len(true_label)

    # ── Results ───────────────────────────────────────────────────────────────
    n = len(test_files)
    word_acc = correct_words / n        if n > 0            else 0.0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0

    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    print(f"  Images evaluated : {n:,}")
    print(f"  Word accuracy    : {word_acc*100:.2f}%  ({correct_words}/{n} exact matches)")
    print(f"  Char accuracy    : {char_acc*100:.2f}%  ({correct_chars}/{total_chars} chars correct)")
    print(f"  Errors logged    : {len(errors)}")
    print(f"{'='*55}\n")

    # ── Sample predictions ────────────────────────────────────────────────────
    print("  Sample predictions (first 15 errors):")
    print(f"  {'Filename':<25} {'True':^8} {'Predicted':^10} {'Match'}")
    print(f"  {'-'*55}")
    for fname, true, pred in errors[:15]:
        match = "✔" if pred == true else "✘"
        print(f"  {fname:<25} {true:^8} {pred:^10} {match}")

    # ── Save error CSV ────────────────────────────────────────────────────────
    if args.save_errors and errors:
        error_path = Path("evaluation_errors.csv")
        with open(error_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "true_label", "predicted", "correct"])
            for fname, true, pred in errors:
                writer.writerow([fname, true, pred, int(pred == true)])
        print(f"\n  Error details saved to: {error_path.resolve()}")

    return word_acc, char_acc


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CAPTCHA model on the Kaggle test split (CPU-only)."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to model_Transformer_new_78acc_state_dict.pth"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the Kaggle CAPTCHA dataset folder (containing .png files)"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Evaluate on N random test samples instead of all (faster). E.g. --samples 500"
    )
    parser.add_argument(
        "--split", type=float, default=0.1,
        help="Fraction used as test set (default 0.1). Must match training split."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible split (default 42). Must match training seed."
    )
    parser.add_argument(
        "--save-errors", action="store_true",
        help="Save a CSV of all incorrect predictions to evaluation_errors.csv"
    )

    args = parser.parse_args()
    evaluate(args)