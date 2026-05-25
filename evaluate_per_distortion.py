"""
Script 3 — Per-Distortion Isolation: Generate + Evaluate
==========================================================
Generates isolated single-distortion test sets from the clean synthetic
images, then evaluates the model on each one.

For each distortion type, 5 intensity levels are tested with 200 images each
(1,000 images per distortion type, 5,000 total).

Distortion types isolated:
  - blur          (radius: 0.3, 0.7, 1.1, 1.5, 2.0)
  - noise         (std: 5, 10, 18, 28, 40)
  - rotation      (degrees: 3, 6, 10, 14, 18)
  - lines         (count: 1, 3, 5, 7, 10)
  - occlusion     (blocks: 1, 2, 3, 4, 5)

Usage:
    python evaluate_per_distortion.py
        --model "models\\new_Models\\model_Transformer_new_78acc_state_dict.pth"
        --clean "data\\synthetic\\clean"
        --output "data\\synthetic\\per_distortion"

Output:
    data\\synthetic\\per_distortion\\  (generated images)
    results\\per_distortion_results.csv
    Console summary table
"""

import csv
import math
import random
import string
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from torchvision import transforms
from tqdm import tqdm


# ─── MODEL ───────────────────────────────────────────────────────────────────

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


# ─── CHARSET & HELPERS ────────────────────────────────────────────────────────

CHARSET     = string.ascii_lowercase + string.ascii_uppercase + string.digits
char2idx    = {c: i for i, c in enumerate(CHARSET)}
idx2char    = {i: c for c, i in char2idx.items()}
BLANK_IDX   = len(CHARSET)
NUM_CLASSES = len(CHARSET)

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess(img: Image.Image) -> torch.Tensor:
    return transform_eval(img.convert("RGB").resize((160, 48)))

def greedy_decode(log_probs: torch.Tensor) -> str:
    indices = log_probs.argmax(1).tolist()
    result, prev = [], BLANK_IDX
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(idx2char[idx])
        prev = idx
    return "".join(result)

def load_model(model_path: str) -> nn.Module:
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = CRNN_ResTransformer(num_classes=NUM_CLASSES)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    rename_map = {
        "cnn_backbone.":    "cnn.",
        "feature_projection.": "proj.",
        "positional_encoder.positional_encoding": "pos_enc.pe",
        "transformer_encoder.": "transformer.",
        "ctc_classifier.":  "fc.",
        "identity_downsample.": "downsample.",
    }
    new_state = {k.replace(o, n): v
                 for k, v in state_dict.items()
                 for o, n in rename_map.items()
                 if True}
    # Cleaner rename
    new_state2 = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in rename_map.items():
            nk = nk.replace(old, new)
        new_state2[nk] = v
    model.load_state_dict(new_state2)
    model.eval()
    print("Model loaded ✔\n")
    return model


# ─── ISOLATED DISTORTION FUNCTIONS ───────────────────────────────────────────

def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_noise(img: Image.Image, std: float) -> Image.Image:
    arr   = np.array(img).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    angle = random.uniform(-degrees, degrees)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

def apply_lines(img: Image.Image, count: int) -> Image.Image:
    img  = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(count):
        draw.line(
            (0, random.randint(0, h), w, random.randint(0, h)),
            fill="black", width=random.randint(1, 3)
        )
    return img

def apply_occlusion(img: Image.Image, count: int) -> Image.Image:
    img  = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    block_w = int(w * 0.12)
    block_h = int(h * 0.35)
    for _ in range(count):
        x = random.randint(0, w - block_w)
        y = random.randint(0, h - block_h)
        draw.rectangle([x, y, x + block_w, y + block_h], fill="black")
    return img


# ─── DISTORTION CONFIGS ───────────────────────────────────────────────────────
# Each entry: (distortion_name, param_name, param_values, apply_fn)

DISTORTIONS = [
    ("blur",      "radius",  [0.3, 0.7, 1.1, 1.5, 2.0], apply_blur),
    ("noise",     "std",     [5,   10,  18,  28,  40  ], apply_noise),
    ("rotation",  "degrees", [3,   6,   10,  14,  18  ], apply_rotation),
    ("lines",     "count",   [1,   3,   5,   7,   10  ], apply_lines),
    ("occlusion", "blocks",  [1,   2,   3,   4,   5   ], apply_occlusion),
]

SAMPLES_PER_LEVEL = 200   # 200 images × 5 levels × 5 distortions = 5,000 total


# ─── MAIN ────────────────────────────────────────────────────────────────────

def evaluate(args):
    model     = load_model(args.model)
    clean_dir = Path(args.clean)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect clean images and their labels
    supported = {".png", ".jpg", ".jpeg"}
    clean_files = sorted(
        p for p in clean_dir.iterdir()
        if p.suffix.lower() in supported
    )
    if not clean_files:
        raise FileNotFoundError(f"No images found in {clean_dir}")

    print(f"Clean images available: {len(clean_files):,}")
    print(f"Samples per distortion level: {SAMPLES_PER_LEVEL}")
    print(f"Total images to generate & evaluate: "
          f"{len(DISTORTIONS) * 5 * SAMPLES_PER_LEVEL:,}\n")

    def label_from_path(p: Path) -> str:
        stem = p.stem
        return stem.split("_", 1)[1] if "_" in stem else stem

    # Filter to valid labels
    clean_files = [f for f in clean_files
                   if all(c in char2idx for c in label_from_path(f))]

    random.seed(42)
    sample_pool = random.sample(clean_files, min(SAMPLES_PER_LEVEL, len(clean_files)))

    all_results = []
    level_stats = defaultdict(lambda: {"correct_words": 0, "correct_chars": 0,
                                        "total_chars": 0, "total": 0})

    for dist_name, param_name, param_values, apply_fn in DISTORTIONS:
        dist_out = out_dir / dist_name
        dist_out.mkdir(exist_ok=True)

        print(f"\n── {dist_name.upper()} ──────────────────────────────")

        for level_idx, param_val in enumerate(param_values):
            level_dir = dist_out / f"level_{level_idx+1}_{param_name}_{param_val}"
            level_dir.mkdir(exist_ok=True)

            correct_w, correct_c, total_c = 0, 0, 0

            with torch.no_grad():
                for src_path in tqdm(
                    sample_pool,
                    desc=f"  {dist_name} {param_name}={param_val}",
                    unit="img"
                ):
                    true_label = label_from_path(src_path)

                    try:
                        img      = Image.open(src_path).convert("RGB").resize((160, 48))
                        img_dist = apply_fn(img, param_val)
                        tensor   = preprocess(img_dist).unsqueeze(0)
                        output   = model(tensor).squeeze(0)
                        pred     = greedy_decode(output)
                    except Exception as e:
                        pred = f"ERROR:{e}"

                    word_ok  = int(pred == true_label)
                    char_ok  = sum(pc == tc for pc, tc in zip(pred, true_label))

                    correct_w += word_ok
                    correct_c += char_ok
                    total_c   += len(true_label)

                    key = (dist_name, level_idx + 1, param_val)
                    level_stats[key]["correct_words"] += word_ok
                    level_stats[key]["correct_chars"] += char_ok
                    level_stats[key]["total_chars"]   += len(true_label)
                    level_stats[key]["total"]         += 1

                    all_results.append({
                        "distortion":  dist_name,
                        "param_name":  param_name,
                        "param_value": param_val,
                        "level":       level_idx + 1,
                        "true_label":  true_label,
                        "predicted":   pred,
                        "word_correct": word_ok,
                        "char_correct": char_ok,
                    })

            n     = len(sample_pool)
            w_acc = correct_w / n      if n > 0      else 0.0
            c_acc = correct_c / total_c if total_c > 0 else 0.0
            print(f"    {param_name}={param_val:<6}  Word: {w_acc*100:5.1f}%  Char: {c_acc*100:5.1f}%")

    # Save CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "per_distortion_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    # Summary table
    print(f"\n{'='*72}")
    print("  PER-DISTORTION ISOLATION RESULTS")
    print(f"{'='*72}")
    print(f"  {'Distortion':<12} {'Level':>5}  {'Param Value':>12}  "
          f"{'Word Acc':>10}  {'Char Acc':>10}")
    print(f"  {'-'*65}")

    prev_dist = None
    for (dist_name, level, param_val), s in sorted(
        level_stats.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        if dist_name != prev_dist and prev_dist is not None:
            print(f"  {'-'*65}")
        prev_dist = dist_name
        n     = s["total"]
        w_acc = s["correct_words"] / n              if n > 0              else 0.0
        c_acc = s["correct_chars"] / s["total_chars"] if s["total_chars"] > 0 else 0.0
        print(f"  {dist_name:<12} {level:>5}  {str(param_val):>12}  "
              f"{w_acc*100:>9.1f}%  {c_acc*100:>9.1f}%")

    print(f"{'='*72}")
    print(f"\nDetailed results saved to: {out_csv.resolve()}\n")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and evaluate per-distortion isolation datasets."
    )
    parser.add_argument("--model", required=True,
        help="Path to model_Transformer_new_78acc_state_dict.pth")
    parser.add_argument("--clean", required=True,
        help="Path to data\\synthetic\\clean folder")
    parser.add_argument("--output", required=True,
        help="Output folder for generated images (e.g. data\\synthetic\\per_distortion)")
    evaluate(parser.parse_args())
