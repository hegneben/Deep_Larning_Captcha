"""
Script 2 — Experimental Ladder Evaluation
==========================================
Evaluates the 7-level additive distortion ladder dataset.
Each level adds one new distortion type on top of the previous.

L0: clean centered text
L1: + random character spacing
L2: + vertical jitter
L3: + font variation (31 fonts)
L4: + character rotation ±12°
L5: + mild noise + 1 occlusion line + blur
L6: + heavier noise + 3 lines + more blur

Usage:
    python evaluate_ladder.py
        --model  "models\\new_Models\\model_Transformer_new_78acc_state_dict.pth"
        --ladder "data\\synthetic\\experimental_ladder"

Output:
    results\\ladder_results.csv
    Console summary table
"""

import csv
import math
import string
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ─── MODEL (same as Script 1) ─────────────────────────────────────────────────

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
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in rename_map.items():
            nk = nk.replace(old, new)
        new_state[nk] = v
    model.load_state_dict(new_state)
    model.eval()
    print("Model loaded ✔\n")
    return model


# ─── LADDER LEVEL DESCRIPTIONS ───────────────────────────────────────────────

LEVEL_DESCRIPTIONS = {
    0: "Clean centered text",
    1: "+ Random spacing",
    2: "+ Vertical jitter",
    3: "+ Font variation",
    4: "+ Character rotation",
    5: "+ Mild noise & lines",
    6: "+ Heavy noise & lines",
}


# ─── EVALUATE ────────────────────────────────────────────────────────────────

def evaluate(args):
    model  = load_model(args.model)
    ladder_dir = Path(args.ladder)
    metadata_csv = ladder_dir / "ladder_metadata.csv"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"ladder_metadata.csv not found in {ladder_dir}")

    # Read metadata
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total ladder images: {len(rows):,}")
    for level in range(7):
        n = sum(1 for r in rows if int(r["ladder_level"]) == level)
        print(f"  L{level} ({LEVEL_DESCRIPTIONS[level]}): {n:,} images")
    print()

    level_stats = defaultdict(lambda: {"correct_words": 0, "correct_chars": 0,
                                        "total_chars": 0, "total": 0,
                                        "fonts": set()})
    all_results = []

    with torch.no_grad():
        for row in tqdm(rows, desc="Evaluating ladder", unit="img"):
            level_int  = int(row["ladder_level"])
            level_name = row["ladder_name"]
            true_label = row["label"]
            font       = row["font"]

            # Build image path — filename in CSV is relative to ladder_dir
            rel_path = row["filename"].replace("\\", "/")
            img_path = ladder_dir / rel_path

            if not img_path.exists():
                continue

            try:
                tensor = preprocess(Image.open(img_path)).unsqueeze(0)
                output = model(tensor).squeeze(0)
                pred   = greedy_decode(output)
            except Exception as e:
                pred = f"ERROR:{e}"

            word_correct = int(pred == true_label)
            char_correct = sum(pc == tc for pc, tc in zip(pred, true_label))

            level_stats[level_int]["correct_words"] += word_correct
            level_stats[level_int]["correct_chars"] += char_correct
            level_stats[level_int]["total_chars"]   += len(true_label)
            level_stats[level_int]["total"]         += 1
            level_stats[level_int]["fonts"].add(font)

            all_results.append({
                "image":        rel_path,
                "true_label":   true_label,
                "predicted":    pred,
                "ladder_level": level_int,
                "ladder_name":  level_name,
                "font":         font,
                "word_correct": word_correct,
                "char_correct": char_correct,
            })

    # Save CSV
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "ladder_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    print(f"\n{'='*70}")
    print("  EXPERIMENTAL LADDER RESULTS")
    print(f"{'='*70}")
    print(f"  {'L':<2}  {'Description':<30} {'N':>5}  {'Word Acc':>10}  {'Char Acc':>10}")
    print(f"  {'-'*65}")

    for level_int in sorted(level_stats.keys()):
        s     = level_stats[level_int]
        n     = s["total"]
        w_acc = s["correct_words"] / n              if n > 0              else 0.0
        c_acc = s["correct_chars"] / s["total_chars"] if s["total_chars"] > 0 else 0.0
        desc  = LEVEL_DESCRIPTIONS.get(level_int, "")
        print(f"  L{level_int}  {desc:<30} {n:>5}  {w_acc*100:>9.2f}%  {c_acc*100:>9.2f}%")

    print(f"{'='*70}")
    print(f"\nDetailed results saved to: {out_csv.resolve()}\n")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the experimental ladder dataset."
    )
    parser.add_argument("--model", required=True,
        help="Path to model_Transformer_new_78acc_state_dict.pth")
    parser.add_argument("--ladder", required=True,
        help="Path to data\\synthetic\\experimental_ladder folder")
    evaluate(parser.parse_args())
