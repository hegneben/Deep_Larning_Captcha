# -*- coding: utf-8 -*-
"""
Use this to convert model into state_dict temporarily, for proper deployement
"""

import sys
from pathlib import Path
import math

import torch
import torch.nn as nn

# =========================
# ARCHITEKTUR
# =========================
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
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CRNN_ResTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32,  64,  stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
        )
        # Nach 3x stride-2: H=48→6, W=160→20
        self.proj    = nn.Linear(256 * 6, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes + 1)  # +1 für blank

    def forward(self, x):
        feat = self.cnn(x)                    # (B, 256, H', W')
        B, C, H, W = feat.shape
        feat = feat.permute(0, 3, 1, 2)       # (B, W, C, H)
        feat = feat.reshape(B, W, C * H)      # (B, W, C*H)
        feat = self.proj(feat)                # (B, W, d_model)
        feat = self.pos_enc(feat)
        feat = self.transformer(feat)         # (B, W, d_model)
        return self.fc(feat).log_softmax(2)   # (B, W, num_classes+1)

#Alias included for loading checkpoints saved with old class name
CaptchaResTransformer = CRNN_ResTransformer
ResidualBlock = ResBlock
SinusoidalPositionalEncoding = PositionalEncoding

setattr(sys.modules["__main__"], "CaptchaResTransformer", CRNN_ResTransformer)
setattr(sys.modules[__name__], "CaptchaResTransformer", CRNN_ResTransformer)

setattr(sys.modules["__main__"], "ResidualBlock", ResBlock)
setattr(sys.modules[__name__], "ResidualBlock", ResBlock)

setattr(sys.modules["__main__"], "SinusoidalPositionalEncoding", PositionalEncoding)
setattr(sys.modules[__name__], "SinusoidalPositionalEncoding", PositionalEncoding)
# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OLD_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "new_Models"
    / "model_Transformer_new_78acc.pth"
)

NEW_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "new_Models"
    / "model_Transformer_new_78acc_state_dict.pth"
)

# ============================================================
# LOAD OLD CHECKPOINT
# ============================================================

print("Loading old checkpoint...")
checkpoint = torch.load(
    str(OLD_MODEL_PATH),
    map_location="cpu",
    weights_only=False
)

# ============================================================
# EXTRACT STATE_DICT
# ============================================================

if isinstance(checkpoint, nn.Module):
    state_dict = checkpoint.state_dict()

elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

else:
    state_dict = checkpoint

# ============================================================
# SAVE CLEAN STATE_DICT
# ============================================================

torch.save(
    {
        "model_state_dict": state_dict
    },
    str(NEW_MODEL_PATH)
)

print("Saved clean state_dict checkpoint:")
print(NEW_MODEL_PATH)