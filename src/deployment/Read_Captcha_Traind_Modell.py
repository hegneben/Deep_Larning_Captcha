import os
import math
import string
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# MODEL PATH
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "new_Models" / "model_Transformer_new_78acc_state_dict.pth"

print("Loading model from:", MODEL_PATH)
print("Exists?", MODEL_PATH.exists())

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

# =========================
# LOAD CHECKPOINT
# =========================
checkpoint = torch.load(str(MODEL_PATH), map_location=device)

# Charset
chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

blank_idx = len(char2idx)
num_classes = len(char2idx)
print(f"Charset: {num_classes} Zeichen  |  blank_idx: {blank_idx}")

model = CRNN_ResTransformer(num_classes=num_classes).to(device)

state_dict = checkpoint["model_state_dict"]

rename_map = {
    "cnn_backbone.": "cnn.",
    "feature_projection.": "proj.",
    "positional_encoder.positional_encoding": "pos_enc.pe",
    "transformer_encoder.": "transformer.",
    "ctc_classifier.": "fc.",
    "identity_downsample.": "downsample.",
}

new_state_dict = {}

for key, value in state_dict.items():
    new_key = key
    for old, new in rename_map.items():
        new_key = new_key.replace(old, new)
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

model.eval()
print("Modell geladen ✔")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =========================
# PREPROCESSING
# =========================
def preprocess(img, size=(160, 48)):
    """Resize mit Padding (Seitenverhältnis bleibt erhalten)."""
    img = img.convert("RGB")
    img.thumbnail(size, Image.LANCZOS)

    result = Image.new("RGB", size, (255, 255, 255))
    offset = (
        (size[0] - img.width)  // 2,
        (size[1] - img.height) // 2
    )
    result.paste(img, offset)
    return result

# =========================
# CTC DECODING
# =========================
def greedy_decode(log_probs):
    """log_probs: (T, 1, C)"""
    indices = log_probs.argmax(2)[:, 0].tolist()
    result  = []
    prev    = blank_idx
    for idx in indices:
        if idx != blank_idx and idx != prev:
            result.append(idx2char[idx])
        prev = idx
    return "".join(result)


def beam_search_decode(log_probs, beam_width=5):
    """log_probs: (T, 1, C)"""
    T, _, C = log_probs.shape
    beam    = {(): (0.0, float("-inf"))}

    for t in range(T):
        new_beam = {}
        lp = log_probs[t, 0]  # (C,)

        for prefix, (p_b, p_nb) in beam.items():
            # Extend mit blank
            new_p_b = torch.logaddexp(
                torch.tensor(p_b), torch.tensor(p_nb)
            ).item() + lp[blank_idx].item()

            if prefix not in new_beam:
                new_beam[prefix] = (float("-inf"), float("-inf"))
            new_beam[prefix] = (
                torch.logaddexp(
                    torch.tensor(new_beam[prefix][0]),
                    torch.tensor(new_p_b)
                ).item(),
                new_beam[prefix][1]
            )

            # Extend mit jedem Zeichen
            for c in range(C):
                if c == blank_idx:
                    continue
                new_prefix = prefix + (c,)
                if len(prefix) > 0 and prefix[-1] == c:
                    new_p_nb = p_b + lp[c].item()
                else:
                    new_p_nb = torch.logaddexp(
                        torch.tensor(p_b), torch.tensor(p_nb)
                    ).item() + lp[c].item()

                if new_prefix not in new_beam:
                    new_beam[new_prefix] = (float("-inf"), float("-inf"))
                new_beam[new_prefix] = (
                    new_beam[new_prefix][0],
                    torch.logaddexp(
                        torch.tensor(new_beam[new_prefix][1]),
                        torch.tensor(new_p_nb)
                    ).item()
                )

        def total(v):
            return torch.logaddexp(torch.tensor(v[0]), torch.tensor(v[1])).item()

        beam = dict(
            sorted(new_beam.items(), key=lambda x: total(x[1]), reverse=True)[:beam_width]
        )

    best = max(beam.items(), key=lambda x: torch.logaddexp(
        torch.tensor(x[1][0]), torch.tensor(x[1][1])
    ).item())
    return "".join(idx2char[i] for i in best[0])


# =========================
# PREDICT
# =========================
def predict(image_path, use_beam=True, beam_width=5, show_images=True):
    if not os.path.exists(image_path):
        print(f"Datei nicht gefunden: {image_path}")
        return None

    img     = Image.open(image_path)
    img_pre = preprocess(img)

    if show_images:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(img_pre)
        axes[1].set_title("Model Input (160×48)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    tensor = transform(img_pre).unsqueeze(0).to(device)  # (1, 3, 48, 160)

    with torch.no_grad():
        output   = model(tensor)                          # (1, W, C)
        log_probs = output.permute(1, 0, 2)               # (W, 1, C)

    greedy = greedy_decode(log_probs)
    beam   = beam_search_decode(log_probs, beam_width=beam_width) if use_beam else None

    print(f"\n  Greedy:      '{greedy}'")
    if beam:
        print(f"  Beam({beam_width}):    '{beam}'")

    return beam if use_beam else greedy


# =========================
# MEHRERE BILDER TESTEN
# =========================
def predict_batch(image_paths, use_beam=False):
    """Mehrere Bilder auf einmal testen (ohne Plot)."""
    results = []
    for path in image_paths:
        pred = predict(path, use_beam=use_beam, show_images=False)
        results.append((os.path.basename(path), pred))
        print(f"  {os.path.basename(path):30s} → '{pred}'")
    return results


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    base_path = r"C:\Users\Benedikt\Downloads"

    print("\n" + "="*50)
    print("  CAPTCHA Inference – ResNet + Transformer")
    print("="*50)
    print("  [1] Einzelnes Bild")
    print("  [2] Alle Bilder in einem Ordner testen")
    print("="*50)

    mode = input("\nModus wählen (1/2): ").strip()

    if mode == "1":
        file_name  = input("Bildname (z.B. test.png): ").strip()
        image_path = os.path.join(base_path, file_name)
        predict(image_path, use_beam=True, beam_width=5, show_images=True)

    elif mode == "2":
        folder = input(f"Ordnerpfad (Enter für {base_path}): ").strip() or base_path
        files  = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"\n{len(files)} Bilder gefunden:\n")
        predict_batch(files, use_beam=False)  # Greedy für Speed bei vielen Bildern

    else:
        print("Ungültige Auswahl.")
