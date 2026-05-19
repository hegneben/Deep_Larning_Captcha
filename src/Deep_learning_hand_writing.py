import os
import string
import torch
import torch.nn as nn
import torch.optim as optim
import math

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.backends.cudnn.benchmark = True

# =========================
# CHARSET
# =========================
# CHARSET
chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
blank_idx = len(chars)
num_classes = len(chars)

# =========================
# PATH
# =========================
path = r"C:\Users\Benedikt\.cache\kagglehub\datasets\parsasam\captcha-dataset\versions\1"

# =========================
# IMAGE PROCESSING
# =========================
def resize(img, size=(160, 48)):
    # Slightly larger for better feature extraction
    return img.convert("RGB").resize(size)

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =========================
# RAM DATASET
# =========================
class RAMCaptchaDataset(Dataset):
    def __init__(self, img_dir, augment=False):
        self.data = []
        self.augment = augment

        files = [f for f in os.listdir(img_dir)
                 if f.endswith((".png", ".jpg", ".jpeg"))]

        print(f"Loading {len(files)} images into RAM...")

        for f in tqdm(files):
            img = Image.open(os.path.join(img_dir, f)).convert("RGB")
            img_resized = resize(img)

            # Store PIL image if augmenting, else pre-convert
            if augment:
                self.data.append((img_resized, os.path.splitext(f)[0]))
            else:
                img_t = transform_val(img_resized)
                label = os.path.splitext(f)[0]
                label_t = torch.tensor([char2idx[c] for c in label.lower()], dtype=torch.long)
                self.data.append((img_t, label_t, label))

        print("Dataset fully in RAM")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.augment:
            img_pil, label_str = self.data[idx]
            img_t = transform(img_pil)
            label_t = torch.tensor([char2idx[c] for c in label_str.lower()], dtype=torch.long)
            return img_t, label_t, label_str
        else:
            return self.data[idx]

# =========================
# COLLATE
# =========================
def collate_fn(batch):
    imgs, labels, label_strs = zip(*batch)
    imgs = torch.stack(imgs)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)
    return imgs, labels_cat, lengths, list(label_strs)

# =========================
# UPGRADE 1: CTC DECODING
# =========================
def ctc_greedy_decode(log_probs, blank=blank_idx):
    """
    log_probs: (T, B, C)
    Returns list of decoded strings.
    """
    pred_indices = log_probs.argmax(2)  # (T, B)
    pred_indices = pred_indices.permute(1, 0)  # (B, T)
    results = []
    for seq in pred_indices:
        chars_out = []
        prev = blank
        for idx in seq.tolist():
            if idx != blank and idx != prev:
                chars_out.append(idx2char[idx])
            prev = idx
        results.append("".join(chars_out))
    return results


def ctc_beam_search_decode(log_probs, beam_width=5, blank=blank_idx):
    """
    Simple beam search CTC decoder (pure Python, no external lib needed).
    log_probs: (T, B, C)  — log probabilities
    Returns list of decoded strings.
    """
    log_probs_np = log_probs.cpu().float()
    T, B, C = log_probs_np.shape
    results = []

    for b in range(B):
        # beam: dict { prefix_tuple: (prob_blank, prob_non_blank) }
        beam = {(): (0.0, float("-inf"))}  # log probs

        for t in range(T):
            new_beam = {}
            log_p = log_probs_np[t, b]  # (C,)

            for prefix, (p_b, p_nb) in beam.items():
                # Extend with blank
                new_p_b = torch.logaddexp(
                    torch.tensor(p_b), torch.tensor(p_nb)
                ).item() + log_p[blank].item()

                key = prefix
                if key not in new_beam:
                    new_beam[key] = (float("-inf"), float("-inf"))
                new_beam[key] = (
                    torch.logaddexp(torch.tensor(new_beam[key][0]), torch.tensor(new_p_b)).item(),
                    new_beam[key][1]
                )

                # Extend with each non-blank char
                for c in range(C):
                    if c == blank:
                        continue
                    new_prefix = prefix + (c,)

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same char: only from blank path
                        new_p_nb = p_b + log_p[c].item()
                    else:
                        new_p_nb = torch.logaddexp(
                            torch.tensor(p_b), torch.tensor(p_nb)
                        ).item() + log_p[c].item()

                    if new_prefix not in new_beam:
                        new_beam[new_prefix] = (float("-inf"), float("-inf"))
                    new_beam[new_prefix] = (
                        new_beam[new_prefix][0],
                        torch.logaddexp(torch.tensor(new_beam[new_prefix][1]), torch.tensor(new_p_nb)).item()
                    )

            # Prune to top beam_width
            def total_prob(v):
                return torch.logaddexp(torch.tensor(v[0]), torch.tensor(v[1])).item()

            beam = dict(sorted(new_beam.items(), key=lambda x: total_prob(x[1]), reverse=True)[:beam_width])

        # Pick best prefix
        best = max(beam.items(), key=lambda x: torch.logaddexp(torch.tensor(x[1][0]), torch.tensor(x[1][1])).item())
        results.append("".join(idx2char[i] for i in best[0]))

    return results


# =========================
# UPGRADE 2: ACCURACY
# =========================
def compute_accuracy(preds, targets):
    """
    Exact-match accuracy: full CAPTCHA string must match.
    Also returns char-level accuracy.
    """
    correct_words = 0
    correct_chars = 0
    total_chars = 0

    for pred, tgt in zip(preds, targets):
        tgt = tgt.lower()
        if pred == tgt:
            correct_words += 1
        # Char-level (align by position)
        for p_c, t_c in zip(pred, tgt):
            if p_c == t_c:
                correct_chars += 1
        total_chars += len(tgt)

    word_acc = correct_words / len(targets) if targets else 0.0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    return word_acc, char_acc


# =========================
# UPGRADE 3: STRONGER ARCHITECTURE
# ResNet-style CNN + Transformer
# =========================

class ResBlock(nn.Module):
    """Basic residual block for the CNN backbone."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

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
    """Standard sinusoidal positional encoding for the Transformer."""
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CRNN_ResTransformer(nn.Module):
    """
    ResNet CNN backbone + Transformer encoder + CTC head.
    Much stronger than GRU for sequence recognition.
    """
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # === ResNet-style CNN Backbone ===
        self.cnn = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 1: 48→24
            ResBlock(32, 64, stride=2),
            # Block 2: 24→12
            ResBlock(64, 128, stride=2),
            # Block 3: 12→6  (height only → then collapse)
            ResBlock(128, 256, stride=2),
            # Extra refinement at small scale
            ResBlock(256, 256, stride=1),
        )
        # After 3× stride-2: H=48→6, W=160→20
        # Feature dim per column: 256 * 6 = 1536 → project to d_model

        self.proj = nn.Linear(256 * 6, d_model)

        # === Transformer Encoder ===
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True   # Pre-LN for more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === CTC Head ===
        self.fc = nn.Linear(d_model, num_classes + 1)  # +1 for blank

    def forward(self, x):
        # x: (B, 3, H, W) = (B, 3, 48, 160)
        feat = self.cnn(x)                   # (B, 256, 6, 20)  [approx]

        B, C, H, W = feat.shape
        feat = feat.permute(0, 3, 1, 2)      # (B, W, C, H)
        feat = feat.reshape(B, W, C * H)     # (B, W, C*H)  → time = width

        feat = self.proj(feat)               # (B, W, d_model)
        feat = self.pos_enc(feat)            # add positional encoding
        feat = self.transformer(feat)        # (B, W, d_model)

        logits = self.fc(feat)               # (B, W, num_classes+1)
        return logits.log_softmax(2)


# =========================
# DATASET + SPLIT
# =========================
dataset = RAMCaptchaDataset(path, augment=True)

train_size = int(0.8 * len(dataset))
val_size   = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn)

# =========================
# MODEL + OPTIMIZER
# =========================
model = CRNN_ResTransformer(num_classes=num_classes, d_model=256, nhead=8, num_layers=4).to(device)
criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

# AdamW + OneCycleLR: typically trains faster and generalizes better
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 20
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.1
)

# AMP
scaler = torch.amp.GradScaler("cuda")

# =========================
# LIVE PLOT
# =========================
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("CAPTCHA ResNet+Transformer – Training Dashboard", fontsize=13)

train_losses, val_losses = [], []
word_accs, char_accs = [], []
lrs = []

# =========================
# TRAIN LOOP
# =========================
best_word_acc = 0.0

for epoch in range(epochs):

    # ───── TRAIN ─────
    model.train()
    train_loss = 0.0

    for images, labels, lengths, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train"):
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            output = model(images)              # (B, T, C)
            input_lengths = torch.full(
                (images.size(0),), output.size(1),
                dtype=torch.long, device=device
            )
            loss = criterion(
                output.permute(1, 0, 2),        # (T, B, C)
                labels, input_lengths, lengths
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    lrs.append(scheduler.get_last_lr()[0])

    # ───── VALIDATION (with CTC Greedy + Accuracy) ─────
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels, lengths, label_strs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Val"):
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                output = model(images)
                input_lengths = torch.full(
                    (images.size(0),), output.size(1),
                    dtype=torch.long, device=device
                )
                loss = criterion(
                    output.permute(1, 0, 2),
                    labels, input_lengths, lengths
                )

            val_loss += loss.item()

            # Greedy decoding for accuracy measurement
            log_probs = output.permute(1, 0, 2)   # (T, B, C)
            preds = ctc_greedy_decode(log_probs)
            all_preds.extend(preds)
            all_targets.extend(label_strs)

    val_losses.append(val_loss / len(val_loader))
    w_acc, c_acc = compute_accuracy(all_preds, all_targets)
    word_accs.append(w_acc)
    char_accs.append(c_acc)

    print(f"\nEpoch {epoch+1:02d} | "
          f"Train Loss: {train_losses[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | "
          f"Word Acc: {w_acc*100:.2f}% | "
          f"Char Acc: {c_acc*100:.2f}% | "
          f"LR: {lrs[-1]:.2e}")

    # Show a few examples
    print("  Examples (pred → target):")
    for p, t in zip(all_preds[:5], all_targets[:5]):
        status = "✔" if p == t.lower() else "✘"
        print(f"    {status} '{p}' → '{t.lower()}'")

    # Save best model
    if w_acc > best_word_acc:
        best_word_acc = w_acc
        torch.save(model.state_dict(), "best_captcha_model.pth")
        print(f"  ★ New best saved: {w_acc*100:.2f}%")

    # ───── LIVE PLOT ─────
    ep_range = range(1, len(train_losses) + 1)

    axes[0].clear()
    axes[0].plot(ep_range, train_losses, label="Train", color="#4C9BE8")
    axes[0].plot(ep_range, val_losses,   label="Val",   color="#E8674C")
    axes[0].set_title("CTC Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].clear()
    axes[1].plot(ep_range, [a * 100 for a in word_accs], label="Word Acc %", color="#4CE87A")
    axes[1].plot(ep_range, [a * 100 for a in char_accs], label="Char Acc %", color="#C84CE8", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 100)
    axes[1].legend()

    axes[2].clear()
    axes[2].plot(ep_range, lrs, color="#E8C84C")
    axes[2].set_title("Learning Rate (OneCycle)")
    axes[2].set_xlabel("Epoch")

    fig.tight_layout()
    plt.pause(0.01)

plt.ioff()

# =========================
# FINAL TEST EVAL (Greedy + Beam Search)
# =========================
print("\n===== FINAL TEST EVALUATION =====")
model.load_state_dict(torch.load("best_captcha_model.pth"))
model.eval()

greedy_preds, beam_preds, test_targets = [], [], []

with torch.no_grad():
    for images, labels, lengths, label_strs in tqdm(test_loader, desc="Test"):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            output = model(images)

        log_probs = output.permute(1, 0, 2)  # (T, B, C)

        greedy_preds.extend(ctc_greedy_decode(log_probs))
        beam_preds.extend(ctc_beam_search_decode(log_probs, beam_width=5))
        test_targets.extend(label_strs)

greedy_word, greedy_char = compute_accuracy(greedy_preds, test_targets)
beam_word, beam_char     = compute_accuracy(beam_preds,   test_targets)

print(f"\nGreedy  → Word Acc: {greedy_word*100:.2f}%  |  Char Acc: {greedy_char*100:.2f}%")
print(f"Beam(5) → Word Acc: {beam_word*100:.2f}%  |  Char Acc: {beam_char*100:.2f}%")

print("\nSample predictions (Greedy | Beam | Target):")
for g, b, t in zip(greedy_preds[:10], beam_preds[:10], test_targets[:10]):
    print(f"  Greedy: '{g}'  Beam: '{b}'  Target: '{t.lower()}'")

plt.show()


