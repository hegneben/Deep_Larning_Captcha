"""
CAPTCHA Recognition — ResNet CNN + Transformer + CTC Loss
==========================================================

WHAT THIS SCRIPT DOES:
    Trains a deep learning model to read text from CAPTCHA images.
    The model uses a ResNet-style CNN to extract visual features,
    a Transformer encoder to model the character sequence, and
    CTC loss to handle variable-length outputs without needing
    pre-segmented characters.

HOW TO RUN:
    1. Create and activate the existing Conda environment:
           conda activate torch_gpu

       (Skip environment creation because it is already prepared.)

    2. Verify PyTorch + CUDA are working:
           python -c "import torch; print(torch.cuda.is_available())"

       -> must print True, otherwise training runs on CPU (~10x slower)

    3. If needed, install/update PyTorch with matching CUDA support:
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

       Check CUDA version first:
           nvidia-smi

       Then use the matching PyTorch build:
           CUDA 11.8  ->  .../whl/cu118
           CUDA 12.1  ->  .../whl/cu121
           CUDA 12.4  ->  .../whl/cu124

    4. Install remaining dependencies:
           pip install pillow tqdm matplotlib

    5. Set DATASET_PATH in captcha_recognition_refactored.py
       to your local dataset directory.

    6. Run the training script:
           python captcha_recognition_refactored.py

REQUIREMENTS:
    - CUDA-capable GPU strongly recommended (CPU training is ~10x slower)
    - Dataset: images named after their label, e.g. "Ab3f.png"
    - Supported image formats: .png, .jpg, .jpeg

IMPORTANT NOTES FOR USERS:
    - DATASET_PATH must point to a flat folder of labeled image files.
      The label is read directly from the filename (without extension).
    - Only characters defined in CHARSET are recognised. Any image whose
      filename contains a character outside the charset will be skipped
      silently -- check your dataset for special characters beforehand.
    - The best model checkpoint is saved to CHECKPOINT_PATH after every
      epoch that achieves a new highest validation word accuracy.
    - Beam search decoding is slower than greedy decoding (runs only
      during the final test evaluation, not during training).
    - If VRAM runs out, reduce BATCH_SIZE (e.g. 64 or 32).
    - Image size (IMAGE_WIDTH x IMAGE_HEIGHT) affects model architecture.
      Do NOT change these values after training has started -- the CNN
      projection layer dimension (CNN_FEATURE_DIM) depends on them.
"""

import os
import math
import string
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION -- edit these values before running
# =============================================================================

DATASET_PATH    = r"C:\Users\Benedikt\.cache\kagglehub\datasets\parsasam\captcha-dataset\versions\1"
CHECKPOINT_PATH = "best_captcha_model.pth"

IMAGE_WIDTH  = 160
IMAGE_HEIGHT = 48

BATCH_SIZE      = 128
NUM_EPOCHS      = 20
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
GRAD_CLIP_NORM  = 5.0

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1

TRANSFORMER_D_MODEL    = 256
TRANSFORMER_NUM_HEADS  = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DROPOUT    = 0.1

BEAM_SEARCH_WIDTH = 5


# =============================================================================
# DEVICE SETUP
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True


# =============================================================================
# CHARACTER SET
# =============================================================================
# Defines which characters the model can predict.
# NOTE: Every character in your image filenames must appear here.
#       Add or remove characters to match your dataset.

CHARSET = string.ascii_lowercase + string.ascii_uppercase + string.digits
# "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

char_to_index = {char: idx for idx, char in enumerate(CHARSET)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

BLANK_TOKEN_INDEX = len(CHARSET)   # = 62
NUM_CLASSES       = len(CHARSET)   # = 62


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

transform_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def resize_image_to_fixed_size(pil_image: Image.Image) -> Image.Image:
    return pil_image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))


# =============================================================================
# DATASET
# =============================================================================
# FIX: Replaced single CaptchaRAMDataset (with augmentation=True applied to
#      all splits) with CaptchaSubsetDataset that takes an explicit file list.
#      The file list is split BEFORE creating datasets, so val/test never
#      receive training augmentation.

class CaptchaSubsetDataset(Dataset):
    """
    Loads a predefined list of CAPTCHA image files into RAM.

    Each image filename (without extension) is treated as the ground-truth label.
    Example: "Ab3f.png" -> label "Ab3f"

    Labels are treated CASE-SENSITIVELY: 'a' and 'A' are different classes.

    Args:
        image_directory   : path to the folder containing image files.
        filenames         : explicit list of filenames to load.
        apply_augmentation: True for train only. False for val and test.
    """

    def __init__(self, image_directory: str, filenames: list,
                 apply_augmentation: bool = False):
        self.samples            = []
        self.apply_augmentation = apply_augmentation

        print(f"Loading {len(filenames)} images into RAM...")

        for filename in tqdm(filenames):
            image_path   = os.path.join(image_directory, filename)
            label_string = os.path.splitext(filename)[0]

            pil_image     = Image.open(image_path).convert("RGB")
            resized_image = resize_image_to_fixed_size(pil_image)

            if apply_augmentation:
                self.samples.append((resized_image, label_string))
            else:
                image_tensor = transform_eval(resized_image)
                label_tensor = torch.tensor(
                    [char_to_index[c] for c in label_string],
                    dtype=torch.long
                )
                self.samples.append((image_tensor, label_tensor, label_string))

        print(f"Dataset fully loaded -- {len(self.samples)} samples in RAM.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_index: int):
        if self.apply_augmentation:
            pil_image, label_string = self.samples[sample_index]
            image_tensor = transform_train(pil_image)
            label_tensor = torch.tensor(
                [char_to_index[c] for c in label_string],
                dtype=torch.long
            )
            return image_tensor, label_tensor, label_string
        else:
            return self.samples[sample_index]


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_variable_length_batch(batch):
    image_list, label_list, label_string_list = zip(*batch)

    image_batch   = torch.stack(image_list)
    label_lengths = torch.tensor([len(label) for label in label_list], dtype=torch.long)
    labels_concat = torch.cat(label_list)

    return image_batch, labels_concat, label_lengths, list(label_string_list)


# =============================================================================
# CTC DECODING
# =============================================================================

def greedy_ctc_decode(log_probabilities: torch.Tensor) -> list[str]:
    """
    Greedy (argmax) CTC decoding.

    Args:
        log_probabilities: shape (T, B, C)

    Returns:
        List of decoded strings, one per sample in the batch.
    """
    most_likely_indices = log_probabilities.argmax(dim=2)
    most_likely_indices = most_likely_indices.permute(1, 0)

    decoded_strings = []
    for sequence in most_likely_indices:
        characters     = []
        previous_index = BLANK_TOKEN_INDEX

        for current_index in sequence.tolist():
            if current_index != BLANK_TOKEN_INDEX and current_index != previous_index:
                characters.append(index_to_char[current_index])
            previous_index = current_index

        decoded_strings.append("".join(characters))

    return decoded_strings


def beam_search_ctc_decode(
    log_probabilities: torch.Tensor,
    beam_width: int = BEAM_SEARCH_WIDTH
) -> list[str]:
    """
    Beam search CTC decoding.

    Args:
        log_probabilities : shape (T, B, C)
        beam_width        : number of candidates to keep

    Returns:
        List of decoded strings, one per sample in the batch.
    """
    log_probs_cpu = log_probabilities.cpu().float()
    num_timesteps, batch_size, num_output_classes = log_probs_cpu.shape

    decoded_strings = []

    for batch_index in range(batch_size):
        beam_candidates = {(): (0.0, float("-inf"))}

        for timestep in range(num_timesteps):
            log_probs_at_t = log_probs_cpu[timestep, batch_index]
            updated_beam   = {}

            for prefix, (prob_blank, prob_non_blank) in beam_candidates.items():
                total_prob = torch.logaddexp(
                    torch.tensor(prob_blank), torch.tensor(prob_non_blank)
                ).item()
                new_prob_blank = total_prob + log_probs_at_t[BLANK_TOKEN_INDEX].item()

                if prefix not in updated_beam:
                    updated_beam[prefix] = (float("-inf"), float("-inf"))
                updated_beam[prefix] = (
                    torch.logaddexp(
                        torch.tensor(updated_beam[prefix][0]),
                        torch.tensor(new_prob_blank)
                    ).item(),
                    updated_beam[prefix][1]
                )

                for char_index in range(num_output_classes):
                    if char_index == BLANK_TOKEN_INDEX:
                        continue

                    extended_prefix = prefix + (char_index,)

                    if len(prefix) > 0 and prefix[-1] == char_index:
                        new_prob_non_blank = prob_blank + log_probs_at_t[char_index].item()
                    else:
                        new_prob_non_blank = total_prob + log_probs_at_t[char_index].item()

                    if extended_prefix not in updated_beam:
                        updated_beam[extended_prefix] = (float("-inf"), float("-inf"))
                    updated_beam[extended_prefix] = (
                        updated_beam[extended_prefix][0],
                        torch.logaddexp(
                            torch.tensor(updated_beam[extended_prefix][1]),
                            torch.tensor(new_prob_non_blank)
                        ).item()
                    )

            def total_log_prob(prob_pair):
                return torch.logaddexp(
                    torch.tensor(prob_pair[0]), torch.tensor(prob_pair[1])
                ).item()

            beam_candidates = dict(
                sorted(updated_beam.items(), key=lambda item: total_log_prob(item[1]), reverse=True)
                [:beam_width]
            )

        best_prefix, _ = max(
            beam_candidates.items(),
            key=lambda item: torch.logaddexp(
                torch.tensor(item[1][0]), torch.tensor(item[1][1])
            ).item()
        )
        decoded_strings.append("".join(index_to_char[i] for i in best_prefix))

    return decoded_strings


# =============================================================================
# ACCURACY METRICS
# =============================================================================

def compute_word_and_char_accuracy(
    predicted_strings: list[str],
    target_strings: list[str]
) -> tuple[float, float]:
    """
    Computes word and character accuracy over a batch of predictions.

    Comparisons are CASE-SENSITIVE: 'Ab3f' != 'ab3f'.

    Returns:
        (word_accuracy, char_accuracy) -- both in range [0.0, 1.0]
    """
    correct_full_strings = 0
    correct_characters   = 0
    total_characters     = 0

    for prediction, target in zip(predicted_strings, target_strings):
        # FIX: case-sensitive comparison -- removed .lower() normalisation
        if prediction == target:
            correct_full_strings += 1

        for predicted_char, target_char in zip(prediction, target):
            if predicted_char == target_char:
                correct_characters += 1

        total_characters += len(target)

    word_accuracy = correct_full_strings / len(target_strings) if target_strings else 0.0
    char_accuracy = correct_characters   / total_characters    if total_characters > 0 else 0.0

    return word_accuracy, char_accuracy


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ResidualBlock(nn.Module):
    """
    A single residual (skip-connection) block from the ResNet family.

    Args:
        in_channels  : number of input feature channels
        out_channels : number of output feature channels
        stride       : spatial downsampling factor (1 = no change, 2 = halve H and W)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            self.identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, input_feature_map: torch.Tensor) -> torch.Tensor:
        identity = input_feature_map

        out = self.relu(self.bn1(self.conv1(input_feature_map)))
        out = self.bn2(self.conv2(out))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(input_feature_map)

        return self.relu(out + identity)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds positional information to Transformer input embeddings.

    Args:
        embedding_dim       : must match the Transformer d_model
        max_sequence_length : maximum number of timesteps expected
        dropout_rate        : applied after adding the positional encoding
    """

    def __init__(self, embedding_dim: int, max_sequence_length: int = 256,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        positional_encoding = torch.zeros(max_sequence_length, embedding_dim)
        positions           = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        frequency_divisors  = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        positional_encoding[:, 0::2] = torch.sin(positions * frequency_divisors)
        positional_encoding[:, 1::2] = torch.cos(positions * frequency_divisors)

        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings + self.positional_encoding[:, :embeddings.size(1)]
        return self.dropout(embeddings)


class CaptchaResTransformer(nn.Module):
    """
    Full CAPTCHA recognition model: ResNet CNN + Transformer encoder + CTC head.

    Input shape  : (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH) = (B, 3, 48, 160)
    Output shape : (B, num_timesteps, NUM_CLASSES + 1) -- log-softmax probabilities

    NOTE:
        - CNN_FEATURE_DIM (= 256 * 6 = 1536) is derived from IMAGE_HEIGHT.
          If IMAGE_HEIGHT changes, update CNN_FEATURE_DIM accordingly:
          CNN_FEATURE_DIM = 256 * (IMAGE_HEIGHT // 8)
        - The number of output timesteps equals IMAGE_WIDTH // 8 = 20.
          CTC requires: num_timesteps >= max_label_length in your dataset.
    """

    CNN_FEATURE_DIM = 256 * (IMAGE_HEIGHT // 8)   # = 1536 for IMAGE_HEIGHT=48

    def __init__(
        self,
        num_output_classes: int,
        transformer_dim:     int   = TRANSFORMER_D_MODEL,
        num_attention_heads: int   = TRANSFORMER_NUM_HEADS,
        num_encoder_layers:  int   = TRANSFORMER_NUM_LAYERS,
        dropout_rate:        float = TRANSFORMER_DROPOUT
    ):
        super().__init__()

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=32,  out_channels=64,  stride=2),
            ResidualBlock(in_channels=64,  out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, stride=1),
        )

        self.feature_projection = nn.Linear(self.CNN_FEATURE_DIM, transformer_dim)

        self.positional_encoder = SinusoidalPositionalEncoding(
            embedding_dim=transformer_dim,
            dropout_rate=dropout_rate
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_attention_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_encoder_layers
        )

        self.ctc_classifier = nn.Linear(transformer_dim, num_output_classes + 1)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_backbone(image_batch)

        batch_size, num_channels, feature_height, num_timesteps = cnn_features.shape
        cnn_features = cnn_features.permute(0, 3, 1, 2)
        sequence     = cnn_features.reshape(batch_size, num_timesteps,
                                            num_channels * feature_height)

        sequence = self.feature_projection(sequence)
        sequence = self.positional_encoder(sequence)
        sequence = self.transformer_encoder(sequence)

        logits = self.ctc_classifier(sequence)
        return logits.log_softmax(dim=2)


# =============================================================================
# DATASET LOADING AND SPLITTING
# =============================================================================
# FIX: The file list is shuffled and split FIRST, then three separate dataset
#      objects are created -- train with augmentation, val/test without.
#      This prevents the data leakage present in the original script.

supported_extensions = (".png", ".jpg", ".jpeg")
all_filenames = [
    f for f in os.listdir(DATASET_PATH)
    if f.endswith(supported_extensions)
    and all(c in char_to_index for c in os.path.splitext(f)[0])
]

random.shuffle(all_filenames)

num_train_samples = int(TRAIN_RATIO * len(all_filenames))
num_val_samples   = int(VAL_RATIO   * len(all_filenames))

train_filenames = all_filenames[:num_train_samples]
val_filenames   = all_filenames[num_train_samples:num_train_samples + num_val_samples]
test_filenames  = all_filenames[num_train_samples + num_val_samples:]

num_test_samples = len(test_filenames)

print(f"Split: {num_train_samples} train / {num_val_samples} val / {num_test_samples} test")

train_dataset = CaptchaSubsetDataset(DATASET_PATH, train_filenames, apply_augmentation=True)
val_dataset   = CaptchaSubsetDataset(DATASET_PATH, val_filenames,   apply_augmentation=False)
test_dataset  = CaptchaSubsetDataset(DATASET_PATH, test_filenames,  apply_augmentation=False)


# =============================================================================
# DATALOADERS
# =============================================================================

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True, collate_fn=collate_variable_length_batch
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=collate_variable_length_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=collate_variable_length_batch
)


# =============================================================================
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# =============================================================================

model = CaptchaResTransformer(num_output_classes=NUM_CLASSES).to(device)

ctc_loss_function = nn.CTCLoss(blank=BLANK_TOKEN_INDEX, zero_infinity=True)

optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_dataloader),
    epochs=NUM_EPOCHS,
    pct_start=0.1
)

grad_scaler = torch.amp.GradScaler("cuda")


# =============================================================================
# LIVE TRAINING DASHBOARD (matplotlib)
# =============================================================================

plt.ion()
dashboard_fig, dashboard_axes = plt.subplots(1, 3, figsize=(15, 4))
dashboard_fig.suptitle("CAPTCHA ResNet+Transformer -- Training Dashboard", fontsize=13)

epoch_train_losses   = []
epoch_val_losses     = []
epoch_word_accs      = []
epoch_char_accs      = []
epoch_learning_rates = []


# =============================================================================
# TRAINING LOOP
# =============================================================================

best_val_word_accuracy = 0.0

for epoch_index in range(NUM_EPOCHS):

    # -- TRAINING PHASE -------------------------------------------------------
    model.train()
    accumulated_train_loss = 0.0

    for image_batch, labels_concat, label_lengths, _ in tqdm(
        train_dataloader, desc=f"Epoch {epoch_index+1}/{NUM_EPOCHS} [train]"
    ):
        image_batch   = image_batch.to(device, non_blocking=True)
        labels_concat = labels_concat.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            model_output = model(image_batch)
            ctc_input    = model_output.permute(1, 0, 2)
            num_timesteps_per_sample = torch.full(
                (image_batch.size(0),),
                fill_value=model_output.size(1),
                dtype=torch.long,
                device=device
            )
            batch_loss = ctc_loss_function(
                ctc_input, labels_concat,
                num_timesteps_per_sample, label_lengths
            )

        grad_scaler.scale(batch_loss).backward()
        grad_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()

        accumulated_train_loss += batch_loss.item()

    mean_train_loss = accumulated_train_loss / len(train_dataloader)
    epoch_train_losses.append(mean_train_loss)
    epoch_learning_rates.append(lr_scheduler.get_last_lr()[0])

    # -- VALIDATION PHASE -----------------------------------------------------
    model.eval()
    accumulated_val_loss   = 0.0
    all_val_predictions    = []
    all_val_target_strings = []

    with torch.no_grad():
        for image_batch, labels_concat, label_lengths, label_strings in tqdm(
            val_dataloader, desc=f"Epoch {epoch_index+1}/{NUM_EPOCHS} [val]"
        ):
            image_batch   = image_batch.to(device, non_blocking=True)
            labels_concat = labels_concat.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                model_output = model(image_batch)
                ctc_input    = model_output.permute(1, 0, 2)
                num_timesteps_per_sample = torch.full(
                    (image_batch.size(0),),
                    fill_value=model_output.size(1),
                    dtype=torch.long, device=device
                )
                batch_loss = ctc_loss_function(
                    ctc_input, labels_concat,
                    num_timesteps_per_sample, label_lengths
                )

            accumulated_val_loss += batch_loss.item()

            val_predictions = greedy_ctc_decode(model_output.permute(1, 0, 2))
            all_val_predictions.extend(val_predictions)
            all_val_target_strings.extend(label_strings)

    mean_val_loss = accumulated_val_loss / len(val_dataloader)
    epoch_val_losses.append(mean_val_loss)

    val_word_accuracy, val_char_accuracy = compute_word_and_char_accuracy(
        all_val_predictions, all_val_target_strings
    )
    epoch_word_accs.append(val_word_accuracy)
    epoch_char_accs.append(val_char_accuracy)

    # -- EPOCH SUMMARY --------------------------------------------------------
    print(
        f"\nEpoch {epoch_index+1:02d} | "
        f"Train Loss: {mean_train_loss:.4f} | "
        f"Val Loss: {mean_val_loss:.4f} | "
        f"Word Acc: {val_word_accuracy*100:.2f}% | "
        f"Char Acc: {val_char_accuracy*100:.2f}% | "
        f"LR: {epoch_learning_rates[-1]:.2e}"
    )

    print("  Sample predictions (pred -> target):")
    for predicted, target in zip(all_val_predictions[:5], all_val_target_strings[:5]):
        match_symbol = "V" if predicted == target else "X"
        print(f"    {match_symbol} '{predicted}' -> '{target}'")

    if val_word_accuracy > best_val_word_accuracy:
        best_val_word_accuracy = val_word_accuracy
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  * New best model saved -- Word Acc: {val_word_accuracy*100:.2f}%")

    # -- UPDATE LIVE DASHBOARD ------------------------------------------------
    epoch_range = range(1, len(epoch_train_losses) + 1)

    dashboard_axes[0].clear()
    dashboard_axes[0].plot(epoch_range, epoch_train_losses, label="Train", color="#4C9BE8")
    dashboard_axes[0].plot(epoch_range, epoch_val_losses,   label="Val",   color="#E8674C")
    dashboard_axes[0].set_title("CTC Loss")
    dashboard_axes[0].set_xlabel("Epoch")
    dashboard_axes[0].legend()

    dashboard_axes[1].clear()
    dashboard_axes[1].plot(epoch_range, [a * 100 for a in epoch_word_accs],
                           label="Word Acc %", color="#4CE87A")
    dashboard_axes[1].plot(epoch_range, [a * 100 for a in epoch_char_accs],
                           label="Char Acc %", color="#C84CE8", linestyle="--")
    dashboard_axes[1].set_title("Accuracy")
    dashboard_axes[1].set_xlabel("Epoch")
    dashboard_axes[1].set_ylim(0, 100)
    dashboard_axes[1].legend()

    dashboard_axes[2].clear()
    dashboard_axes[2].plot(epoch_range, epoch_learning_rates, color="#E8C84C")
    dashboard_axes[2].set_title("Learning Rate (OneCycleLR)")
    dashboard_axes[2].set_xlabel("Epoch")

    dashboard_fig.tight_layout()
    plt.pause(0.01)

plt.ioff()


# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================

print("\n===== FINAL TEST EVALUATION =====")
model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
model.eval()

greedy_test_predictions      = []
beam_search_test_predictions = []
test_target_strings          = []

with torch.no_grad():
    for image_batch, labels_concat, label_lengths, label_strings in tqdm(
        test_dataloader, desc="Test evaluation"
    ):
        image_batch = image_batch.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            model_output = model(image_batch)

        log_probs_sequence = model_output.permute(1, 0, 2)

        greedy_test_predictions.extend(greedy_ctc_decode(log_probs_sequence))
        beam_search_test_predictions.extend(
            beam_search_ctc_decode(log_probs_sequence, beam_width=BEAM_SEARCH_WIDTH)
        )
        test_target_strings.extend(label_strings)

greedy_word_acc, greedy_char_acc = compute_word_and_char_accuracy(
    greedy_test_predictions, test_target_strings
)
beam_word_acc, beam_char_acc = compute_word_and_char_accuracy(
    beam_search_test_predictions, test_target_strings
)

print(f"\nGreedy decode  -> Word Acc: {greedy_word_acc*100:.2f}%  |  Char Acc: {greedy_char_acc*100:.2f}%")
print(f"Beam search(5) -> Word Acc: {beam_word_acc*100:.2f}%  |  Char Acc: {beam_char_acc*100:.2f}%")

print("\nSample predictions (Greedy | Beam Search | Target):")
for greedy_pred, beam_pred, target in zip(
    greedy_test_predictions[:10],
    beam_search_test_predictions[:10],
    test_target_strings[:10]
):
    print(f"  Greedy: '{greedy_pred}'  |  Beam: '{beam_pred}'  |  Target: '{target}'")

plt.show()
