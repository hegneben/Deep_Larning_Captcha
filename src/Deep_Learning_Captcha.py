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

       → must print True, otherwise training runs on CPU (~10x slower)

    3. If needed, install/update PyTorch with matching CUDA support:
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

       Check CUDA version first:
           nvidia-smi

       Then use the matching PyTorch build:
           CUDA 11.8  →  .../whl/cu118
           CUDA 12.1  →  .../whl/cu121
           CUDA 12.4  →  .../whl/cu124

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
      silently — check your dataset for special characters beforehand.
    - The best model checkpoint is saved to CHECKPOINT_PATH after every
      epoch that achieves a new highest validation word accuracy.
    - Beam search decoding is slower than greedy decoding (runs only
      during the final test evaluation, not during training).
    - If VRAM runs out, reduce BATCH_SIZE (e.g. 64 or 32).
    - Image size (IMAGE_WIDTH x IMAGE_HEIGHT) affects model architecture.
      Do NOT change these values after training has started — the CNN
      projection layer dimension (CNN_FEATURE_DIM) depends on them.
"""

import os
import math
import string

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION — edit these values before running
# =============================================================================

DATASET_PATH   = r"C:\Users\Benedikt\.cache\kagglehub\datasets\parsasam\captcha-dataset\versions\1"
CHECKPOINT_PATH = "best_captcha_model.pth"

# Image dimensions — all images are resized to this before feeding the model.
# Must be divisible by 8 (three stride-2 CNN blocks halve H and W each time).
IMAGE_WIDTH  = 160   # pixels — controls the number of output timesteps (W / 8 = 20)
IMAGE_HEIGHT = 48    # pixels — controls CNN feature height (H / 8 = 6)

# Training hyperparameters
BATCH_SIZE      = 128   # reduce to 64 or 32 if VRAM is insufficient
NUM_EPOCHS      = 20
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4  # L2 regularisation — prevents overfitting
GRAD_CLIP_NORM  = 5.0   # max gradient norm — prevents exploding gradients

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO is derived automatically: 1 - TRAIN_RATIO - VAL_RATIO

# Transformer architecture
TRANSFORMER_D_MODEL    = 256   # embedding dimension (must be divisible by NUM_HEADS)
TRANSFORMER_NUM_HEADS  = 8     # attention heads — each focuses on a different feature
TRANSFORMER_NUM_LAYERS = 4     # stacked encoder layers — more = stronger, slower
TRANSFORMER_DROPOUT    = 0.1   # regularisation inside the Transformer

# Beam search width for final test decoding (higher = more accurate but slower)
BEAM_SEARCH_WIDTH = 5


# =============================================================================
# DEVICE SETUP
# =============================================================================

# Automatically use GPU if available. On CPU, training is feasible but slow.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lets cuDNN auto-select the fastest convolution algorithm for fixed input sizes.
torch.backends.cudnn.benchmark = True


# =============================================================================
# CHARACTER SET
# =============================================================================
# Defines which characters the model can predict.
# NOTE: Every character in your image filenames must appear here.
#       Add or remove characters to match your dataset.

CHARSET = string.ascii_lowercase + string.ascii_uppercase + string.digits
# e.g. "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Map each character to an integer index (used to encode labels as tensors)
char_to_index = {char: idx for idx, char in enumerate(CHARSET)}

# Reverse map: integer index back to character (used during decoding)
index_to_char = {idx: char for char, idx in char_to_index.items()}

# CTC blank token index — must be outside the charset range.
# CTC loss uses this to separate repeated characters and pad sequences.
BLANK_TOKEN_INDEX = len(CHARSET)   # = 62 for the default charset

NUM_CLASSES = len(CHARSET)        # = 62 (not counting the blank token)


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================
# Two separate pipelines: one with data augmentation (training), one without (val/test).
# Augmentation artificially increases dataset diversity by randomly changing
# brightness and contrast, making the model more robust to real-world variation.

# NOTE: ColorJitter parameters (brightness, contrast, saturation) are mild.
#       Increasing them too much can make the text unreadable for the model.

transform_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # Normalize maps [0,1] pixel values to [-1,1], which helps training stability.
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def resize_image_to_fixed_size(pil_image: Image.Image) -> Image.Image:
    """
    Resize a PIL image to (IMAGE_WIDTH x IMAGE_HEIGHT) in RGB mode.

    NOTE: This uses simple stretching (no padding). If your CAPTCHAs have
    very different aspect ratios, consider using pad_to_target() instead
    to preserve proportions and avoid distortion.
    """
    return pil_image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))


# =============================================================================
# DATASET — loads all images into RAM for fast training
# =============================================================================

class CaptchaRAMDataset(Dataset):
    """
    Loads all CAPTCHA images into RAM at startup for maximum training speed.

    Each image filename (without extension) is treated as the ground-truth label.
    Example: "Ab3f.png" → label "Ab3f"

    IMPORTANT:
        - Only files with extensions .png, .jpg, .jpeg are loaded.
        - Characters outside CHARSET are not supported and will cause an error.
          Pre-filter your dataset if it contains special characters.
        - With augment=True, PIL images are stored (not tensors) so that
          ColorJitter is applied freshly on every __getitem__ call.
          This uses more memory but gives better training diversity.
        - With augment=False, images are pre-converted to tensors for speed.
    """

    def __init__(self, image_directory: str, apply_augmentation: bool = False):
        self.samples            = []
        self.apply_augmentation = apply_augmentation

        supported_extensions = (".png", ".jpg", ".jpeg")
        image_filenames = [
            filename for filename in os.listdir(image_directory)
            if filename.endswith(supported_extensions)
        ]

        print(f"Loading {len(image_filenames)} images into RAM...")

        for filename in tqdm(image_filenames):
            image_path  = os.path.join(image_directory, filename)
            label_string = os.path.splitext(filename)[0]   # e.g. "Ab3f"

            pil_image = Image.open(image_path).convert("RGB")
            resized_image = resize_image_to_fixed_size(pil_image)

            if apply_augmentation:
                # Store PIL image — transform applied at __getitem__ time
                self.samples.append((resized_image, label_string))
            else:
                # Pre-convert to tensor for fast inference
                image_tensor = transform_eval(resized_image)
                label_tensor = torch.tensor(
                    [char_to_index[c] for c in label_string],
                    dtype=torch.long
                )
                self.samples.append((image_tensor, label_tensor, label_string))

        print(f"Dataset fully loaded — {len(self.samples)} samples in RAM.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_index: int):
        if self.apply_augmentation:
            pil_image, label_string = self.samples[sample_index]
            # Apply random augmentation fresh on every access
            image_tensor = transform_train(pil_image)
            label_tensor = torch.tensor(
                [char_to_index[c] for c in label_string],
                dtype=torch.long
            )
            return image_tensor, label_tensor, label_string
        else:
            return self.samples[sample_index]


# =============================================================================
# COLLATE FUNCTION — batches samples of variable label length for CTC
# =============================================================================

def collate_variable_length_batch(batch):
    """
    Combines a list of (image, label_tensor, label_string) tuples into a batch.

    CTC loss requires all label tensors to be concatenated into a 1D tensor
    along with their individual lengths — it does NOT use padding.

    Returns:
        image_batch   : (B, C, H, W) stacked image tensors
        labels_concat : 1D tensor of all concatenated label indices
        label_lengths : 1D tensor with the length of each label in the batch
        label_strings : list of raw label strings (for accuracy evaluation)
    """
    image_list, label_list, label_string_list = zip(*batch)

    image_batch   = torch.stack(image_list)
    label_lengths = torch.tensor([len(label) for label in label_list], dtype=torch.long)
    labels_concat = torch.cat(label_list)

    return image_batch, labels_concat, label_lengths, list(label_string_list)


# =============================================================================
# CTC DECODING — converting model output back to text strings
# =============================================================================

def greedy_ctc_decode(log_probabilities: torch.Tensor) -> list[str]:
    """
    Greedy (argmax) CTC decoding — fast but slightly less accurate than beam search.

    Algorithm:
        1. Take the most likely class at each timestep (argmax).
        2. Collapse consecutive repeated indices.
        3. Remove blank tokens.

    Args:
        log_probabilities: shape (T, B, C) — T timesteps, B batch, C classes+blank

    Returns:
        List of decoded strings, one per sample in the batch.

    NOTE: This is used during training validation for speed.
          Use beam_search_ctc_decode() for the final evaluation.
    """
    # Shape: (T, B) — index of most likely class at each timestep
    most_likely_indices = log_probabilities.argmax(dim=2)
    most_likely_indices = most_likely_indices.permute(1, 0)   # → (B, T)

    decoded_strings = []
    for sequence in most_likely_indices:
        characters     = []
        previous_index = BLANK_TOKEN_INDEX

        for current_index in sequence.tolist():
            # Keep character only if it's not blank AND not a repeat of the previous
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
    Beam search CTC decoding — more accurate than greedy, but slower.

    Keeps the top `beam_width` candidate sequences at each timestep,
    rather than committing to just the argmax character.

    Args:
        log_probabilities : shape (T, B, C)
        beam_width        : number of candidates to keep (higher = more accurate)

    Returns:
        List of decoded strings, one per sample in the batch.

    NOTE: Pure Python implementation — no external library needed.
          For very long sequences or large batches, this can be slow.
          Consider the `ctcdecode` library for production use.
    """
    log_probs_cpu = log_probabilities.cpu().float()
    num_timesteps, batch_size, num_output_classes = log_probs_cpu.shape

    decoded_strings = []

    for batch_index in range(batch_size):
        # Beam state: maps prefix_tuple → (log_prob_ending_in_blank, log_prob_ending_in_non_blank)
        beam_candidates = {(): (0.0, float("-inf"))}

        for timestep in range(num_timesteps):
            log_probs_at_t = log_probs_cpu[timestep, batch_index]   # shape (C,)
            updated_beam   = {}

            for prefix, (prob_blank, prob_non_blank) in beam_candidates.items():
                # --- Extend with blank ---
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

                # --- Extend with each non-blank character ---
                for char_index in range(num_output_classes):
                    if char_index == BLANK_TOKEN_INDEX:
                        continue

                    extended_prefix = prefix + (char_index,)

                    # If the new character equals the last character in the prefix,
                    # it can only come from a blank path (otherwise it's a repeat collapse)
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

            # Keep only top beam_width candidates by total log-probability
            def total_log_prob(prob_pair):
                return torch.logaddexp(
                    torch.tensor(prob_pair[0]), torch.tensor(prob_pair[1])
                ).item()

            beam_candidates = dict(
                sorted(updated_beam.items(), key=lambda item: total_log_prob(item[1]), reverse=True)
                [:beam_width]
            )

        # Select the best prefix from the final beam
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
    Computes two accuracy metrics over a batch of predictions.

    Word accuracy: fraction of samples where the entire string matches exactly.
    Char accuracy: fraction of individual characters that match (position-aligned).

    NOTE: Word accuracy is the stricter metric — even one wrong character
          in a CAPTCHA causes a full miss. Char accuracy gives partial credit
          and is useful for diagnosing how close the model is getting.

    Returns:
        (word_accuracy, char_accuracy) — both in range [0.0, 1.0]
    """
    correct_full_strings = 0
    correct_characters   = 0
    total_characters     = 0

    for prediction, target in zip(predicted_strings, target_strings):
        # CAPTCHAs are case-sensitive in V2; compare as-is
        if prediction == target.lower():
            correct_full_strings += 1

        # Character-level accuracy (align by position, ignore length mismatch tail)
        for predicted_char, target_char in zip(prediction, target.lower()):
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

    The skip connection adds the input directly to the output, which:
    - Prevents the vanishing gradient problem in deep networks
    - Allows the model to learn "what to change" rather than "what to output"

    If stride > 1 or channel count changes, a 1x1 conv downsamples the identity
    path to match the output shape before adding.

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

        # Downsample the identity path when shape changes
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

        # Add skip connection then activate
        return self.relu(out + identity)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds positional information to Transformer input embeddings.

    Transformers have no built-in sense of order (they process all positions
    in parallel). This module injects sine/cosine signals at different
    frequencies to let the model distinguish position 1 from position 20, etc.

    Args:
        embedding_dim : must match the Transformer d_model
        max_sequence_length : maximum number of timesteps expected
        dropout_rate  : applied after adding the positional encoding
    """

    def __init__(self, embedding_dim: int, max_sequence_length: int = 256,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Pre-compute fixed positional encoding matrix of shape (max_len, d_model)
        positional_encoding = torch.zeros(max_sequence_length, embedding_dim)
        positions           = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        frequency_divisors  = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        positional_encoding[:, 0::2] = torch.sin(positions * frequency_divisors)
        positional_encoding[:, 1::2] = torch.cos(positions * frequency_divisors)

        # Register as buffer so it moves with .to(device) but isn't a trainable param
        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, T, d_model)
        embeddings = embeddings + self.positional_encoding[:, :embeddings.size(1)]
        return self.dropout(embeddings)


class CaptchaResTransformer(nn.Module):
    """
    Full CAPTCHA recognition model: ResNet CNN + Transformer encoder + CTC head.

    Architecture overview:
        1. ResNet CNN backbone — extracts local visual features from the image
        2. Feature reshape    — converts the 2D feature map to a 1D sequence
        3. Linear projection  — maps CNN features to Transformer embedding dim
        4. Positional encoding — tells the Transformer which position each feature is at
        5. Transformer encoder — models long-range dependencies across the sequence
        6. CTC head           — predicts a probability distribution over characters
                                 at each timestep

    Input shape  : (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH) = (B, 3, 48, 160)
    Output shape : (B, num_timesteps, NUM_CLASSES + 1)  — log-softmax probabilities

    NOTE:
        - CNN_FEATURE_DIM (= 256 * 6 = 1536) is derived from IMAGE_HEIGHT.
          If IMAGE_HEIGHT changes, update CNN_FEATURE_DIM accordingly:
          CNN_FEATURE_DIM = 256 * (IMAGE_HEIGHT // 8)
        - The number of output timesteps equals IMAGE_WIDTH // 8 = 20.
          CTC requires: num_timesteps >= max_label_length in your dataset.
    """

    # Derived constant — must match IMAGE_HEIGHT // 8 * 256
    CNN_FEATURE_DIM = 256 * (IMAGE_HEIGHT // 8)   # = 1536 for IMAGE_HEIGHT=48

    def __init__(
        self,
        num_output_classes: int,
        transformer_dim:    int = TRANSFORMER_D_MODEL,
        num_attention_heads: int = TRANSFORMER_NUM_HEADS,
        num_encoder_layers: int = TRANSFORMER_NUM_LAYERS,
        dropout_rate:       float = TRANSFORMER_DROPOUT
    ):
        super().__init__()

        # ── 1. ResNet CNN Backbone ──────────────────────────────────────────
        # Reduces spatial size: H=48→6, W=160→20 (three stride-2 blocks)
        # Increases feature channels: 3→32→64→128→256
        self.cnn_backbone = nn.Sequential(
            # Stem: initial feature extraction without downsampling
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            ResidualBlock(in_channels=32,  out_channels=64,  stride=2),  # H: 48→24
            ResidualBlock(in_channels=64,  out_channels=128, stride=2),  # H: 24→12
            ResidualBlock(in_channels=128, out_channels=256, stride=2),  # H: 12→6
            ResidualBlock(in_channels=256, out_channels=256, stride=1),  # refinement
        )

        # ── 2 & 3. Feature reshape + linear projection ──────────────────────
        # After CNN: (B, 256, 6, 20)
        # After reshape: (B, 20, 256*6=1536)
        # After projection: (B, 20, transformer_dim=256)
        self.feature_projection = nn.Linear(self.CNN_FEATURE_DIM, transformer_dim)

        # ── 4. Positional encoding ──────────────────────────────────────────
        self.positional_encoder = SinusoidalPositionalEncoding(
            embedding_dim=transformer_dim,
            dropout_rate=dropout_rate
        )

        # ── 5. Transformer encoder ──────────────────────────────────────────
        # norm_first=True means Pre-LayerNorm, which is more stable than Post-LN
        # and generally converges faster for recognition tasks.
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_attention_heads,
            dim_feedforward=transformer_dim * 4,   # standard FFN expansion ratio
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_encoder_layers
        )

        # ── 6. CTC classification head ──────────────────────────────────────
        # +1 for the CTC blank token (index = NUM_CLASSES = 62)
        self.ctc_classifier = nn.Linear(transformer_dim, num_output_classes + 1)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_batch : (B, 3, H, W) normalised image tensor

        Returns:
            log_softmax output of shape (B, T, C+1) where
            T = number of timesteps (= W // 8 = 20) and
            C+1 = number of classes including blank token
        """
        # CNN: extract spatial features
        cnn_features = self.cnn_backbone(image_batch)   # (B, 256, 6, 20)

        # Reshape: treat image width as the sequence dimension
        batch_size, num_channels, feature_height, num_timesteps = cnn_features.shape
        cnn_features = cnn_features.permute(0, 3, 1, 2)             # (B, W, C, H)
        sequence     = cnn_features.reshape(batch_size, num_timesteps,
                                            num_channels * feature_height)
        # sequence shape: (B, 20, 1536)

        # Project to Transformer embedding dimension
        sequence = self.feature_projection(sequence)     # (B, 20, 256)

        # Add positional encoding so the Transformer knows the column order
        sequence = self.positional_encoder(sequence)     # (B, 20, 256)

        # Transformer: model relationships between all timesteps simultaneously
        sequence = self.transformer_encoder(sequence)    # (B, 20, 256)

        # Predict character probabilities at each timestep
        logits = self.ctc_classifier(sequence)           # (B, 20, 63)

        return logits.log_softmax(dim=2)


# =============================================================================
# DATASET LOADING AND SPLITTING
# =============================================================================

full_dataset = CaptchaRAMDataset(DATASET_PATH, apply_augmentation=True)

num_train_samples = int(TRAIN_RATIO * len(full_dataset))
num_val_samples   = int(VAL_RATIO   * len(full_dataset))
num_test_samples  = len(full_dataset) - num_train_samples - num_val_samples

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [num_train_samples, num_val_samples, num_test_samples]
)

print(f"Split: {num_train_samples} train / {num_val_samples} val / {num_test_samples} test")

# =============================================================================
# DATALOADERS
# =============================================================================
# pin_memory=True speeds up CPU→GPU transfers when using a CUDA device.
# num_workers=0 avoids multiprocessing issues on Windows; increase on Linux.

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

# CTC loss: handles variable-length labels without explicit alignment.
# zero_infinity=True silently ignores batches where the label is longer
# than the input sequence (which would produce -inf loss and NaN gradients).
ctc_loss_function = nn.CTCLoss(blank=BLANK_TOKEN_INDEX, zero_infinity=True)

# AdamW: like Adam but with decoupled weight decay — better regularisation.
optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# OneCycleLR: warm up LR to max_lr in the first 10% of training, then decay.
# This generally converges faster and to a better minimum than a fixed LR.
lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_dataloader),
    epochs=NUM_EPOCHS,
    pct_start=0.1   # 10% warmup
)

# AMP GradScaler: scales loss up before backward to prevent underflow in fp16,
# then scales gradients back down before the optimizer step.
# NOTE: Only effective on CUDA. On CPU, autocast is a no-op.
grad_scaler = torch.amp.GradScaler("cuda")


# =============================================================================
# LIVE TRAINING DASHBOARD (matplotlib)
# =============================================================================

plt.ion()
dashboard_fig, dashboard_axes = plt.subplots(1, 3, figsize=(15, 4))
dashboard_fig.suptitle("CAPTCHA ResNet+Transformer — Training Dashboard", fontsize=13)

epoch_train_losses = []
epoch_val_losses   = []
epoch_word_accs    = []
epoch_char_accs    = []
epoch_learning_rates = []


# =============================================================================
# TRAINING LOOP
# =============================================================================

best_val_word_accuracy = 0.0

for epoch_index in range(NUM_EPOCHS):

    # ── TRAINING PHASE ────────────────────────────────────────────────────────
    model.train()
    accumulated_train_loss = 0.0

    for image_batch, labels_concat, label_lengths, _ in tqdm(
        train_dataloader, desc=f"Epoch {epoch_index+1}/{NUM_EPOCHS} [train]"
    ):
        image_batch   = image_batch.to(device, non_blocking=True)
        labels_concat = labels_concat.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass — uses fp16 where safe, fp32 where needed
        with torch.amp.autocast("cuda"):
            model_output = model(image_batch)     # (B, T, C+1) log-softmax

            # CTC loss expects (T, B, C) — permute from (B, T, C)
            ctc_input = model_output.permute(1, 0, 2)

            # All sequences in the batch have the same number of timesteps
            num_timesteps_per_sample = torch.full(
                (image_batch.size(0),),
                fill_value=model_output.size(1),
                dtype=torch.long,
                device=device
            )

            batch_loss = ctc_loss_function(
                ctc_input,
                labels_concat,
                num_timesteps_per_sample,
                label_lengths
            )

        # Backward pass with gradient scaling (AMP)
        grad_scaler.scale(batch_loss).backward()

        # Unscale before clipping so clip threshold is in the original scale
        grad_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()

        accumulated_train_loss += batch_loss.item()

    mean_train_loss = accumulated_train_loss / len(train_dataloader)
    epoch_train_losses.append(mean_train_loss)
    epoch_learning_rates.append(lr_scheduler.get_last_lr()[0])

    # ── VALIDATION PHASE ──────────────────────────────────────────────────────
    model.eval()
    accumulated_val_loss    = 0.0
    all_val_predictions     = []
    all_val_target_strings  = []

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

            # Decode predictions with fast greedy decoding
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

    # ── EPOCH SUMMARY ─────────────────────────────────────────────────────────
    print(
        f"\nEpoch {epoch_index+1:02d} | "
        f"Train Loss: {mean_train_loss:.4f} | "
        f"Val Loss: {mean_val_loss:.4f} | "
        f"Word Acc: {val_word_accuracy*100:.2f}% | "
        f"Char Acc: {val_char_accuracy*100:.2f}% | "
        f"LR: {epoch_learning_rates[-1]:.2e}"
    )

    # Show a few example predictions to monitor training quality visually
    print("  Sample predictions (pred → target):")
    for predicted, target in zip(all_val_predictions[:5], all_val_target_strings[:5]):
        match_symbol = "✔" if predicted == target.lower() else "✘"
        print(f"    {match_symbol} '{predicted}' → '{target.lower()}'")

    # Save checkpoint whenever validation word accuracy improves
    if val_word_accuracy > best_val_word_accuracy:
        best_val_word_accuracy = val_word_accuracy
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  ★ New best model saved — Word Acc: {val_word_accuracy*100:.2f}%")

    # ── UPDATE LIVE DASHBOARD ──────────────────────────────────────────────────
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
# Reloads the best checkpoint and evaluates on the held-out test set
# using both greedy and beam search decoding for comparison.

print("\n===== FINAL TEST EVALUATION =====")
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

greedy_test_predictions    = []
beam_search_test_predictions = []
test_target_strings        = []

with torch.no_grad():
    for image_batch, labels_concat, label_lengths, label_strings in tqdm(
        test_dataloader, desc="Test evaluation"
    ):
        image_batch = image_batch.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            model_output = model(image_batch)

        log_probs_sequence = model_output.permute(1, 0, 2)   # (T, B, C+1)

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

print(f"\nGreedy decode  → Word Acc: {greedy_word_acc*100:.2f}%  |  Char Acc: {greedy_char_acc*100:.2f}%")
print(f"Beam search(5) → Word Acc: {beam_word_acc*100:.2f}%  |  Char Acc: {beam_char_acc*100:.2f}%")

print("\nSample predictions (Greedy | Beam Search | Target):")
for greedy_pred, beam_pred, target in zip(
    greedy_test_predictions[:10],
    beam_search_test_predictions[:10],
    test_target_strings[:10]
):
    print(f"  Greedy: '{greedy_pred}'  |  Beam: '{beam_pred}'  |  Target: '{target.lower()}'")

plt.show()
