# Distortion Module

This module creates controlled CAPTCHA distortion datasets for robustness experiments.

## Files

```text
src/distortions/
├── __init__.py
├── levels.py                    # Level 1/2/3 parameter definitions
├── pipeline.py                  # Reusable DistortionPipeline class
├── generate_distorted_dataset.py # Batch CLI generator
└── preview_distortions.py        # Creates a visual preview image
```

## Generate a distorted benchmark

From the repository root:

```bash
python -m src.distortions.generate_distorted_dataset \
  --input-dir data/clean_captchas \
  --output-dir data/distorted \
  --levels 1 2 3 \
  --seed 42
```

This creates:

```text
data/distorted/
├── distortion_metadata.csv
├── level_1_mild/
├── level_2_moderate/
└── level_3_severe/
```

The metadata CSV stores image path, label, distortion level, distortion name, and seed.

## Preview the distortion levels

```bash
python -m src.distortions.preview_distortions \
  --image tests/Lin_Modell_new.png \
  --output plots/distortion_preview.png
```

## Use from Python

```python
from PIL import Image
from src.distortions import DistortionPipeline

image = Image.open("captcha.png")
distorted = DistortionPipeline(level=2, seed=42).apply(image)
distorted.save("captcha_level_2.png")
```

## Level meaning

| Level | Name | Intended meaning |
|---|---|---|
| 1 | mild | Humans should read almost all samples correctly. |
| 2 | moderate | Recognition should begin to degrade. |
| 3 | severe | Tests the failure boundary for model and humans. |

