"""Create side-by-side preview images for distortion levels 1, 2, and 3.

Example:
    python -m src.distortions.preview_distortions \
        --image tests/Lin_Modell_new.png \
        --output plots/distortion_preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .pipeline import DistortionPipeline


def make_preview(image_path: Path, output_path: Path, seed: int = 42) -> Path:
    with Image.open(image_path) as image:
        clean = image.convert("RGB")

    panels = [("clean", clean)]
    for level in (1, 2, 3):
        panels.append((f"level {level}", DistortionPipeline(level=level, seed=seed).apply(clean)))

    panel_width, panel_height = clean.size
    label_height = 30
    canvas = Image.new("RGB", (panel_width * len(panels), panel_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for i, (label, panel) in enumerate(panels):
        x = i * panel_width
        canvas.paste(panel, (x, label_height))
        draw.text((x + 10, 8), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a distortion preview image.")
    parser.add_argument("--image", type=Path, required=True, help="Input CAPTCHA image.")
    parser.add_argument("--output", type=Path, required=True, help="Output preview image path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = make_preview(args.image, args.output, seed=args.seed)
    print(f"Preview written to: {output}")


if __name__ == "__main__":
    main()
