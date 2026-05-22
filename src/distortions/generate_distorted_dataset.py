"""Generate distorted CAPTCHA datasets for levels 1, 2, and 3.

Example:
    python -m src.distortions.generate_distorted_dataset \
        --input-dir data/clean_captchas \
        --output-dir data/distorted \
        --levels 1 2 3 \
        --seed 42

Input labels are inferred from file names by default:
    abc123.png -> label abc123

A metadata CSV is written to the output folder with columns:
    image_id,source_path,output_path,label,distortion_level,distortion_name,seed
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .levels import get_distortion_level
from .pipeline import DistortionPipeline


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(input_dir: Path) -> list[Path]:
    """Return image paths under ``input_dir`` in deterministic order."""
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def infer_label_from_filename(path: Path) -> str:
    """Infer CAPTCHA text from the file stem.

    Examples:
        ABC12.png -> ABC12
        ABC12_001.png -> ABC12_001
    """
    return path.stem


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    levels: Iterable[int],
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Generate distorted copies of all images and return metadata CSV path."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    metadata_path = output_dir / "distortion_metadata.csv"
    rows: list[dict[str, str | int]] = []

    for level in levels:
        cfg = get_distortion_level(level)
        level_dir = output_dir / f"level_{level}_{cfg.name}"
        level_dir.mkdir(parents=True, exist_ok=True)

        for index, source_path in enumerate(image_paths):
            relative = source_path.relative_to(input_dir)
            target_path = level_dir / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)

            item_seed = seed + (level * 1_000_000) + index

            if target_path.exists() and not overwrite:
                status = "skipped_existing"
            else:
                try:
                    with Image.open(source_path) as image:
                        distorted = DistortionPipeline(level=level, seed=item_seed).apply(image)
                        distorted.save(target_path)
                    status = "created"
                except UnidentifiedImageError:
                    status = "invalid_image"

            rows.append(
                {
                    "image_id": source_path.stem,
                    "source_path": str(source_path),
                    "output_path": str(target_path),
                    "label": infer_label_from_filename(source_path),
                    "distortion_level": level,
                    "distortion_name": cfg.name,
                    "seed": item_seed,
                    "status": status,
                }
            )

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "source_path",
                "output_path",
                "label",
                "distortion_level",
                "distortion_name",
                "seed",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate distorted CAPTCHA benchmark datasets.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with clean CAPTCHA images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for distorted output images.")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Distortion levels to generate. Default: 1 2 3.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing distorted images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        levels=args.levels,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Distorted dataset metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
