# -*- coding: utf-8 -*-
"""
This program generates the experimental ladder dataset based on defined changes
"""

import csv
import random
import argparse

import numpy as np

from .ladder_config import (
    OUTPUT_DIR,
    NUM_SAMPLES_PER_LEVEL,
    LADDER_LEVELS,
)

from .ladder_renderer import render_ladder_sample


def generate_dataset(num_samples_per_level, seed):
    random.seed(seed)
    np.random.seed(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = OUTPUT_DIR / "ladder_metadata.csv"

    with open(metadata_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "filename",
            "label",
            "ladder_level",
            "ladder_name",
            "font",
        ])

        for level, level_name in LADDER_LEVELS.items():
            level_dir = OUTPUT_DIR / f"level_{level}_{level_name}"
            level_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_samples_per_level):
                image, label, font_name = render_ladder_sample(level)

                filename = f"{i:06d}_{label}.png"
                save_path = level_dir / filename

                image.save(save_path)

                writer.writerow([
                    str(save_path.relative_to(OUTPUT_DIR)),
                    label,
                    level,
                    level_name,
                    font_name,
                ])

            print(f"Generated level {level}: {level_name}")

    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES_PER_LEVEL,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_dataset(
        num_samples_per_level=args.num_samples,
        seed=args.seed,
    )