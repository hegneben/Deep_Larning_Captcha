# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:35:55 2026

@author: Luka Ilisevic
"""

"""
Per-Distortion Visual Dataset Generator
========================================
Generates isolated single-distortion image sets for visual demonstration.
No model loading — pure image generation, runs in ~30 seconds.

Output structure on Desktop:
    CAPTCHA_Visual_Demo/
        blur/
            level_1_radius_0.3/   ← 5 images
            level_2_radius_0.7/
            level_3_radius_1.1/
            level_4_radius_1.5/
            level_5_radius_2.0/
        noise/    (same structure)
        rotation/ (same structure)
        lines/    (same structure)
        occlusion/(same structure)
        clean/    ← original 5 images for reference

Usage:
    python generate_visual_dataset.py
        --clean "data\\synthetic\\clean"

Optional:
    --output "C:\\Users\\YourName\\Desktop"   (default: auto-detect Desktop)
"""

import os
import random
import string
import argparse
import platform
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ─── DISTORTION FUNCTIONS ────────────────────────────────────────────────────

def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(img: Image.Image, std: float) -> Image.Image:
    arr   = np.array(img).astype(np.int16)
    rng   = np.random.default_rng(42)
    noise = rng.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_rotation(img: Image.Image, max_degrees: float, seed: int) -> Image.Image:
    rng   = random.Random(seed)
    angle = rng.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def apply_lines(img: Image.Image, count: int, seed: int) -> Image.Image:
    img  = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    rng  = random.Random(seed)
    for _ in range(count):
        draw.line(
            (0, rng.randint(0, h), w, rng.randint(0, h)),
            fill="black",
            width=rng.randint(1, 3),
        )
    return img


def apply_occlusion(img: Image.Image, count: int, seed: int) -> Image.Image:
    img  = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    rng  = random.Random(seed)
    block_w = int(w * 0.12)
    block_h = int(h * 0.35)
    for _ in range(count):
        x = rng.randint(0, max(1, w - block_w))
        y = rng.randint(0, max(1, h - block_h))
        draw.rectangle([x, y, x + block_w, y + block_h], fill="black")
    return img


# ─── DISTORTION CONFIG ───────────────────────────────────────────────────────
# (name, param_label, param_values, apply_fn)

DISTORTIONS = [
    (
        "blur",
        "radius",
        [0.3, 0.7, 1.1, 1.5, 2.0],
        lambda img, v, seed: apply_blur(img, v),
    ),
    (
        "noise",
        "std",
        [5, 10, 18, 28, 40],
        lambda img, v, seed: apply_noise(img, v),
    ),
    (
        "rotation",
        "degrees",
        [3, 6, 10, 14, 18],
        lambda img, v, seed: apply_rotation(img, v, seed),
    ),
    (
        "lines",
        "count",
        [1, 3, 5, 7, 10],
        lambda img, v, seed: apply_lines(img, v, seed),
    ),
    (
        "occlusion",
        "blocks",
        [1, 2, 3, 4, 5],
        lambda img, v, seed: apply_occlusion(img, v, seed),
    ),
]

# The 5 labels we've been using throughout the project
TARGET_LABELS = ["RwUUE", "Q12gk", "pelrH", "g9soP", "IjQgf"]

SUPPORTED = {".png", ".jpg", ".jpeg"}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def find_desktop() -> Path:
    """Auto-detect the Desktop folder cross-platform."""
    system = platform.system()
    if system == "Windows":
        # Try USERPROFILE first, fall back to HOME
        base = os.environ.get("USERPROFILE") or os.environ.get("HOME", "~")
        desktop = Path(base) / "Desktop"
    elif system == "Darwin":
        desktop = Path.home() / "Desktop"
    else:
        desktop = Path.home() / "Desktop"

    if not desktop.exists():
        # Fall back to home directory if Desktop doesn't exist
        desktop = Path.home()
        print(f"  ⚠  Desktop not found — using home directory: {desktop}")

    return desktop


def label_from_path(p: Path) -> str:
    stem = p.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def collect_source_images(clean_dir: Path) -> list[Path]:
    """Find the 5 target label images from the clean synthetic folder."""
    all_images = sorted(
        p for p in clean_dir.iterdir()
        if p.suffix.lower() in SUPPORTED
    )

    # Match our 5 specific labels
    matched = []
    for img_path in all_images:
        lbl = label_from_path(img_path)
        if lbl in TARGET_LABELS and lbl not in [label_from_path(m) for m in matched]:
            matched.append(img_path)
        if len(matched) == len(TARGET_LABELS):
            break

    if not matched:
        raise FileNotFoundError(
            f"Could not find any of {TARGET_LABELS} in {clean_dir}\n"
            "Make sure --clean points to data\\synthetic\\clean"
        )

    return matched


# ─── MAIN ────────────────────────────────────────────────────────────────────

def generate(args):
    clean_dir  = Path(args.clean)
    output_dir = Path(args.output) if args.output else find_desktop()
    demo_dir   = output_dir / "CAPTCHA_Visual_Demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Per-Distortion Visual Dataset Generator")
    print(f"{'='*55}")
    print(f"  Clean images : {clean_dir}")
    print(f"  Output       : {demo_dir}")
    print(f"{'='*55}\n")

    # Collect source images
    sources = collect_source_images(clean_dir)
    print(f"Found {len(sources)} source images:")
    for p in sources:
        print(f"  {p.name}  →  label: {label_from_path(p)}")
    print()

    # Save clean reference copies
    clean_out = demo_dir / "clean"
    clean_out.mkdir(exist_ok=True)
    for src in sources:
        img = Image.open(src).convert("RGB").resize((160, 48))
        img.save(clean_out / src.name)
    print(f"✔  Saved {len(sources)} clean reference images → clean/")

    # Generate per-distortion images
    total_saved = 0
    for dist_name, param_label, param_values, apply_fn in DISTORTIONS:
        dist_dir = demo_dir / dist_name
        dist_dir.mkdir(exist_ok=True)

        print(f"\n── {dist_name.upper()} ──────────────────────────────────")

        for level_idx, param_val in enumerate(param_values):
            level_name = f"level_{level_idx + 1}_{param_label}_{param_val}"
            level_dir  = dist_dir / level_name
            level_dir.mkdir(exist_ok=True)

            saved = 0
            for src in sources:
                label = label_from_path(src)
                img   = Image.open(src).convert("RGB").resize((160, 48))
                seed  = hash(f"{dist_name}_{param_val}_{label}") % (2**31)

                try:
                    distorted = apply_fn(img, param_val, seed)
                    out_path  = level_dir / src.name
                    distorted.save(out_path)
                    saved += 1
                except Exception as e:
                    print(f"    ⚠  {src.name}: {e}")

            total_saved += saved
            print(f"  ✔  {level_name:<35}  {saved} images saved")

    print(f"\n{'='*55}")
    print(f"  Done! {total_saved} distorted images generated")
    print(f"  Folder: {demo_dir}")
    print(f"{'='*55}\n")
    print("  Folder structure:")
    print("    CAPTCHA_Visual_Demo/")
    print("      clean/              ← original 5 images")
    for dist_name, param_label, param_values, _ in DISTORTIONS:
        print(f"      {dist_name}/")
        for i, v in enumerate(param_values):
            print(f"        level_{i+1}_{param_label}_{v}/   ← 5 images")
    print()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-distortion visual demo images (no model needed)."
    )
    parser.add_argument(
        "--clean", required=True,
        help="Path to data\\synthetic\\clean folder"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output parent folder (default: Desktop)"
    )
    generate(parser.parse_args())