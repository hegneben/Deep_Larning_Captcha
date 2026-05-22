"""Definitions for the three CAPTCHA distortion levels.

The goal is to keep the experiment controlled and repeatable:
- Level 1: mild distortion; humans should read almost everything.
- Level 2: moderate distortion; recognition should begin to degrade.
- Level 3: severe distortion; this is intended to expose model and human limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class DistortionLevel:
    """Parameter bundle for one distortion level."""

    level: int
    name: str
    description: str

    # Geometric changes
    max_rotation_degrees: float
    perspective_strength: float
    shear_strength: float

    # Pixel/image changes
    blur_radius: float
    gaussian_noise_std: float
    contrast_factor_min: float
    contrast_factor_max: float
    brightness_factor_min: float
    brightness_factor_max: float

    # Visual clutter
    line_count: int
    line_width_min: int
    line_width_max: int
    dot_count: int
    occlusion_count: int
    occlusion_size_fraction: float


DISTORTION_LEVELS: dict[int, DistortionLevel] = {
    1: DistortionLevel(
        level=1,
        name="mild",
        description="Small rotations, light blur, low noise, and a few thin lines.",
        max_rotation_degrees=4.0,
        perspective_strength=0.025,
        shear_strength=0.03,
        blur_radius=0.45,
        gaussian_noise_std=6.0,
        contrast_factor_min=0.90,
        contrast_factor_max=1.10,
        brightness_factor_min=0.95,
        brightness_factor_max=1.05,
        line_count=2,
        line_width_min=1,
        line_width_max=2,
        dot_count=60,
        occlusion_count=0,
        occlusion_size_fraction=0.00,
    ),
    2: DistortionLevel(
        level=2,
        name="moderate",
        description="Noticeable warping, blur, noise, line clutter, and light occlusion.",
        max_rotation_degrees=9.0,
        perspective_strength=0.060,
        shear_strength=0.08,
        blur_radius=1.10,
        gaussian_noise_std=15.0,
        contrast_factor_min=0.75,
        contrast_factor_max=1.25,
        brightness_factor_min=0.85,
        brightness_factor_max=1.15,
        line_count=5,
        line_width_min=2,
        line_width_max=4,
        dot_count=180,
        occlusion_count=1,
        occlusion_size_fraction=0.10,
    ),
    3: DistortionLevel(
        level=3,
        name="severe",
        description="Strong warping, heavy noise, low contrast, clutter, and occlusion.",
        max_rotation_degrees=16.0,
        perspective_strength=0.110,
        shear_strength=0.14,
        blur_radius=1.80,
        gaussian_noise_std=28.0,
        contrast_factor_min=0.55,
        contrast_factor_max=1.45,
        brightness_factor_min=0.70,
        brightness_factor_max=1.25,
        line_count=9,
        line_width_min=3,
        line_width_max=6,
        dot_count=360,
        occlusion_count=3,
        occlusion_size_fraction=0.18,
    ),
}


def get_distortion_level(level: int) -> DistortionLevel:
    """Return the configured distortion level or raise a helpful error."""
    try:
        return DISTORTION_LEVELS[level]
    except KeyError as exc:
        valid = ", ".join(str(k) for k in sorted(DISTORTION_LEVELS))
        raise ValueError(f"Unknown distortion level {level!r}. Valid levels: {valid}.") from exc
