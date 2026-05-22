"""Distortion tools for CAPTCHA robustness experiments.

Typical usage:

    from src.distortions import DistortionPipeline
    distorted = DistortionPipeline(level=2, seed=42).apply(image)
"""

from .levels import DistortionLevel, DISTORTION_LEVELS
from .pipeline import DistortionPipeline

__all__ = ["DistortionLevel", "DISTORTION_LEVELS", "DistortionPipeline"]
