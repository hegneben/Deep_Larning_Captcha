"""Reusable CAPTCHA distortion pipeline.

This module intentionally depends only on Pillow and NumPy, which are already common
in the project. It accepts a PIL image and returns a distorted PIL image.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .levels import DistortionLevel, get_distortion_level


RGB = Tuple[int, int, int]


class DistortionPipeline:
    """Apply controlled CAPTCHA distortions for robustness experiments.

    Parameters
    ----------
    level:
        Distortion level: 1, 2, or 3.
    seed:
        Optional random seed. Use this for reproducible experiments.
    background_color:
        Color used to fill newly exposed image corners after geometric transforms.
    """

    def __init__(
        self,
        level: int,
        seed: int | None = None,
        background_color: RGB = (255, 255, 255),
    ) -> None:
        self.config: DistortionLevel = get_distortion_level(level)
        self.seed = seed
        self.background_color = background_color
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def apply(self, image: Image.Image) -> Image.Image:
        """Return a distorted copy of ``image``."""
        img = image.convert("RGB")

        # Keep order deterministic and research-friendly.
        img = self._adjust_brightness_and_contrast(img)
        img = self._rotate(img)
        img = self._shear(img)
        img = self._perspective_warp(img)
        img = self._draw_interference_lines(img)
        img = self._draw_noise_dots(img)
        img = self._draw_occlusions(img)
        img = self._add_gaussian_noise(img)
        img = self._blur(img)

        return img

    def apply_to_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        """Read one image, distort it, and save it."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(input_path) as image:
            distorted = self.apply(image)
            distorted.save(output_path)

        return output_path

    def _adjust_brightness_and_contrast(self, image: Image.Image) -> Image.Image:
        cfg = self.config
        contrast = self._rng.uniform(cfg.contrast_factor_min, cfg.contrast_factor_max)
        brightness = self._rng.uniform(cfg.brightness_factor_min, cfg.brightness_factor_max)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        return image

    def _rotate(self, image: Image.Image) -> Image.Image:
        degrees = self._rng.uniform(-self.config.max_rotation_degrees, self.config.max_rotation_degrees)
        return image.rotate(
            degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=self.background_color,
        )

    def _shear(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        shear = self._rng.uniform(-self.config.shear_strength, self.config.shear_strength)
        return image.transform(
            (width, height),
            Image.Transform.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=self.background_color,
        )

    def _perspective_warp(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        s = self.config.perspective_strength
        dx = int(width * s)
        dy = int(height * s)

        # Distorted destination corners.
        dst = [
            (self._rng.randint(0, dx), self._rng.randint(0, dy)),
            (width - self._rng.randint(0, dx), self._rng.randint(0, dy)),
            (width - self._rng.randint(0, dx), height - self._rng.randint(0, dy)),
            (self._rng.randint(0, dx), height - self._rng.randint(0, dy)),
        ]
        src = [(0, 0), (width, 0), (width, height), (0, height)]
        coeffs = _find_perspective_coefficients(dst, src)

        return image.transform(
            (width, height),
            Image.Transform.PERSPECTIVE,
            coeffs,
            resample=Image.Resampling.BICUBIC,
            fillcolor=self.background_color,
        )

    def _draw_interference_lines(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for _ in range(self.config.line_count):
            points = self._random_curve_points(width, height)
            color = self._random_darkish_color()
            line_width = self._rng.randint(self.config.line_width_min, self.config.line_width_max)
            draw.line(points, fill=color, width=line_width, joint="curve")

        return image

    def _draw_noise_dots(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for _ in range(self.config.dot_count):
            x = self._rng.randint(0, width - 1)
            y = self._rng.randint(0, height - 1)
            radius = 1 if self.config.level < 3 else self._rng.randint(1, 2)
            color = self._random_darkish_color()
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        return image

    def _draw_occlusions(self, image: Image.Image) -> Image.Image:
        if self.config.occlusion_count <= 0:
            return image

        draw = ImageDraw.Draw(image)
        width, height = image.size
        max_w = max(1, int(width * self.config.occlusion_size_fraction))
        max_h = max(1, int(height * self.config.occlusion_size_fraction))

        for _ in range(self.config.occlusion_count):
            x1 = self._rng.randint(0, max(0, width - max_w))
            y1 = self._rng.randint(0, max(0, height - max_h))
            x2 = x1 + self._rng.randint(max(3, max_w // 3), max_w)
            y2 = y1 + self._rng.randint(max(3, max_h // 3), max_h)
            fill = self._random_lightish_color()
            draw.rectangle((x1, y1, min(x2, width), min(y2, height)), fill=fill)

        return image

    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        std = self.config.gaussian_noise_std
        if std <= 0:
            return image

        arr = np.asarray(image).astype(np.float32)
        noise = self._np_rng.normal(loc=0.0, scale=std, size=arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    def _blur(self, image: Image.Image) -> Image.Image:
        radius = self.config.blur_radius
        if radius <= 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _random_curve_points(self, width: int, height: int) -> list[tuple[int, int]]:
        point_count = 4 if self.config.level == 1 else 6
        return [
            (int(i * width / (point_count - 1)), self._rng.randint(0, height - 1))
            for i in range(point_count)
        ]

    def _random_darkish_color(self) -> RGB:
        return (
            self._rng.randint(0, 150),
            self._rng.randint(0, 150),
            self._rng.randint(0, 150),
        )

    def _random_lightish_color(self) -> RGB:
        base = self._rng.randint(170, 255)
        return (
            min(255, base + self._rng.randint(-20, 20)),
            min(255, base + self._rng.randint(-20, 20)),
            min(255, base + self._rng.randint(-20, 20)),
        )


def _find_perspective_coefficients(
    dst_points: Iterable[tuple[int, int]],
    src_points: Iterable[tuple[int, int]],
) -> list[float]:
    """Return Pillow perspective transform coefficients.

    Pillow expects coefficients that map output coordinates back to input
    coordinates, hence destination points are provided first.
    """
    matrix = []
    vector = []

    for (x, y), (u, v) in zip(dst_points, src_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])

    coeffs = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
    return coeffs.tolist()
