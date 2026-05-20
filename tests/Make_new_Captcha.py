"""
CAPTCHA Image Generator
=======================
All settings are at the top of this file under "SETTINGS".
Do not edit anything below the "DO NOT EDIT BELOW" line.

Run with:  python captcha_generator.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# =============================================================================
# SETTINGS — Edit everything here
# =============================================================================

# Output
OUTPUT_PATH = "captcha.png"          # File name / path of the saved image

# Canvas
CANVAS_WIDTH  = 400                  # Image width in pixels
CANVAS_HEIGHT = 150                  # Image height in pixels
BACKGROUND_COLOR = (180, 190, 200)   # Background color (R, G, B)

# Text
CAPTCHA_TEXT  = "finish"             # The word shown in the CAPTCHA
TEXT_COLOR    = (0, 100, 0)          # Text color (R, G, B)  →  dark green
TEXT_POSITION = (100, 25)            # Top-left corner of the text (x, y)
FONT_PATH     = "arial.ttf"          # .ttf font file; falls back to default if not found
                                     #   Windows: "arial.ttf"
                                     #   Linux:   "DejaVuSans.ttf"
                                     #   macOS:   "Helvetica.ttc"
FONT_SIZE     = 80                   # Font size in points

# Distortion
DISTORTION_ENABLED = True            # Set to False to skip distortion
DISTORTION_STRENGTH  = 2             # Blur radius: 0 = off, 1 = subtle, 5 = strong
DISTORTION_FILTER    = ImageFilter.GaussianBlur(DISTORTION_STRENGTH)

# DISTORTION_FILTER  = ImageFilter.EDGE_ENHANCE_MORE  # Options:
                                     #   ImageFilter.EDGE_ENHANCE_MORE  (sharp edges)
                                     #   ImageFilter.BLUR               (soft blur)
                                     #   ImageFilter.SMOOTH             (gentle smoothing)
                                     #   ImageFilter.GaussianBlur(2)    (blur with radius)

# Noise lines — each entry is (points, color, width)
# points: list of (x, y) coordinates connected in order  →  more points = more curves
# color:  line color as (R, G, B)
# width:  line thickness in pixels
NOISE_LINES = [
    ([(0, 110), (100, 100), (200, 120), (300, 100), (400, 110)], (0,  80,   0), 5),  # green wave
    ([(0,  60), (120,  50), (220,  40), (320,  30), (400,  25)], (150, 50,  50), 4),  # red diagonal
    ([(40,  30), (60,   80)],                                     (70,  50, 120), 5),  # purple short
]

# =============================================================================
# DO NOT EDIT BELOW — Generator logic
# =============================================================================

@dataclass
class _NoiseLine:
    points: list[tuple[int, int]]
    color:  tuple[int, int, int]
    width:  int


@dataclass
class _Config:
    output_path:        str
    canvas_size:        tuple[int, int]
    background_color:   tuple[int, int, int]
    text:               str
    text_color:         tuple[int, int, int]
    text_position:      tuple[int, int]
    font_path:          str
    font_size:          int
    distortion_enabled: bool
    distortion_filter:  ImageFilter.Filter
    noise_lines:        list[_NoiseLine]


class CaptchaGenerator:
    def __init__(self, config: _Config) -> None:
        self._cfg = config

    def generate(self) -> Path:
        image, draw = self._create_canvas()
        font = self._load_font()
        self._draw_text(draw, font)
        self._draw_noise_lines(draw)
        image = self._apply_distortion(image)
        return self._save(image)

    def _create_canvas(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", self._cfg.canvas_size, color=self._cfg.background_color)
        return image, ImageDraw.Draw(image)

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(self._cfg.font_path, self._cfg.font_size)
        except IOError:
            print(f"Warning: font '{self._cfg.font_path}' not found — using built-in fallback.")
            return ImageFont.load_default()

    def _draw_text(self, draw: ImageDraw.ImageDraw, font) -> None:
        draw.text(self._cfg.text_position, self._cfg.text, fill=self._cfg.text_color, font=font)

    def _draw_noise_lines(self, draw: ImageDraw.ImageDraw) -> None:
        for line in self._cfg.noise_lines:
            draw.line(line.points, fill=line.color, width=line.width)

    def _apply_distortion(self, image: Image.Image) -> Image.Image:
        if self._cfg.distortion_enabled:
            return image.filter(self._cfg.distortion_filter)
        return image

    def _save(self, image: Image.Image) -> Path:
        output = Path(self._cfg.output_path)
        image.save(output)
        print(f"CAPTCHA saved to '{output}'.")
        return output


def main() -> None:
    config = _Config(
        output_path        = OUTPUT_PATH,
        canvas_size        = (CANVAS_WIDTH, CANVAS_HEIGHT),
        background_color   = BACKGROUND_COLOR,
        text               = CAPTCHA_TEXT,
        text_color         = TEXT_COLOR,
        text_position      = TEXT_POSITION,
        font_path          = FONT_PATH,
        font_size          = FONT_SIZE,
        distortion_enabled = DISTORTION_ENABLED,
        distortion_filter  = DISTORTION_FILTER,
        noise_lines        = [_NoiseLine(p, c, w) for p, c, w in NOISE_LINES],
    )
    CaptchaGenerator(config).generate()


if __name__ == "__main__":
    main()
