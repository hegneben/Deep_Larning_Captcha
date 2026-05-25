# -*- coding: utf-8 -*-
"""
The goal of this program is to define the basic configuration of the 
CAPTCHA alterations that we implementated
"""

import string
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

CHARSET = string.ascii_lowercase + string.ascii_uppercase + string.digits

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 48
FONT_SIZE = 30
SEQUENCE_LENGTH = 5

FONTS_DIR = BASE_DIR / "fonts"
OUTPUT_DIR = BASE_DIR / "data" / "synthetic" / "experimental_ladder"

NUM_SAMPLES_PER_LEVEL = 1000

LADDER_LEVELS = {
    0: "clean_centered_text",
    1: "random_spacing",
    2: "vertical_jitter",
    3: "font_variation",
    4: "character_rotation",
    5: "mild_captcha_noise",
    6: "full_captcha_like",
}