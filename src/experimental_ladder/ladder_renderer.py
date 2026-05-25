# -*- coding: utf-8 -*-
"""
The purpose of this code is to generate CAPTCHA-like distortions in a controlled 
manner. Thse include changes to:
    - occlusion lines
    - noise addition
    - spacing variation
    - positional jitter
    - character rotation
    - font variation
    - mixed, mild CAPTCHA-like distortions
    - full CAPTCHA-like distortion
"""

import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .ladder_config import (
    CHARSET,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    FONT_SIZE,
    SEQUENCE_LENGTH,
    FONTS_DIR,
)


def generate_label(length=SEQUENCE_LENGTH):
    return "".join(random.choice(CHARSET) for _ in range(length))


def get_font_files():
    font_files = list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf"))

    if not font_files:
        raise FileNotFoundError(f"No fonts found in {FONTS_DIR}")

    return font_files


def select_font(level):
    font_files = get_font_files()

    if level >= 3:
        font_path = random.choice(font_files)
    else:
        font_path = font_files[0]

    return font_path, ImageFont.truetype(str(font_path), FONT_SIZE)


def draw_centered_text(label, font):
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (IMAGE_WIDTH - text_width) // 2
    y = (IMAGE_HEIGHT - text_height) // 2 - 2

    draw.text((x, y), label, font=font, fill="black")

    return image


def draw_per_character(label, font, level):
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")

    x = random.randint(5, 12)

    for char in label:
        char_img = Image.new("RGBA", (40, IMAGE_HEIGHT), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)

        y = 5

        if level >= 2:
            y += random.randint(-5, 5)

        char_draw.text((5, y), char, font=font, fill=(0, 0, 0, 255))

        if level >= 4:
            angle = random.uniform(-12, 12)
            char_img = char_img.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255, 0),
            )

        image.paste(char_img, (x, 0), char_img)

        if level >= 1:
            x += random.randint(20, 28)
        else:
            x += 24

    return image


def add_noise(image, std=18):
    arr = np.array(image).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_occlusion_lines(image, n_lines=2):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(n_lines):
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)
        draw.line(
            (0, y1, width, y2),
            fill="black",
            width=random.randint(1, 2),
        )

    return image


def add_mild_captcha_effects(image, level):
    if level >= 5:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
        image = add_noise(image, std=10)
        image = add_occlusion_lines(image, n_lines=1)

    if level >= 6:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.7))
        image = add_noise(image, std=18)
        image = add_occlusion_lines(image, n_lines=3)

    return image


def render_ladder_sample(level):
    label = generate_label()
    font_path, font = select_font(level)

    if level == 0:
        image = draw_centered_text(label, font)
    else:
        image = draw_per_character(label, font, level)

    image = add_mild_captcha_effects(image, level)

    return image, label, font_path.name