"""
The purpose of this program is to build a synthetic CAPTCHA-like images
- First without distortions to establish a reliable baseline performance for the deep learning model
- Second with image distortions/corruption, in order to determine the limits of the learning model's performance

Proposed Distortion Regime:
---------------------------------------------------------------------------------------
Corruption                     Parameter	  Severity 1	  Severity 3	Severity 5
----------------------------------------------------------------------------------------
Gaussian Blur	               sigma	      0.5             2.0	        4.0
Rotation	                   degrees	      5°	          15°	        30°
Gaussian Noise	               std dev	      0.02	          0.08	        0.16
Occlusion Lines	               line count	  1	              3	            6
Warp Distortion                displacement	  low	          medium	    high



"""

import os
import random
import string

from PIL import Image, ImageDraw, ImageFont

# Generalize so path is folder structure agnostic
# _____________________________________________________________________________
from pathlib import Path

base_directory = Path(__file__).resolve().parent.parent

output_directory = base_directory / "data" / "synthetic" / "clean"

os.makedirs(output_directory, exist_ok=True)


# Establish synthetic CAPTCHA-like data set using the following configuration:
# _____________________________________________________________________________
image_width = 200
image_height = 48

font_size = 30

charset = string.ascii_letters + string.digits

sequence_length = 5

font_directory = base_directory / "fonts"

font_path = font_directory / "arial.ttf"

print("Looking for font at:", font_path)
print("Exists?", font_path.exists())

font = ImageFont.truetype(str(font_path), font_size)

# Generate a random label
# _____________________________________________________________________________

def generate_label(length=sequence_length):
    return "".join(random.choice(charset) for _ in range(length))

# Render alphanumeric text as image
# _____________________________________________________________________________

def render_text_image(label):
    
    # start with white background
    image = Image.new("RGB", (image_width, image_height), color = "white")
    
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.truetype(str(font_path), font_size)
    
    # Define text starting position on the background
    x = 10
    y = 5
    
    # Draw characters with random spacing
    #for char in label:
        
    #    draw.text((x, y), char, fill = "black", font = font)
        
    #    x += 20
        
    #    return image
    
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_width - text_width) // 2
    y = (image_height - text_height) // 2 - 2

    draw.text((x, y), label, fill="black", font=font)
    
    return image
    
# Main
# _____________________________________________________________________________

import csv

num_samples = 1000

if __name__ == "__main__":

    images_dir = output_directory / "images"
    os.makedirs(images_dir, exist_ok=True)

    metadata_path = output_directory / "labels.csv"

    with open(metadata_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "filename",
            "label",
            "sequence_length",
            "font",
            "distortion_name",
            "distortion_level"
        ])

        for i in range(num_samples):
            label = generate_label()
            image = render_text_image(label)

            filename = f"{i:06d}_{label}.png"
            save_path = images_dir / filename

            image.save(save_path)

            writer.writerow([
                filename,
                label,
                len(label),
                font_path.name,
                "clean",
                0
            ])

    print(f"Generated {num_samples} clean samples.")
    print(f"Images saved to: {images_dir}")
    print(f"Metadata saved to: {metadata_path}")