# -*- coding: utf-8 -*-
"""
Evaluate existing CAPTCHA model on clean and distorted synthetic datasets.

Purpose:
- Use the already-trained model as the machine evaluator.
- Measure performance by distortion level.
- Save results for later analysis and plotting.
"""

from pathlib import Path
import csv
import string
import torch

from Read_Captcha_Traind_Modell import predict


BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_LABELS = BASE_DIR / "data" / "synthetic" / "clean" / "labels.csv"
DISTORTED_METADATA = BASE_DIR / "data" / "synthetic" / "distorted" / "distortion_metadata.csv"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "distortion_evaluation.csv"


def character_accuracy(true_label, predicted_label):
    if predicted_label is None:
        return 0.0

    max_len = max(len(true_label), len(predicted_label))

    if max_len == 0:
        return 1.0

    correct = sum(1 for a, b in zip(true_label, predicted_label) if a == b)

    return correct / max_len


def sequence_accuracy(true_label, predicted_label):
    return int(true_label == predicted_label)


def evaluate_clean():
    rows = []

    with open(CLEAN_LABELS, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = BASE_DIR / "data" / "synthetic" / "clean" / row["filename"]
            true_label = row["label"]

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue

            pred = predict(str(image_path), use_beam=False, show_images=False)

            rows.append({
                "filename": row["filename"],
                "image_path": str(image_path),
                "true_label": true_label,
                "predicted_label": pred,
                "distortion_name": "clean",
                "distortion_level": 0,
                "sequence_accuracy": sequence_accuracy(true_label, pred),
                "character_accuracy": character_accuracy(true_label, pred),
            })

    return rows


def evaluate_distorted():
    rows = []

    with open(DISTORTED_METADATA, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] != "created" and row["status"] != "skipped_existing":
                continue

            image_path = Path(row["output_path"])
            true_label = row["label"]

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue
            
            pred = predict(str(image_path), use_beam=False, show_images=False)

            rows.append({
                "filename": image_path.name,
                "image_path": str(image_path),
                "true_label": true_label,
                "predicted_label": pred,
                "distortion_name": row["distortion_name"],
                "distortion_level": row["distortion_level"],
                "sequence_accuracy": sequence_accuracy(true_label, pred),
                "character_accuracy": character_accuracy(true_label, pred),
            })

    return rows


if __name__ == "__main__":

    all_rows = []

    print("Evaluating clean synthetic images...")
    all_rows.extend(evaluate_clean())

    print("Evaluating distorted synthetic images...")
    all_rows.extend(evaluate_distorted())

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "filename",
            "image_path",
            "true_label",
            "predicted_label",
            "distortion_name",
            "distortion_level",
            "sequence_accuracy",
            "character_accuracy",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved evaluation results to: {OUTPUT_CSV}")

