# -*- coding: utf-8 -*-
"""
Created on Mon May 25 04:04:38 2026

@author: Luka Ilisevic
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

SUMMARY_DIR = BASE_DIR / "results" / "summary_tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BY_LEVEL_CSV = SUMMARY_DIR / "summary_by_distortion_level.csv"
BY_DISTORTION_CSV = SUMMARY_DIR / "summary_by_distortion.csv"
DROP_CSV = SUMMARY_DIR / "accuracy_drop_from_clean.csv"


def plot_accuracy_by_distortion_level():
    df = pd.read_csv(BY_LEVEL_CSV)

    plt.figure(figsize=(10, 6))

    for distortion_name, group in df.groupby("distortion_name"):
        group = group.sort_values("distortion_level")
        plt.plot(
            group["distortion_level"],
            group["sequence_accuracy_mean"],
            marker="o",
            label=distortion_name,
        )

    plt.xlabel("Distortion Level")
    plt.ylabel("Mean Sequence Accuracy")
    plt.title("Sequence Accuracy by Distortion Severity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURE_DIR / "sequence_accuracy_by_distortion_level.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_character_accuracy_by_distortion_level():
    df = pd.read_csv(BY_LEVEL_CSV)

    plt.figure(figsize=(10, 6))

    for distortion_name, group in df.groupby("distortion_name"):
        group = group.sort_values("distortion_level")
        plt.plot(
            group["distortion_level"],
            group["character_accuracy_mean"],
            marker="o",
            label=distortion_name,
        )

    plt.xlabel("Distortion Level")
    plt.ylabel("Mean Character Accuracy")
    plt.title("Character Accuracy by Distortion Severity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURE_DIR / "character_accuracy_by_distortion_level.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_average_accuracy_by_distortion():
    df = pd.read_csv(BY_DISTORTION_CSV)

    df = df.sort_values("sequence_accuracy_mean")

    plt.figure(figsize=(10, 6))
    plt.bar(
        df["distortion_name"],
        df["sequence_accuracy_mean"],
    )

    plt.xlabel("Distortion Type")
    plt.ylabel("Mean Sequence Accuracy")
    plt.title("Average Sequence Accuracy by Distortion Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURE_DIR / "average_sequence_accuracy_by_distortion.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_accuracy_drop_from_clean():
    df = pd.read_csv(DROP_CSV)

    df = df.sort_values("sequence_accuracy_drop_from_clean", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(
        df["distortion_name"] + "_L" + df["distortion_level"].astype(str),
        df["sequence_accuracy_drop_from_clean"],
    )

    plt.xlabel("Distortion Condition")
    plt.ylabel("Sequence Accuracy Drop from Clean")
    plt.title("Accuracy Drop Relative to Clean Synthetic Baseline")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    output_path = FIGURE_DIR / "sequence_accuracy_drop_from_clean.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_accuracy_by_distortion_level()
    plot_character_accuracy_by_distortion_level()
    plot_average_accuracy_by_distortion()
    plot_accuracy_drop_from_clean()

    print("\nAll plots saved.")