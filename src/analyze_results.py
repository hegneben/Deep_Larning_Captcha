# -*- coding: utf-8 -*-
"""
The purpose of this program is to assess results from the different synthetic
data sets that were used to test the model's abilities
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DISTORTION_RESULTS = BASE_DIR / "results" / "distortion_evaluation.csv"
OUTPUT_DIR = BASE_DIR / "results" / "summary_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    df = pd.read_csv(DISTORTION_RESULTS)

    df["distortion_level"] = pd.to_numeric(df["distortion_level"], errors="coerce")
    df["sequence_accuracy"] = pd.to_numeric(df["sequence_accuracy"], errors="coerce")
    df["character_accuracy"] = pd.to_numeric(df["character_accuracy"], errors="coerce")

    return df


def summarize_by_distortion(df):
    summary = (
        df.groupby(["distortion_name", "distortion_level"])
        .agg(
            n_samples=("filename", "count"),
            sequence_accuracy_mean=("sequence_accuracy", "mean"),
            character_accuracy_mean=("character_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(["distortion_name", "distortion_level"])
    )

    return summary


def summarize_by_distortion_only(df):
    summary = (
        df.groupby("distortion_name")
        .agg(
            n_samples=("filename", "count"),
            sequence_accuracy_mean=("sequence_accuracy", "mean"),
            character_accuracy_mean=("character_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("sequence_accuracy_mean")
    )

    return summary


def compute_accuracy_drop(df):
    clean_df = df[df["distortion_name"] == "clean"]

    if clean_df.empty:
        print("No clean baseline found. Skipping accuracy drop calculation.")
        return None

    clean_seq_acc = clean_df["sequence_accuracy"].mean()
    clean_char_acc = clean_df["character_accuracy"].mean()

    summary = summarize_by_distortion(df)

    summary["sequence_accuracy_drop_from_clean"] = (
        clean_seq_acc - summary["sequence_accuracy_mean"]
    )

    summary["character_accuracy_drop_from_clean"] = (
        clean_char_acc - summary["character_accuracy_mean"]
    )

    return summary


def save_table(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    df = load_results()

    print("Loaded results:")
    print(df.head())

    by_level = summarize_by_distortion(df)
    by_distortion = summarize_by_distortion_only(df)
    drop_summary = compute_accuracy_drop(df)

    save_table(by_level, "summary_by_distortion_level.csv")
    save_table(by_distortion, "summary_by_distortion.csv")

    if drop_summary is not None:
        save_table(drop_summary, "accuracy_drop_from_clean.csv")

    print("\nDone.")