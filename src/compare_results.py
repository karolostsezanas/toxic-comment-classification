"""Combine experiment summaries and plot full-dataset model comparisons."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

def main() -> None:
    """Merge summary CSV files and generate comparison charts by metric."""
    summary_files = list(RESULTS_DIR.glob("summary_metrics_*.csv"))

    print(f"Found {len(summary_files)} summary files.")

    if not summary_files:
        raise FileNotFoundError("No summary_metrics files found in outputs/results.")

    frames = []

    for file in summary_files:
        print(f"Reading: {file.name}")
        df = pd.read_csv(file)
        df["source_file"] = file.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "sample" in combined.columns:
        max_sample = combined["sample"].max()
        combined = combined[combined["sample"] == max_sample].copy()
        print(f"Using only full-dataset rows with sample = {max_sample}")

    combined["experiment"] = combined["feature"] + " + " + combined["model"]
    combined = combined.sort_values("macro_auc", ascending=False)

    output_csv = RESULTS_DIR / "combined_model_comparison_full_dataset.csv"
    combined.to_csv(output_csv, index=False)

    metrics = [
        "macro_auc",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "hamming_accuracy",
    ]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plot_data = combined.sort_values(metric)
        plt.barh(plot_data["experiment"], plot_data[metric])
        plt.xlabel(metric)
        plt.title(f"Full-dataset model comparison by {metric}")
        plt.tight_layout()

        output_figure = FIGURES_DIR / f"full_dataset_comparison_{metric}.png"
        plt.savefig(output_figure, dpi=200)
        plt.close()

        print(f"Saved figure: {output_figure.name}")

    print("Saved combined CSV:")
    print(output_csv)
    print()
    print(combined[[
        "feature",
        "model",
        "macro_auc",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "hamming_accuracy",
    ]])

if __name__ == "__main__":
    main()
