"""Exploratory data analysis utilities for the toxic comments dataset."""

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

from config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, LABELS
from text_utils import tokenize, top_words

def load_train() -> pd.DataFrame:
    """Load the training dataset from the project data directory."""
    path = DATA_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError("Place train.csv inside the data folder.")
    return pd.read_csv(path)

def save_class_balance(df: pd.DataFrame) -> None:
    """Save class-balance tables and a bar chart for positive labels."""
    counts = df[LABELS].sum().sort_values(ascending=False)
    total = len(df)

    table = pd.DataFrame({
        "positive_comments": counts,
        "positive_percent": (counts / total * 100).round(3),
    })
    table.loc["clean"] = [
        int((df[LABELS].sum(axis=1) == 0).sum()),
        round((df[LABELS].sum(axis=1) == 0).mean() * 100, 3),
    ]
    table.to_csv(RESULTS_DIR / "class_balance.csv")

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title("Positive examples per toxicity label")
    plt.ylabel("Number of comments")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=200)
    plt.close()

def save_multilabel_distribution(df: pd.DataFrame) -> None:
    """Save the distribution of label counts assigned to each comment."""
    label_count = df[LABELS].sum(axis=1)
    distribution = label_count.value_counts().sort_index()
    distribution.to_csv(RESULTS_DIR / "labels_per_comment.csv", header=["comments"])

    plt.figure(figsize=(8, 5))
    distribution.plot(kind="bar")
    plt.title("Number of toxicity labels per comment")
    plt.xlabel("Labels assigned to one comment")
    plt.ylabel("Number of comments")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "labels_per_comment.png", dpi=200)
    plt.close()

def save_token_length_analysis(df: pd.DataFrame) -> None:
    """Summarize and plot token-count statistics for each label group."""
    df = df.copy()
    df["token_count"] = df["comment_text"].fillna("").map(lambda x: len(tokenize(x)))

    rows = []
    rows.append({
        "group": "all_comments",
        "count": len(df),
        "mean_tokens": df["token_count"].mean(),
        "median_tokens": df["token_count"].median(),
        "p95_tokens": df["token_count"].quantile(0.95),
    })

    clean_mask = df[LABELS].sum(axis=1) == 0
    rows.append({
        "group": "clean",
        "count": int(clean_mask.sum()),
        "mean_tokens": df.loc[clean_mask, "token_count"].mean(),
        "median_tokens": df.loc[clean_mask, "token_count"].median(),
        "p95_tokens": df.loc[clean_mask, "token_count"].quantile(0.95),
    })

    for label in LABELS:
        mask = df[label] == 1
        rows.append({
            "group": label,
            "count": int(mask.sum()),
            "mean_tokens": df.loc[mask, "token_count"].mean(),
            "median_tokens": df.loc[mask, "token_count"].median(),
            "p95_tokens": df.loc[mask, "token_count"].quantile(0.95),
        })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "token_length_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    clipped = df["token_count"].clip(upper=df["token_count"].quantile(0.99))
    plt.hist(clipped, bins=60)
    plt.title("Comment token length distribution, clipped at 99th percentile")
    plt.xlabel("Token count")
    plt.ylabel("Number of comments")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "token_length_distribution.png", dpi=200)
    plt.close()

def save_label_correlation(df: pd.DataFrame) -> None:
    """Compute pairwise label correlations and save a heatmap."""
    corr = df[LABELS].corr()
    corr.to_csv(RESULTS_DIR / "label_correlation.csv")

    plt.figure(figsize=(8, 7))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation between toxicity labels")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_correlation_heatmap.png", dpi=200)
    plt.close()

def save_top_words(df: pd.DataFrame) -> None:
    """Extract and plot the most common informative words per label."""
    all_rows = []
    for label in LABELS:
        texts = df.loc[df[label] == 1, "comment_text"].fillna("")
        words = top_words(texts, n=20)
        label_table = pd.DataFrame(words, columns=["word", "count"])
        label_table["label"] = label
        all_rows.append(label_table)

        plt.figure(figsize=(9, 5))
        label_table.set_index("word")["count"].sort_values().plot(kind="barh")
        plt.title(f"Most common useful words in {label} comments")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"top_words_{label}.png", dpi=200)
        plt.close()

    pd.concat(all_rows, ignore_index=True).to_csv(RESULTS_DIR / "top_words_by_label.csv", index=False)

def main() -> None:
    """Run the full EDA pipeline and write outputs to disk."""
    df = load_train()
    save_class_balance(df)
    save_multilabel_distribution(df)
    save_token_length_analysis(df)
    save_label_correlation(df)
    save_top_words(df)
    print("EDA complete. Check outputs/figures and outputs/results.")

if __name__ == "__main__":
    main()
