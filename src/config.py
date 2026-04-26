"""Project-wide paths, labels, and shared configuration constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

RANDOM_STATE = 42

for folder in [FIGURES_DIR, RESULTS_DIR, SUBMISSIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
