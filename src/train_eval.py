"""Train, evaluate, and compare toxic-comment classification experiments."""

import argparse
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, LABELS, RANDOM_STATE
from device import get_device
from text_utils import basic_clean

def load_data(sample: int) -> tuple[pd.Series, pd.DataFrame]:
    """Load and optionally subsample the training data and labels."""
    path = DATA_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError("Place train.csv inside the data folder.")

    df = pd.read_csv(path)
    if sample and sample > 0 and sample < len(df):
        df = df.sample(sample, random_state=RANDOM_STATE).reset_index(drop=True)

    texts = df["comment_text"].fillna("").map(basic_clean)
    y = df[LABELS].astype(int)
    return texts, y

def build_tfidf_svd(X_train: pd.Series, X_valid: pd.Series, n_components: int) -> tuple[np.ndarray, np.ndarray, object]:
    """Build dense TF-IDF plus SVD features for train and validation sets."""
    vectorizer = TfidfVectorizer(
        max_features=60000,
        min_df=3,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    reducer = make_pipeline(
        TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE),
        Normalizer(copy=False),
    )

    X_train_sparse = vectorizer.fit_transform(X_train)
    X_valid_sparse = vectorizer.transform(X_valid)

    X_train_dense = reducer.fit_transform(X_train_sparse)
    X_valid_dense = reducer.transform(X_valid_sparse)

    feature_object = {
        "vectorizer": vectorizer,
        "reducer": reducer,
    }
    return X_train_dense, X_valid_dense, feature_object

def build_sbert(X_train: pd.Series, X_valid: pd.Series, device: str, model_name: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, object]:
    """Encode train and validation comments with a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    X_train_dense = model.encode(
        X_train.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    X_valid_dense = model.encode(
        X_valid.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return X_train_dense, X_valid_dense, model_name

def build_models(selected_models: list[str]) -> dict[str, object]:
    """Instantiate the subset of supported classifiers requested by the user."""
    all_models = {
        "logreg": OneVsRestClassifier(
            LogisticRegression(
                C=2.0,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "svm": OneVsRestClassifier(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                dual=False,
                random_state=RANDOM_STATE,
            )
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    return {name: all_models[name] for name in selected_models}

def get_score_matrix(model, X) -> np.ndarray:
    """Extract class-wise scores from the model for ranking-based metrics."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if isinstance(scores, list):
            scores = np.vstack([arr[:, 1] for arr in scores]).T
        return np.asarray(scores)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        return scores

    preds = model.predict(X)
    return np.asarray(preds)

def evaluate_model(model_name: str, feature_name: str, model, X_valid, y_valid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute per-label and aggregate validation metrics for a trained model."""
    y_true = y_valid.values
    y_pred = np.asarray(model.predict(X_valid))
    y_scores = get_score_matrix(model, X_valid)

    rows = []
    for i, label in enumerate(LABELS):
        try:
            auc_value = roc_auc_score(y_true[:, i], y_scores[:, i])
        except ValueError:
            auc_value = np.nan

        rows.append({
            "feature": feature_name,
            "model": model_name,
            "label": label,
            "accuracy": accuracy_score(y_true[:, i], y_pred[:, i]),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "auc": auc_value,
        })

    per_label = pd.DataFrame(rows)
    summary = {
        "feature": feature_name,
        "model": model_name,
        "macro_accuracy": per_label["accuracy"].mean(),
        "macro_precision": per_label["precision"].mean(),
        "macro_recall": per_label["recall"].mean(),
        "macro_f1": per_label["f1"].mean(),
        "macro_auc": per_label["auc"].mean(),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "hamming_accuracy": 1.0 - hamming_loss(y_true, y_pred),
    }
    return per_label, summary

def plot_roc_curves(model_name: str, feature_name: str, y_valid: pd.DataFrame, y_scores: np.ndarray) -> None:
    """Plot one-vs-rest ROC curves for all toxicity labels."""
    y_true = y_valid.values
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(LABELS):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            label_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} AUC={label_auc:.3f}")
        except ValueError:
            continue

    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.title(f"ROC curves: {feature_name} + {model_name}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"roc_{feature_name}_{model_name}.png", dpi=200)
    plt.close()

def plot_summary(summary_df: pd.DataFrame, feature_name: str) -> None:
    """Create summary bar charts for the main model comparison metrics."""
    metrics = ["macro_auc", "macro_f1", "macro_precision", "macro_recall", "hamming_accuracy"]
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        summary_df.set_index("model")[metric].sort_values().plot(kind="barh")
        plt.title(f"{metric} by model, feature={feature_name}")
        plt.xlabel(metric)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"summary_{feature_name}_{metric}.png", dpi=200)
        plt.close()

def main() -> None:
    """Run one experiment configuration and persist metrics, plots, and models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=["tfidf_svd", "sbert"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--svd-components", type=int, default=300)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", nargs="+", default=["logreg", "svm", "rf"], choices=["logreg", "svm", "rf"])
    args = parser.parse_args()

    selected_device = get_device(args.device)
    print(f"Selected device: {selected_device}")

    texts, y = load_data(args.sample)
    X_train, X_valid, y_train, y_valid = train_test_split(
        texts,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y["toxic"],
    )

    if args.feature == "tfidf_svd":
        X_train_features, X_valid_features, feature_object = build_tfidf_svd(
            X_train,
            X_valid,
            n_components=args.svd_components,
        )
    else:
        X_train_features, X_valid_features, feature_object = build_sbert(
            X_train,
            X_valid,
            device=selected_device,
            model_name=args.sbert_model,
            batch_size=args.batch_size,
        )

    models = build_models(args.models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_per_label = []
    all_summary = []

    for model_name, model in models.items():
        print(f"Training {model_name} with {args.feature}")
        start = time.time()
        model.fit(X_train_features, y_train.values)
        train_seconds = time.time() - start

        per_label, summary = evaluate_model(model_name, args.feature, model, X_valid_features, y_valid)
        summary["train_seconds"] = round(train_seconds, 3)
        summary["device"] = selected_device
        summary["sample"] = args.sample if args.sample else len(texts)

        y_scores = get_score_matrix(model, X_valid_features)
        plot_roc_curves(model_name, args.feature, y_valid, y_scores)

        all_per_label.append(per_label)
        all_summary.append(summary)

        model_path = RESULTS_DIR / f"model_{args.feature}_{model_name}_{timestamp}.joblib"
        joblib.dump(model, model_path)

    per_label_df = pd.concat(all_per_label, ignore_index=True)
    summary_df = pd.DataFrame(all_summary).sort_values("macro_auc", ascending=False)

    per_label_path = RESULTS_DIR / f"per_label_metrics_{args.feature}_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"summary_metrics_{args.feature}_{timestamp}.csv"
    config_path = RESULTS_DIR / f"run_config_{args.feature}_{timestamp}.json"

    per_label_df.to_csv(per_label_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df, args.feature)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Experiment complete.")
    print(summary_df.to_string(index=False))
    print(f"Saved per-label metrics to: {per_label_path}")
    print(f"Saved summary metrics to: {summary_path}")

if __name__ == "__main__":
    main()
