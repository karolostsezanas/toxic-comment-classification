"""Train a final model and write a Kaggle-style submission file."""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import expit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from config import DATA_DIR, SUBMISSIONS_DIR, LABELS, RANDOM_STATE
from device import get_device
from text_utils import basic_clean

def load_files():
    """Load the train, test, and sample submission CSV files."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_path = DATA_DIR / "sample_submission.csv"

    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    return train_df, test_df, sample

def get_model(model_name: str):
    """Build the requested final submission classifier."""
    if model_name == "logreg":
        return OneVsRestClassifier(
            LogisticRegression(
                C=2.0,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        )

    if model_name == "svm":
        return OneVsRestClassifier(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                dual=False,
                random_state=RANDOM_STATE,
            )
        )

    raise ValueError("Supported final submission models: logreg, svm")

def get_scores(model, X_test) -> np.ndarray:
    """Return calibrated prediction scores from the fitted model."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
        if isinstance(scores, list):
            return np.vstack([arr[:, 1] for arr in scores]).T
        return np.asarray(scores)

    if hasattr(model, "decision_function"):
        return expit(np.asarray(model.decision_function(X_test)))

    raise RuntimeError("Model does not expose predict_proba or decision_function.")

def build_tfidf_full(X_train, X_test):
    """Create combined word- and character-level TF-IDF features."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=100000,
            min_df=3,
            max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("char", TfidfVectorizer(
            max_features=100000,
            min_df=3,
            ngram_range=(3, 5),
            sublinear_tf=True,
            analyzer="char",
        )),
    ])

    X_train_features = features.fit_transform(X_train)
    X_test_features = features.transform(X_test)
    return X_train_features, X_test_features

def build_tfidf_svd(X_train, X_test, n_components):
    """Create dense TF-IDF features reduced with truncated SVD."""
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
    X_test_sparse = vectorizer.transform(X_test)
    X_train_features = reducer.fit_transform(X_train_sparse)
    X_test_features = reducer.transform(X_test_sparse)
    return X_train_features, X_test_features

def build_sbert(X_train, X_test, device, model_name, batch_size):
    """Encode text with a sentence-transformer model on the selected device."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    X_train_features = model.encode(
        X_train.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    X_test_features = model.encode(
        X_test.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return X_train_features, X_test_features

def main():
    """Train the selected model pipeline and save a submission CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=["tfidf_full", "tfidf_svd", "sbert"], default="tfidf_full")
    parser.add_argument("--model", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--svd-components", type=int, default=300)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    selected_device = get_device(args.device)
    print(f"Selected device: {selected_device}")

    train_df, test_df, sample = load_files()

    if args.sample and args.sample > 0 and args.sample < len(train_df):
        train_df = train_df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    X_train = train_df["comment_text"].fillna("").map(basic_clean)
    X_test = test_df["comment_text"].fillna("").map(basic_clean)
    y_train = train_df[LABELS].astype(int).values

    if args.feature == "tfidf_full":
        X_train_features, X_test_features = build_tfidf_full(X_train, X_test)
    elif args.feature == "tfidf_svd":
        X_train_features, X_test_features = build_tfidf_svd(X_train, X_test, args.svd_components)
    else:
        X_train_features, X_test_features = build_sbert(
            X_train,
            X_test,
            selected_device,
            args.sbert_model,
            args.batch_size,
        )

    model = get_model(args.model)
    print(f"Training final model: {args.feature} + {args.model}")
    model.fit(X_train_features, y_train)

    scores = get_scores(model, X_test_features)
    scores = np.clip(scores, 0.0, 1.0)

    submission = sample.copy()
    for i, label in enumerate(LABELS):
        submission[label] = scores[:, i]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SUBMISSIONS_DIR / f"submission_{args.feature}_{args.model}_{timestamp}.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")

if __name__ == "__main__":
    main()
