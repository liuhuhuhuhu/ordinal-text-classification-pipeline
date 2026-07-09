"""
RA Coding Test — Ordinal Text Classification Pipeline

This script runs the full pipeline end-to-end:
1. Clean raw training data.
2. Train and evaluate a TF-IDF + Logistic Regression baseline.
3. Train and evaluate a continuous-score MPNet + Ridge model with threshold calibration.
4. Generate predictions_baseline.csv and predictions_best.csv for data/eval_posts.csv.

Expected project structure:

project_root/
├── data/
│   ├── posts_raw.csv
│   └── eval_posts.csv
├── src/
│   └── pipeline.py
├── requirements.txt
├── run.sh
└── REPORT.md

Run from project root:

    python src/pipeline.py

or:

    bash run.sh
"""

from __future__ import annotations

import html
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# ==========================================================
# Global configuration
# ==========================================================

SEED = 42
LABELS = [0, 1, 2, 3, 4]

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "posts_raw.csv"
EVAL_PATH = DATA_DIR / "eval_posts.csv"

PRED_BASELINE_PATH = Path("predictions_baseline.csv")
PRED_BEST_PATH = Path("predictions_best.csv")

random.seed(SEED)
np.random.seed(SEED)


# ==========================================================
# Text and label cleaning utilities
# ==========================================================

def canonicalize_label(x) -> float:
    """
    Convert heterogeneous raw labels to the canonical ordinal scale 0-4.

    Returns np.nan for missing or unrecognized labels.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in {"", "?", "nan", "none", "missing"}:
        return np.nan

    # Numeric labels such as 0, 1, 2, 3, 4, 3.0, 4.0
    try:
        value = float(s)
        if value.is_integer() and int(value) in LABELS:
            return int(value)
    except ValueError:
        pass

    # Textual labels and abbreviations.
    label_map = {
        "vneg": 0,
        "very negative": 0,
        "neg": 1,
        "negative": 1,
        "neu": 2,
        "neutral": 2,
        "pos": 3,
        "positive": 3,
        "vpos": 4,
        "very positive": 4,
    }

    return label_map.get(s, np.nan)


def clean_text(text) -> str | float:
    """
    Apply conservative text normalization.

    The goal is to remove obvious collection artifacts while preserving
    sentiment-bearing linguistic signals such as negations and punctuation.
    """
    if pd.isna(text):
        return np.nan

    s = str(text)
    s = html.unescape(s)
    s = s.replace("\\/", "/")

    # Preserve discourse structure using placeholders.
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USER ", s)

    # Normalize whitespace introduced during scraping or export.
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ==========================================================
# Evaluation utilities
# ==========================================================

def evaluate_ordinal_classification(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: List[int] = LABELS,
) -> Dict[str, object]:
    """
    Evaluate predictions using the ordinal metric suite.

    Includes standard classification metrics plus two ordinal metrics:
    within-one accuracy and signed bias.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    return {
        "exact_accuracy": accuracy_score(y_true, y_pred),
        "within_one_accuracy": np.mean(np.abs(y_pred - y_true) <= 1),
        "mae": mean_absolute_error(y_true, y_pred),
        "signed_bias": np.mean(y_pred - y_true),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def save_scalar_metrics(metrics: Dict[str, object], path: Path) -> None:
    """Save scalar metrics as a one-row CSV."""
    scalar_metrics = {
        "exact_accuracy": metrics["exact_accuracy"],
        "within_one_accuracy": metrics["within_one_accuracy"],
        "mae": metrics["mae"],
        "signed_bias": metrics["signed_bias"],
        "macro_f1": metrics["macro_f1"],
    }
    pd.DataFrame([scalar_metrics]).to_csv(path, index=False)


def save_confusion_matrix(metrics: Dict[str, object], path: Path) -> None:
    """Save a 5x5 confusion matrix as CSV."""
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"true_{i}" for i in LABELS],
        columns=[f"pred_{i}" for i in LABELS],
    )
    cm.to_csv(path)


# ==========================================================
# Threshold calibration
# ==========================================================

def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Convert continuous scores into ordinal labels using cut-point thresholds.
    """
    return np.digitize(scores, thresholds).astype(int)


def fit_thresholds_dp(
    scores: np.ndarray,
    y_true: np.ndarray,
    labels: List[int] = LABELS,
) -> np.ndarray:
    """
    Fit four ordinal thresholds by maximizing exact accuracy on calibration data.

    This is an exact dynamic-programming search over sorted score partitions.
    It finds the best contiguous segmentation of the calibration examples into
    five ordered bins corresponding to labels 0, 1, 2, 3, 4.
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true, dtype=int)

    order = np.argsort(scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]

    n = len(scores_sorted)
    k = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # prefix_counts[c, i] = number of observations with label c among first i rows.
    prefix_counts = np.zeros((k, n + 1), dtype=int)
    for i in range(n):
        prefix_counts[:, i + 1] = prefix_counts[:, i]
        prefix_counts[label_to_idx[y_sorted[i]], i + 1] += 1

    def segment_correct(label_idx: int, start: int, end: int) -> int:
        return prefix_counts[label_idx, end] - prefix_counts[label_idx, start]

    # dp[j, i] = max correct count using first i observations and first j labels.
    dp = np.full((k + 1, n + 1), -np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    dp[0, 0] = 0

    for j in range(1, k + 1):
        label_idx = j - 1
        for i in range(j, n + 1):
            best_score = -np.inf
            best_m = 0
            for m in range(j - 1, i):
                candidate = dp[j - 1, m] + segment_correct(label_idx, m, i)
                if candidate > best_score:
                    best_score = candidate
                    best_m = m
            dp[j, i] = best_score
            back[j, i] = best_m

    # Recover split points.
    split_points = []
    i = n
    for j in range(k, 0, -1):
        m = back[j, i]
        split_points.append(m)
        i = m

    split_points = list(reversed(split_points))[1:]

    thresholds = []
    for split in split_points:
        if split <= 0:
            threshold = scores_sorted[0] - 1e-6
        elif split >= n:
            threshold = scores_sorted[-1] + 1e-6
        else:
            threshold = (scores_sorted[split - 1] + scores_sorted[split]) / 2
        thresholds.append(threshold)

    return np.array(thresholds)


# ==========================================================
# Part 1: Data engineering
# ==========================================================

def clean_training_data() -> pd.DataFrame:
    """
    Clean posts_raw.csv and save a modeling-ready dataset.
    """
    raw = pd.read_csv(RAW_PATH)
    audit = []

    def log_step(name: str, data: pd.DataFrame) -> None:
        audit.append(
            {
                "step": name,
                "rows": len(data),
                "unique_texts": data["text_clean"].nunique() if "text_clean" in data.columns else np.nan,
                "missing_labels": data["label_clean"].isna().sum() if "label_clean" in data.columns else np.nan,
                "missing_texts": data["text_clean"].isna().sum() if "text_clean" in data.columns else np.nan,
            }
        )

    raw["label_clean"] = raw["label"].apply(canonicalize_label)
    raw["text_clean"] = raw["text"].apply(clean_text)
    log_step("raw_loaded", raw)

    clean = raw.dropna(subset=["label_clean"]).copy()
    log_step("drop_missing_or_unrecognized_labels", clean)

    clean = clean.dropna(subset=["text_clean"]).copy()
    clean = clean[clean["text_clean"].str.len() > 0].copy()
    log_step("drop_missing_or_empty_text", clean)

    # Remove duplicate texts with conflicting labels.
    label_nunique_by_text = clean.groupby("text_clean")["label_clean"].nunique()
    conflicting_texts = label_nunique_by_text[label_nunique_by_text > 1].index

    clean = clean[~clean["text_clean"].isin(conflicting_texts)].copy()
    log_step("drop_conflicting_duplicate_texts", clean)

    # Deduplicate texts with consistent labels.
    clean = (
        clean.sort_values(["text_clean", "post_id"])
        .drop_duplicates(subset=["text_clean"], keep="first")
        .copy()
    )
    log_step("deduplicate_consistent_texts", clean)

    final = clean[["post_id", "text_clean", "label_clean", "source", "collected_at"]].copy()
    final = final.rename(columns={"text_clean": "text", "label_clean": "label"})
    final["label"] = final["label"].astype(int)

    final.to_csv(OUT_DIR / "posts_clean.csv", index=False)

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT_DIR / "cleaning_audit.csv", index=False)

    class_dist = final["label"].value_counts().sort_index().reset_index()
    class_dist.columns = ["label", "count"]
    class_dist["share"] = class_dist["count"] / class_dist["count"].sum()
    class_dist.to_csv(OUT_DIR / "class_distribution.csv", index=False)

    print("Part 1 complete: cleaned dataset saved to outputs/posts_clean.csv")
    return final


# ==========================================================
# Part 2: Baseline model
# ==========================================================

def train_baseline_and_predict(
    df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Train TF-IDF + Logistic Regression baseline and generate eval predictions.
    """
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf.fit_transform(X_train_full)
    X_val_tfidf = tfidf.transform(X_val)
    X_eval_tfidf = tfidf.transform(eval_df["text_clean"].astype(str))

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )
    clf.fit(X_train_tfidf, y_train_full)

    y_val_pred = clf.predict(X_val_tfidf)
    metrics = evaluate_ordinal_classification(y_val, y_val_pred)

    save_scalar_metrics(metrics, OUT_DIR / "baseline_validation_metrics.csv")
    save_confusion_matrix(metrics, OUT_DIR / "baseline_confusion_matrix.csv")

    baseline_pred = clf.predict(X_eval_tfidf)
    baseline_output = pd.DataFrame(
        {
            "post_id": eval_df["post_id"],
            "label": baseline_pred.astype(int),
        }
    )
    baseline_output["label"] = baseline_output["label"].clip(0, 4).astype(int)
    baseline_output.to_csv(PRED_BASELINE_PATH, index=False)

    print("Part 2 complete: predictions_baseline.csv generated")
    return metrics, X_train_full, X_val, y_train_full, y_val


# ==========================================================
# Part 3: Best continuous-score model
# ==========================================================

def train_best_model_and_predict(
    eval_df: pd.DataFrame,
    X_train_full: pd.Series,
    X_val: pd.Series,
    y_train_full: pd.Series,
    y_val: pd.Series,
) -> Dict[str, object]:
    """
    Train MPNet embeddings + Ridge regression + threshold calibration.
    """
    X_train_model, X_cal, y_train_model, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        stratify=y_train_full,
        random_state=SEED,
    )

    print("Loading sentence-transformer model: all-mpnet-base-v2")
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("Encoding training, calibration, validation, and eval texts...")
    X_train_embed = encoder.encode(
        X_train_model.tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    X_cal_embed = encoder.encode(
        X_cal.tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    X_val_embed = encoder.encode(
        X_val.tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    X_eval_embed = encoder.encode(
        eval_df["text_clean"].astype(str).tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )

    alpha_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    alpha_results = []

    for alpha in alpha_grid:
        ridge = Ridge(alpha=alpha, random_state=SEED)
        ridge.fit(X_train_embed, y_train_model)

        cal_scores = np.clip(ridge.predict(X_cal_embed), 0, 4)
        thresholds = fit_thresholds_dp(cal_scores, y_cal.to_numpy(), labels=LABELS)
        cal_pred = apply_thresholds(cal_scores, thresholds)
        cal_metrics = evaluate_ordinal_classification(y_cal, cal_pred)

        alpha_results.append(
            {
                "alpha": alpha,
                "cal_exact_accuracy": cal_metrics["exact_accuracy"],
                "cal_mae": cal_metrics["mae"],
                "thresholds": thresholds,
            }
        )

    alpha_results_df = pd.DataFrame(alpha_results)
    best_row = (
        alpha_results_df.sort_values(
            ["cal_exact_accuracy", "cal_mae"],
            ascending=[False, True],
        ).iloc[0]
    )

    best_alpha = float(best_row["alpha"])
    best_thresholds = best_row["thresholds"]

    print(f"Best alpha selected on calibration split: {best_alpha}")
    print(f"Best thresholds: {best_thresholds}")

    # Refit final Ridge model on the same model-training split.
    best_ridge = Ridge(alpha=best_alpha, random_state=SEED)
    best_ridge.fit(X_train_embed, y_train_model)

    val_scores = np.clip(best_ridge.predict(X_val_embed), 0, 4)
    val_pred = apply_thresholds(val_scores, best_thresholds)
    metrics = evaluate_ordinal_classification(y_val, val_pred)

    save_scalar_metrics(metrics, OUT_DIR / "mpnet_validation_metrics.csv")
    save_confusion_matrix(metrics, OUT_DIR / "mpnet_confusion_matrix.csv")

    alpha_results_df.drop(columns=["thresholds"]).to_csv(
        OUT_DIR / "mpnet_alpha_search.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "threshold": ["t0_1", "t1_2", "t2_3", "t3_4"],
            "value": best_thresholds,
        }
    ).to_csv(OUT_DIR / "mpnet_thresholds.csv", index=False)

    eval_scores = np.clip(best_ridge.predict(X_eval_embed), 0, 4)
    best_pred = apply_thresholds(eval_scores, best_thresholds)

    best_output = pd.DataFrame(
        {
            "post_id": eval_df["post_id"],
            "label": best_pred.astype(int),
        }
    )
    best_output["label"] = best_output["label"].clip(0, 4).astype(int)
    best_output.to_csv(PRED_BEST_PATH, index=False)

    print("Part 3/4 complete: predictions_best.csv generated")
    return metrics


# ==========================================================
# Prediction-file checks
# ==========================================================

def check_prediction_file(pred_df: pd.DataFrame, eval_df: pd.DataFrame, name: str) -> None:
    """Validate prediction file format required by the instructions."""
    assert list(pred_df.columns) == ["post_id", "label"], f"{name}: invalid columns"
    assert len(pred_df) == len(eval_df), f"{name}: row count mismatch"
    assert pred_df["post_id"].equals(eval_df["post_id"]), f"{name}: post_id order mismatch"
    assert pred_df["label"].between(0, 4).all(), f"{name}: labels outside 0-4"
    assert pd.api.types.is_integer_dtype(pred_df["label"]), f"{name}: label must be integer"
    print(f"{name} passed all checks.")


# ==========================================================
# Main pipeline
# ==========================================================

def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {RAW_PATH}")
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {EVAL_PATH}")

    print("Starting ordinal text classification pipeline...")

    df = clean_training_data()

    eval_df = pd.read_csv(EVAL_PATH)
    eval_df["text_clean"] = eval_df["text"].apply(clean_text)
    eval_df["text_clean"] = eval_df["text_clean"].fillna("").astype(str)

    baseline_metrics, X_train_full, X_val, y_train_full, y_val = train_baseline_and_predict(
        df=df,
        eval_df=eval_df,
    )

    best_metrics = train_best_model_and_predict(
        eval_df=eval_df,
        X_train_full=X_train_full,
        X_val=X_val,
        y_train_full=y_train_full,
        y_val=y_val,
    )

    baseline_output = pd.read_csv(PRED_BASELINE_PATH)
    best_output = pd.read_csv(PRED_BEST_PATH)

    check_prediction_file(baseline_output, eval_df, "predictions_baseline.csv")
    check_prediction_file(best_output, eval_df, "predictions_best.csv")

    comparison = pd.DataFrame(
        [
            {
                "model": "tfidf_logistic_baseline",
                "exact_accuracy": baseline_metrics["exact_accuracy"],
                "within_one_accuracy": baseline_metrics["within_one_accuracy"],
                "mae": baseline_metrics["mae"],
                "signed_bias": baseline_metrics["signed_bias"],
                "macro_f1": baseline_metrics["macro_f1"],
            },
            {
                "model": "mpnet_ridge_calibrated",
                "exact_accuracy": best_metrics["exact_accuracy"],
                "within_one_accuracy": best_metrics["within_one_accuracy"],
                "mae": best_metrics["mae"],
                "signed_bias": best_metrics["signed_bias"],
                "macro_f1": best_metrics["macro_f1"],
            },
        ]
    )
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    print("\nFinal validation comparison:")
    print(comparison)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
