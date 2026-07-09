#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Expanded ordinal text classification pipeline.

This script is designed for GitHub submission and reproducibility. It keeps
the main intermediate experiments from the notebook while removing Colab-only
code such as files.upload(), files.download(), and notebook display calls.

Expected input files:
    data/posts_raw.csv
    data/eval_posts.csv

Generated submission files:
    predictions_baseline.csv
    predictions_best.csv

Generated diagnostic files:
    outputs/*.csv
"""

import html
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer


# ==========================================================
# Global configuration
# ==========================================================

SEED = 42
LABELS = [0, 1, 2, 3, 4]

random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "posts_raw.csv"
EVAL_PATH = DATA_DIR / "eval_posts.csv"


# ==========================================================
# Utility functions
# ==========================================================

def canonicalize_label(x):
    """
    Convert heterogeneous sentiment labels to a standardized
    five-point ordinal sentiment scale.

    Returns
    -------
    int or np.nan
        Integer label in {0,1,2,3,4}, or np.nan if missing/unrecognized.
    """

    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in {"", "?", "nan", "none", "missing"}:
        return np.nan

    try:
        value = float(s)
        if value.is_integer() and int(value) in LABELS:
            return int(value)
    except ValueError:
        pass

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


def clean_text(text):
    """
    Light text normalization.

    The pipeline intentionally avoids aggressive preprocessing such as
    stopword removal or stemming because negations and function words often
    carry sentiment information in short social-media texts.
    """

    if pd.isna(text):
        return np.nan

    s = str(text)
    s = html.unescape(s)
    s = s.replace("\\/", "/")

    # Preserve structural information with placeholders.
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USER ", s)

    # Normalize whitespace.
    s = re.sub(r"\s+", " ", s).strip()

    return s


def evaluate_ordinal_classification(y_true, y_pred, labels=LABELS):
    """
    Evaluate ordinal classification results using the lab's metric suite.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "exact_accuracy": accuracy_score(y_true, y_pred),
        "within_one_accuracy": np.mean(np.abs(y_pred - y_true) <= 1),
        "mae": mean_absolute_error(y_true, y_pred),
        "signed_bias": np.mean(y_pred - y_true),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "per_class_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def scalar_metrics_dict(model_name: str, metrics: Dict) -> Dict:
    return {
        "model": model_name,
        "exact_accuracy": metrics["exact_accuracy"],
        "within_one_accuracy": metrics["within_one_accuracy"],
        "mae": metrics["mae"],
        "signed_bias": metrics["signed_bias"],
        "macro_f1": metrics["macro_f1"],
    }


def save_metric_outputs(prefix: str, metrics: Dict):
    """
    Save scalar metrics, per-class F1, and confusion matrix.
    """

    pd.DataFrame([scalar_metrics_dict(prefix, metrics)]).to_csv(
        OUT_DIR / f"{prefix}_metrics.csv",
        index=False,
    )

    pd.DataFrame({
        "label": LABELS,
        "f1": metrics["per_class_f1"],
    }).to_csv(
        OUT_DIR / f"{prefix}_per_class_f1.csv",
        index=False,
    )

    pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"true_{i}" for i in LABELS],
        columns=[f"pred_{i}" for i in LABELS],
    ).to_csv(OUT_DIR / f"{prefix}_confusion_matrix.csv")


def round_scores_to_labels(scores):
    return np.clip(np.rint(scores), 0, 4).astype(int)


def apply_thresholds(scores, thresholds):
    """
    Convert continuous scores to ordinal labels using fitted cut points.
    """

    return np.digitize(scores, thresholds).astype(int)


def fit_thresholds_dp(scores, y_true, labels=LABELS):
    """
    Fit four thresholds by maximizing exact accuracy on a calibration set.

    This is an exact dynamic-programming search over partitions of sorted
    continuous scores into five contiguous bins corresponding to labels 0-4.
    """

    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    order = np.argsort(scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]

    n = len(scores_sorted)
    k = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    prefix_counts = np.zeros((k, n + 1), dtype=int)

    for i in range(n):
        prefix_counts[:, i + 1] = prefix_counts[:, i]
        prefix_counts[label_to_idx[y_sorted[i]], i + 1] += 1

    def segment_correct(label_idx, start, end):
        return prefix_counts[label_idx, end] - prefix_counts[label_idx, start]

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

    return np.asarray(thresholds)


def check_prediction_file(pred_df, eval_df, name):
    assert list(pred_df.columns) == ["post_id", "label"], (
        f"{name}: columns must be ['post_id', 'label']"
    )
    assert len(pred_df) == len(eval_df), (
        f"{name}: row count does not match eval_posts.csv"
    )
    assert pred_df["post_id"].equals(eval_df["post_id"]), (
        f"{name}: post_id order does not match eval_posts.csv"
    )
    assert pred_df["label"].between(0, 4).all(), (
        f"{name}: labels must be in 0-4"
    )
    assert pd.api.types.is_integer_dtype(pred_df["label"]), (
        f"{name}: label must be integer dtype"
    )
    print(f"{name} passed all checks.")


# ==========================================================
# Part 1: Data cleaning
# ==========================================================

def clean_training_data() -> pd.DataFrame:
    """
    Turn posts_raw.csv into a modeling-ready dataset.
    """

    df = pd.read_csv(RAW_PATH)

    audit = []

    def log_step(name, data):
        audit.append({
            "step": name,
            "rows": len(data),
            "unique_texts": (
                data["text_clean"].nunique()
                if "text_clean" in data.columns
                else np.nan
            ),
            "missing_labels": (
                data["label_clean"].isna().sum()
                if "label_clean" in data.columns
                else np.nan
            ),
            "missing_texts": (
                data["text_clean"].isna().sum()
                if "text_clean" in data.columns
                else np.nan
            ),
        })

    df["label_clean"] = df["label"].apply(canonicalize_label)
    df["text_clean"] = df["text"].apply(clean_text)

    log_step("raw_loaded", df)

    clean = df.dropna(subset=["label_clean"]).copy()
    log_step("drop_missing_or_unrecognized_labels", clean)

    clean = clean.dropna(subset=["text_clean"]).copy()
    clean = clean[clean["text_clean"].str.len() > 0].copy()
    log_step("drop_missing_or_empty_text", clean)

    label_nunique_by_text = clean.groupby("text_clean")["label_clean"].nunique()
    conflicting_texts = label_nunique_by_text[label_nunique_by_text > 1].index

    conflict_examples = (
        clean[clean["text_clean"].isin(conflicting_texts)]
        .sort_values("text_clean")
    )
    conflict_examples[[
        "post_id",
        "text",
        "label",
        "label_clean",
        "text_clean",
    ]].to_csv(OUT_DIR / "conflicting_duplicate_examples.csv", index=False)

    clean = clean[~clean["text_clean"].isin(conflicting_texts)].copy()
    log_step("drop_conflicting_duplicate_texts", clean)

    clean = (
        clean.sort_values(["text_clean", "post_id"])
        .drop_duplicates(subset=["text_clean"], keep="first")
        .copy()
    )
    log_step("deduplicate_consistent_texts", clean)

    final = clean[[
        "post_id",
        "text_clean",
        "label_clean",
        "source",
        "collected_at",
    ]].copy()

    final = final.rename(columns={
        "text_clean": "text",
        "label_clean": "label",
    })
    final["label"] = final["label"].astype(int)

    final.to_csv(OUT_DIR / "posts_clean.csv", index=False)

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT_DIR / "cleaning_audit.csv", index=False)

    class_dist = final["label"].value_counts().sort_index().reset_index()
    class_dist.columns = ["label", "count"]
    class_dist["share"] = class_dist["count"] / class_dist["count"].sum()
    class_dist.to_csv(OUT_DIR / "class_distribution.csv", index=False)

    print("Part 1 complete: cleaned data saved.")
    print(audit_df)
    print(class_dist)

    return final


# ==========================================================
# Data splitting
# ==========================================================

def make_splits(df: pd.DataFrame):
    """
    Recreate the validation split used across all models.
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

    X_train_model, X_cal, y_train_model, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        stratify=y_train_full,
        random_state=SEED,
    )

    print("Split sizes:")
    print("  train_full:", len(X_train_full))
    print("  train_model:", len(X_train_model))
    print("  calibration:", len(X_cal))
    print("  validation:", len(X_val))

    return X_train_full, X_val, y_train_full, y_val, X_train_model, X_cal, y_train_model, y_cal


# ==========================================================
# Part 2: Baseline TF-IDF + Logistic Regression
# ==========================================================

def train_baseline(
    X_train_full,
    X_val,
    y_train_full,
    y_val,
    eval_df,
):
    """
    Train the Part 2 sparse-feature baseline and predict eval posts.
    """

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

    save_metric_outputs("baseline", metrics)

    eval_pred = clf.predict(X_eval_tfidf)

    baseline_output = pd.DataFrame({
        "post_id": eval_df["post_id"],
        "label": eval_pred.astype(int),
    })
    baseline_output["label"] = baseline_output["label"].clip(0, 4).astype(int)
    baseline_output.to_csv("predictions_baseline.csv", index=False)

    print("Part 2 baseline complete.")
    print(scalar_metrics_dict("baseline", metrics))

    return {
        "name": "part2_baseline_tfidf_logistic",
        "tfidf": tfidf,
        "model": clf,
        "metrics": metrics,
    }


# ==========================================================
# Part 3A: Expected value over class probabilities
# ==========================================================

def expected_value_probability_model(
    X_train_model,
    X_cal,
    X_val,
    y_train_model,
    y_cal,
    y_val,
):
    """
    Continuous score = expected value of class probabilities from a classifier.
    """

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf.fit_transform(X_train_model)
    X_cal_tfidf = tfidf.transform(X_cal)
    X_val_tfidf = tfidf.transform(X_val)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )

    clf.fit(X_train_tfidf, y_train_model)

    class_values = np.asarray(LABELS)

    cal_scores = clf.predict_proba(X_cal_tfidf) @ class_values
    val_scores = clf.predict_proba(X_val_tfidf) @ class_values

    rounding_pred = round_scores_to_labels(val_scores)
    rounding_metrics = evaluate_ordinal_classification(y_val, rounding_pred)

    thresholds = fit_thresholds_dp(cal_scores, y_cal.to_numpy(), LABELS)
    calibrated_pred = apply_thresholds(val_scores, thresholds)
    calibrated_metrics = evaluate_ordinal_classification(y_val, calibrated_pred)

    save_metric_outputs("expected_value_rounding", rounding_metrics)
    save_metric_outputs("expected_value_calibrated", calibrated_metrics)

    pd.DataFrame({
        "threshold": ["t0_1", "t1_2", "t2_3", "t3_4"],
        "value": thresholds,
    }).to_csv(OUT_DIR / "expected_value_thresholds.csv", index=False)

    print("Expected-value probability model complete.")
    print(scalar_metrics_dict("expected_value_calibrated", calibrated_metrics))

    return {
        "name": "expected_value_calibrated",
        "metrics": calibrated_metrics,
        "rounding_metrics": rounding_metrics,
        "thresholds": thresholds,
    }


# ==========================================================
# Part 3B: TF-IDF + Ridge Regression + Threshold Calibration
# ==========================================================

def tfidf_ridge_model(
    X_train_model,
    X_cal,
    X_val,
    y_train_model,
    y_cal,
    y_val,
):
    """
    Directly regress ordinal labels as continuous outcomes using sparse TF-IDF.
    """

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
    )

    X_train = tfidf.fit_transform(X_train_model)
    X_cal_mat = tfidf.transform(X_cal)
    X_val_mat = tfidf.transform(X_val)

    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train, y_train_model)

    cal_scores = np.clip(ridge.predict(X_cal_mat), 0, 4)
    val_scores = np.clip(ridge.predict(X_val_mat), 0, 4)

    rounding_pred = round_scores_to_labels(val_scores)
    rounding_metrics = evaluate_ordinal_classification(y_val, rounding_pred)

    thresholds = fit_thresholds_dp(cal_scores, y_cal.to_numpy(), LABELS)
    calibrated_pred = apply_thresholds(val_scores, thresholds)
    calibrated_metrics = evaluate_ordinal_classification(y_val, calibrated_pred)

    save_metric_outputs("tfidf_ridge_rounding", rounding_metrics)
    save_metric_outputs("tfidf_ridge_calibrated", calibrated_metrics)

    pd.DataFrame({
        "threshold": ["t0_1", "t1_2", "t2_3", "t3_4"],
        "value": thresholds,
    }).to_csv(OUT_DIR / "tfidf_ridge_thresholds.csv", index=False)

    print("TF-IDF Ridge model complete.")
    print(scalar_metrics_dict("tfidf_ridge_calibrated", calibrated_metrics))

    return {
        "name": "tfidf_ridge_calibrated",
        "metrics": calibrated_metrics,
        "rounding_metrics": rounding_metrics,
        "thresholds": thresholds,
    }


# ==========================================================
# Part 3C: Ordinal Logistic Regression with mord
# ==========================================================

def ordinal_logistic_model(
    X_train_model,
    X_val,
    y_train_model,
    y_val,
):
    """
    Optional ordinal logistic model using mord.

    This model is included as an intermediate ordinal experiment. It directly
    predicts ordinal classes but does not expose a continuous score in all mord
    versions, so it is not selected as the primary final model.
    """

    try:
        from mord import LogisticIT
    except ImportError:
        print("mord is not installed; skipping ordinal logistic experiment.")
        return None

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
    )

    X_train = tfidf.fit_transform(X_train_model)
    X_val_mat = tfidf.transform(X_val)

    model = LogisticIT(alpha=1.0)
    model.fit(X_train, y_train_model)

    pred = model.predict(X_val_mat)
    metrics = evaluate_ordinal_classification(y_val, pred)

    save_metric_outputs("ordinal_logistic_direct", metrics)

    print("Ordinal logistic model complete.")
    print(scalar_metrics_dict("ordinal_logistic_direct", metrics))

    return {
        "name": "ordinal_logistic_direct",
        "metrics": metrics,
    }


# ==========================================================
# Part 3D: Cumulative binary ordinal classifiers
# ==========================================================

def cumulative_binary_model(
    X_train_model,
    X_cal,
    X_val,
    y_train_model,
    y_cal,
    y_val,
):
    """
    Train four cumulative classifiers:
        P(y > 0), P(y > 1), P(y > 2), P(y > 3)

    Continuous score is the sum of cumulative probabilities.
    """

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )

    X_train = tfidf.fit_transform(X_train_model)
    X_cal_mat = tfidf.transform(X_cal)
    X_val_mat = tfidf.transform(X_val)

    binary_models = {}

    for t in [0, 1, 2, 3]:
        y_binary = (y_train_model > t).astype(int)

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            C=2.0,
            random_state=SEED,
        )

        clf.fit(X_train, y_binary)
        binary_models[t] = clf

    def cumulative_probs(models, X_matrix):
        probs = []

        for t in [0, 1, 2, 3]:
            p = models[t].predict_proba(X_matrix)[:, 1]
            probs.append(p)

        return np.vstack(probs).T

    def enforce_monotonicity(probs):
        probs_mono = probs.copy()

        for j in range(1, probs_mono.shape[1]):
            probs_mono[:, j] = np.minimum(probs_mono[:, j - 1], probs_mono[:, j])

        return probs_mono

    cal_probs = enforce_monotonicity(cumulative_probs(binary_models, X_cal_mat))
    val_probs = enforce_monotonicity(cumulative_probs(binary_models, X_val_mat))

    cal_scores = cal_probs.sum(axis=1)
    val_scores = val_probs.sum(axis=1)

    direct_pred = (val_probs >= 0.5).sum(axis=1)
    direct_metrics = evaluate_ordinal_classification(y_val, direct_pred)

    rounding_pred = round_scores_to_labels(val_scores)
    rounding_metrics = evaluate_ordinal_classification(y_val, rounding_pred)

    thresholds = fit_thresholds_dp(cal_scores, y_cal.to_numpy(), LABELS)
    calibrated_pred = apply_thresholds(val_scores, thresholds)
    calibrated_metrics = evaluate_ordinal_classification(y_val, calibrated_pred)

    save_metric_outputs("cumulative_binary_direct", direct_metrics)
    save_metric_outputs("cumulative_binary_rounding", rounding_metrics)
    save_metric_outputs("cumulative_binary_calibrated", calibrated_metrics)

    pd.DataFrame({
        "threshold": ["t0_1", "t1_2", "t2_3", "t3_4"],
        "value": thresholds,
    }).to_csv(OUT_DIR / "cumulative_binary_thresholds.csv", index=False)

    print("Cumulative binary model complete.")
    print(scalar_metrics_dict("cumulative_binary_calibrated", calibrated_metrics))

    return {
        "name": "cumulative_binary_calibrated",
        "metrics": calibrated_metrics,
        "direct_metrics": direct_metrics,
        "rounding_metrics": rounding_metrics,
        "thresholds": thresholds,
    }


# ==========================================================
# Part 3E: Sentence embeddings + Ridge + Threshold Calibration
# ==========================================================

def embedding_ridge_model(
    model_name: str,
    short_name: str,
    X_train_model,
    X_cal,
    X_val,
    y_train_model,
    y_cal,
    y_val,
    eval_df=None,
    alpha_grid=None,
):
    """
    Frozen sentence embeddings + Ridge regression + threshold calibration.

    If eval_df is provided, the function also generates eval predictions.
    """

    if alpha_grid is None:
        alpha_grid = [1.0]

    print(f"Loading sentence-transformer: {model_name}")
    encoder = SentenceTransformer(model_name)

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

    alpha_results = []

    for alpha in alpha_grid:
        ridge = Ridge(alpha=alpha, random_state=SEED)
        ridge.fit(X_train_embed, y_train_model)

        cal_scores = np.clip(ridge.predict(X_cal_embed), 0, 4)
        thresholds = fit_thresholds_dp(cal_scores, y_cal.to_numpy(), LABELS)

        cal_pred = apply_thresholds(cal_scores, thresholds)
        cal_metrics = evaluate_ordinal_classification(y_cal, cal_pred)

        alpha_results.append({
            "alpha": alpha,
            "cal_exact_accuracy": cal_metrics["exact_accuracy"],
            "cal_mae": cal_metrics["mae"],
            "thresholds": thresholds,
        })

    alpha_results_df = pd.DataFrame(alpha_results)
    best_row = (
        alpha_results_df
        .sort_values(["cal_exact_accuracy", "cal_mae"], ascending=[False, True])
        .iloc[0]
    )

    best_alpha = best_row["alpha"]
    best_thresholds = best_row["thresholds"]

    ridge = Ridge(alpha=best_alpha, random_state=SEED)
    ridge.fit(X_train_embed, y_train_model)

    val_scores = np.clip(ridge.predict(X_val_embed), 0, 4)

    rounding_pred = round_scores_to_labels(val_scores)
    rounding_metrics = evaluate_ordinal_classification(y_val, rounding_pred)

    calibrated_pred = apply_thresholds(val_scores, best_thresholds)
    calibrated_metrics = evaluate_ordinal_classification(y_val, calibrated_pred)

    save_metric_outputs(f"{short_name}_ridge_rounding", rounding_metrics)
    save_metric_outputs(f"{short_name}_ridge_calibrated", calibrated_metrics)

    alpha_results_df.drop(columns=["thresholds"]).to_csv(
        OUT_DIR / f"{short_name}_alpha_search.csv",
        index=False,
    )

    pd.DataFrame({
        "threshold": ["t0_1", "t1_2", "t2_3", "t3_4"],
        "value": best_thresholds,
    }).to_csv(OUT_DIR / f"{short_name}_thresholds.csv", index=False)

    eval_pred = None
    if eval_df is not None:
        X_eval_embed = encoder.encode(
            eval_df["text_clean"].astype(str).tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32,
        )
        eval_scores = np.clip(ridge.predict(X_eval_embed), 0, 4)
        eval_pred = apply_thresholds(eval_scores, best_thresholds)

    print(f"{short_name} embedding Ridge model complete.")
    print("Best alpha:", best_alpha)
    print(scalar_metrics_dict(f"{short_name}_ridge_calibrated", calibrated_metrics))

    return {
        "name": f"{short_name}_ridge_calibrated",
        "encoder": encoder,
        "ridge": ridge,
        "best_alpha": best_alpha,
        "thresholds": best_thresholds,
        "metrics": calibrated_metrics,
        "rounding_metrics": rounding_metrics,
        "eval_pred": eval_pred,
    }


# ==========================================================
# Main pipeline
# ==========================================================

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {RAW_PATH}")

    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {EVAL_PATH}")

    # Part 1
    df = clean_training_data()

    # Load and clean evaluation data.
    eval_df = pd.read_csv(EVAL_PATH)
    eval_df["text_clean"] = eval_df["text"].apply(clean_text)
    eval_df["text_clean"] = eval_df["text_clean"].fillna("").astype(str)

    # Shared splits for fair validation comparison.
    (
        X_train_full,
        X_val,
        y_train_full,
        y_val,
        X_train_model,
        X_cal,
        y_train_model,
        y_cal,
    ) = make_splits(df)

    # Part 2 baseline.
    baseline = train_baseline(
        X_train_full,
        X_val,
        y_train_full,
        y_val,
        eval_df,
    )

    # Intermediate Part 3 experiments.
    experiments = [baseline]

    experiments.append(
        expected_value_probability_model(
            X_train_model,
            X_cal,
            X_val,
            y_train_model,
            y_cal,
            y_val,
        )
    )

    experiments.append(
        tfidf_ridge_model(
            X_train_model,
            X_cal,
            X_val,
            y_train_model,
            y_cal,
            y_val,
        )
    )

    ordinal_logistic_result = ordinal_logistic_model(
        X_train_model,
        X_val,
        y_train_model,
        y_val,
    )
    if ordinal_logistic_result is not None:
        experiments.append(ordinal_logistic_result)

    experiments.append(
        cumulative_binary_model(
            X_train_model,
            X_cal,
            X_val,
            y_train_model,
            y_cal,
            y_val,
        )
    )

    # Sentence embedding Ridge models.
    minilm_result = embedding_ridge_model(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        short_name="minilm",
        X_train_model=X_train_model,
        X_cal=X_cal,
        X_val=X_val,
        y_train_model=y_train_model,
        y_cal=y_cal,
        y_val=y_val,
        eval_df=None,
        alpha_grid=[1.0],
    )
    experiments.append(minilm_result)

    mpnet_result = embedding_ridge_model(
        model_name="sentence-transformers/all-mpnet-base-v2",
        short_name="mpnet",
        X_train_model=X_train_model,
        X_cal=X_cal,
        X_val=X_val,
        y_train_model=y_train_model,
        y_cal=y_cal,
        y_val=y_val,
        eval_df=eval_df,
        alpha_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    )
    experiments.append(mpnet_result)

    # Save comparison table.
    comparison_rows = []

    for exp in experiments:
        comparison_rows.append(scalar_metrics_dict(exp["name"], exp["metrics"]))

        if "rounding_metrics" in exp:
            comparison_rows.append(
                scalar_metrics_dict(exp["name"] + "_rounding", exp["rounding_metrics"])
            )

        if "direct_metrics" in exp:
            comparison_rows.append(
                scalar_metrics_dict(exp["name"] + "_direct", exp["direct_metrics"])
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values("exact_accuracy", ascending=False)
    comparison.to_csv(OUT_DIR / "all_model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(comparison)

    # Official best-model predictions use MPNet + Ridge + calibrated thresholds.
    if mpnet_result["eval_pred"] is None:
        raise RuntimeError("MPNet model did not produce eval predictions.")

    best_output = pd.DataFrame({
        "post_id": eval_df["post_id"],
        "label": mpnet_result["eval_pred"].astype(int),
    })
    best_output["label"] = best_output["label"].clip(0, 4).astype(int)
    best_output.to_csv("predictions_best.csv", index=False)

    baseline_output = pd.read_csv("predictions_baseline.csv")
    best_output_check = pd.read_csv("predictions_best.csv")

    check_prediction_file(baseline_output, eval_df, "predictions_baseline.csv")
    check_prediction_file(best_output_check, eval_df, "predictions_best.csv")

    print("\nFinal selected best model: MPNet + Ridge + calibrated thresholds")
    print("Baseline exact accuracy:", baseline["metrics"]["exact_accuracy"])
    print("Best model exact accuracy:", mpnet_result["metrics"]["exact_accuracy"])
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
