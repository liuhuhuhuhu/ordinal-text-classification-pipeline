# RA Coding Test — Ordinal Text Classification Pipeline

**Author:** Yurou Liu  

# 1. Data Engineering

The raw dataset contained **9,980 social-media posts** with heterogeneous labels, missing values, text artifacts, and duplicate observations.

## Cleaning decisions

1. Canonicalized all label representations into the ordinal scale:

```text
0 = very negative
1 = negative
2 = neutral
3 = positive
4 = very positive
```

2. Removed rows with missing or invalid labels.
3. Removed empty text observations after preprocessing.
4. Removed duplicate texts with **conflicting labels**, treating them as irreducible annotation noise.
5. Retained a single representative copy for duplicates with identical labels.

## Cleaning audit

| Step | Rows |
|------|------|
| Raw dataset | 9980 |
| Remove missing labels | 9789 |
| Remove empty texts | 9765 |
| Remove conflicting duplicates | 9671 |
| Deduplicate consistent copies | 9411 |

A total of **47 conflicting duplicate groups (94 rows)** were removed.

Final class distribution:

| Label | Share |
|------|------|
| 0 | 12.8% |
| 1 | 25.9% |
| 2 | 19.2% |
| 3 | 27.0% |
| 4 | 15.1% |

The dataset exhibits moderate imbalance but does not require explicit resampling.


# 2. Baseline Model

The baseline follows a standard sparse text classification pipeline:

```text
TF-IDF → Logistic Regression
```

A stratified train-validation split with a fixed random seed was used.

## Validation performance

| Metric | Value |
|-------|------|
| Exact Accuracy | 0.4057 |
| Within-One Accuracy | 0.7950 |
| MAE | 0.8842 |
| Signed Bias | 0.0133 |
| Macro-F1 | 0.3905 |

The baseline captures lexical sentiment cues reasonably well but ignores the ordinal structure of labels.

# 3. Continuous Scoring and Threshold Calibration

Because sentiment labels satisfy:

```text
0 < 1 < 2 < 3 < 4
```

treating them as unordered classes discards useful information.

Several ordinal-aware approaches were explored, including:

- expected value over class probabilities,
- ordinal logistic regression,
- cumulative binary classifiers,
- regression on sentence embeddings.

The best-performing model used:

```text
all-mpnet-base-v2 embeddings
        ↓
Ridge Regression
        ↓
Continuous sentiment score
        ↓
Threshold calibration
        ↓
Ordinal labels
```

The model generates a continuous latent sentiment score and estimates four cut-point thresholds on a held-out calibration split by maximizing exact accuracy.

This design separates:

- representation learning,
- score estimation,
- discretization.

Threshold calibration consistently outperformed naive rounding by adapting decision boundaries to empirical score distributions.

## Validation performance

| Metric | Baseline | Best Model |
|-------|---------|-----------|
| Exact Accuracy | 0.4057 | **0.4663** |
| Within-One Accuracy | 0.7950 | **0.9134** |
| MAE | 0.8842 | **0.6314** |
| Signed Bias | 0.0133 | 0.0674 |
| Macro-F1 | 0.3905 | **0.4125** |

The final model improved exact accuracy by **6.1 percentage points**, reaching the expected mid-to-high 40% range described in the instructions.

# 4. Error Analysis

The most difficult class is **label 2 (neutral sentiment)**.

Most errors involve adjacent classes:

```text
2 → 1
2 → 3
1 → 2
3 → 2
```

while extreme mistakes such as:

```text
0 → 4
4 → 0
```

are rare.

This suggests that the model learned the latent sentiment ordering even when exact class assignment remained uncertain.

Extreme sentiment classes (0 and 4) are comparatively easier because they contain stronger lexical and semantic polarity cues.

Prediction errors therefore appear largely **ordinal rather than random**, which is desirable for downstream social-science applications.


# 5. Future Work

Given additional time, I would explore:

- transformer fine-tuning with regression objectives,
- ordinal neural losses,
- isotonic calibration,
- sparse-dense ensemble models.


# 6. Hours Log

| Task | Hours |
|------|------|
| Data engineering | 1 |
| Baseline model | 2 |
| Continuous scoring and calibration | 4 |
| Reporting | 2 |
| **Total** | **9** |








