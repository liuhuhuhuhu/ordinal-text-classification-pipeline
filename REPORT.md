# Part 1 — Data Engineering

## 1. Initial Data Audit

The raw dataset contained **9,980 observations** and five variables:

- `post_id`
- `text`
- `label`
- `source`
- `collected_at`

Missing values were concentrated in the sentiment labels and, to a much smaller extent, in the text field.

| Variable | Missing Values |
|----------|---------------|
| text | 8 |
| label | 123 |
| post_id | 0 |
| source | 0 |
| collected_at | 0 |

The raw labels appeared in heterogeneous formats including:

- integers (`0`, `1`, `2`, `3`, `4`)
- floating-point values (`1.0`, `3.0`)
- abbreviations (`pos`, `neg`, `neu`, `vpos`, `vneg`)
- textual labels with inconsistent capitalization (e.g., `Positive`, `VERY NEGATIVE`, `neutral`, `Neutral`)

This heterogeneity required explicit label harmonization before modeling.


## 2. Label Harmonization

All labels were mapped to a canonical five-point ordinal sentiment scale:

| Final Label | Meaning |
|------------|---------|
| 0 | Very Negative |
| 1 | Negative |
| 2 | Neutral |
| 3 | Positive |
| 4 | Very Positive |

The harmonization procedure converted:

- numeric strings such as `3` and `3.0`;
- abbreviations such as `pos`, `neg`, `neu`, `vpos`, and `vneg`;
- textual labels with inconsistent capitalization.

Unrecognized labels and explicit placeholders (e.g., `?`) were treated as missing values and removed from the modeling dataset.

A total of **191 observations** were removed during this step.


## 3. Text Normalization

A conservative text normalization strategy was adopted.

The following transformations were applied:

- HTML entity decoding;
- normalization of escaped characters;
- replacement of URLs with the placeholder `URL`;
- replacement of user mentions with the placeholder `USER`;
- whitespace normalization.

Aggressive preprocessing techniques such as:

- stopword removal;
- stemming;
- lemmatization;

were intentionally avoided because negations, function words, and punctuation often carry sentiment information in short social-media texts.

This approach follows common practice in computational social science and political text analysis, where preserving linguistic information is generally preferred over aggressive text reduction.


## 4. Duplicate Handling

Duplicate texts were handled differently depending on whether their labels agreed.

### 4.1 Consistent Duplicate Texts

If multiple observations contained identical texts and identical labels, only a single representative copy was retained.

This avoids overweighting repeated content during model training.

### 4.2 Conflicting Duplicate Texts

If identical texts received different labels, the entire duplicate group was removed.

This conservative strategy was chosen because conflicting labels indicate disagreement in the supervision signal and uncertainty regarding the true sentiment label.

Removing these observations prioritizes label quality over sample size and avoids training on contradictory examples.

The cleaning pipeline identified:

- **47 conflicting duplicate groups**
- **94 observations involved in annotation conflicts**


## 5. Cleaning Audit Trail

The table below summarizes the number of observations remaining after each cleaning step.

| Cleaning Step | Rows Remaining |
|--------------|---------------|
| Raw dataset loaded | 9,980 |
| Remove missing or unrecognized labels | 9,789 |
| Remove missing or empty texts | 9,765 |
| Remove conflicting duplicate texts | 9,671 |
| Remove consistent duplicate texts | 9,411 |

The final modeling dataset contains **9,411 unique observations**.


## 6. Final Class Distribution

The final class distribution after cleaning is shown below.

| Label | Count | Share |
|------|------:|------:|
| 0 | 1,206 | 12.81% |
| 1 | 2,442 | 25.95% |
| 2 | 1,805 | 19.18% |
| 3 | 2,538 | 26.97% |
| 4 | 1,420 | 15.09% |

The resulting dataset exhibits moderate class imbalance, with mildly positive and mildly negative posts being more common than extreme sentiment categories.

This distribution is broadly consistent with expectations for naturally occurring social-media sentiment data.


## 7. Summary of Cleaning Decisions

| Issue | Decision |
|------|----------|
| Heterogeneous label formats | Harmonized to a common ordinal scale (0–4) |
| Missing labels | Removed |
| Placeholder labels (`?`) | Treated as missing and removed |
| Missing texts | Removed |
| URLs | Replaced with `URL` placeholder |
| User mentions | Replaced with `USER` placeholder |
| Duplicate texts with identical labels | Keep one representative copy |
| Duplicate texts with conflicting labels | Remove entire duplicate group |
| Stopword removal / stemming | Not applied |

The entire cleaning pipeline was implemented programmatically and can be reproduced end-to-end from the raw data using the provided scripts.


# Part 2 — Baseline Model

## 2.1 Model Specification

As a classical baseline, I implemented a sparse text representation using TF-IDF features combined with a linear classifier.

The baseline pipeline consisted of:

- TF-IDF vectorization;
- Logistic Regression classifier;
- stratified train-validation split;
- fixed random seed for reproducibility.

The final training pipeline was:

```
Text
↓
TF-IDF representation
↓
Logistic Regression
↓
Ordinal prediction (0–4)
```

This baseline serves as the reference point for the ordinal-aware continuous scoring model developed in Part 3.


## 2.2 Train-Validation Split

The cleaned dataset containing 9,411 observations was divided into:

- Training set: 7,528 observations (80%)
- Validation set: 1,883 observations (20%)

Stratified sampling was applied using the sentiment labels to preserve the class distribution across both subsets.

A fixed random seed (`random_state=42`) was used to ensure reproducibility.

The resulting label distributions were nearly identical across the training and validation sets, confirming successful stratification.


## 2.3 Feature Engineering

Text was represented using TF-IDF features.

The vectorizer configuration was:

| Parameter | Value |
|----------|------|
| ngram_range | (1,2) |
| min_df | 3 |
| max_df | 0.95 |
| max_features | 30,000 |
| sublinear_tf | True |

Both unigrams and bigrams were included because sentiment in short social-media texts is often expressed through short phrases such as:

- "not good"
- "very bad"
- "so happy"

The resulting feature matrices had dimensions:

| Dataset | Shape |
|---------|-------|
| Training | (7528, 10903) |
| Validation | (1883, 10903) |


## 2.4 Classifier

The baseline classifier was multinomial Logistic Regression.

The following settings were used:

| Parameter | Value |
|----------|------|
| max_iter | 2000 |
| class_weight | balanced |
| solver | liblinear |
| random_state | 42 |

Class balancing was enabled to mitigate the moderate class imbalance present in the cleaned dataset.


## 2.5 Validation Performance

The baseline model achieved:

| Metric | Value |
|--------|------|
| Exact Accuracy | 0.4057 |
| Within-One Accuracy | 0.7950 |
| MAE | 0.8842 |
| Signed Bias | 0.0133 |
| Macro-F1 | 0.3905 |

The exact accuracy is consistent with expectations for a classical sparse-feature baseline on a five-class ordinal sentiment classification task.


## 2.6 Per-Class Performance

| Label | F1 Score |
|------|---------|
| 0 | 0.3780 |
| 1 | 0.4476 |
| 2 | 0.2376 |
| 3 | 0.4685 |
| 4 | 0.4211 |

The model performed best on mildly positive (`label=3`) and mildly negative (`label=1`) posts.

The most difficult category was the neutral class (`label=2`), which achieved the lowest F1 score.

This pattern is common in sentiment classification because neutral posts often share lexical characteristics with weakly positive and weakly negative posts.


## 2.7 Error Structure

The confusion matrix reveals that most classification errors occurred between adjacent sentiment categories rather than extreme categories.

Examples include:

- `0 → 1`
- `1 → 2`
- `2 → 3`
- `3 → 4`

Large errors such as:

- `0 → 4`
- `4 → 0`

were comparatively rare.

This observation is supported by the high within-one accuracy of 79.5%.

The model therefore appears to capture the ordinal structure of the sentiment labels despite relatively modest exact accuracy.


## 2.8 Signed Bias Analysis

The signed bias was:

```
0.0133
```

which is very close to zero.

This suggests that the model does not systematically overpredict or underpredict sentiment and exhibits little directional bias across the ordinal scale.


Confusion Matrix Analysis

The confusion matrix reveals that prediction errors are concentrated near the diagonal.

Most mistakes occur between adjacent sentiment categories such as:

- 0 ↔ 1
- 1 ↔ 2
- 2 ↔ 3
- 3 ↔ 4

Extreme errors, including:

- 0 → 4
- 4 → 0

are comparatively rare.

This pattern indicates that the model captures the ordinal structure of the sentiment scale despite relatively modest exact accuracy.

The neutral category (label = 2) was the most difficult to classify, exhibiting the lowest F1 score and substantial confusion with neighboring categories.

Such behavior is common in sentiment analysis tasks because neutral texts often overlap lexically with weakly positive and weakly negative expressions.

## 2.9 Summary

The TF-IDF + Logistic Regression baseline provides a strong and interpretable benchmark for the task.

Its main limitations are:

- difficulty distinguishing neutral from weak sentiment;
- inability to explicitly model the ordinal relationships among labels.

These limitations motivate the continuous scoring and threshold calibration approach introduced in Part 3.










