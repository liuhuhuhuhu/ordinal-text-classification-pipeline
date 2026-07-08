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

---

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

---

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

---

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

---

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

---

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

---

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
