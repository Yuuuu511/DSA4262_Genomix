# Preparation for training on all three datasets

This document records the preparation audit. On 6 October 2026 the user selected option 3: data2 labels at least 0.5 are positive. Training has now completed; see [the all-datasets training report](all_datasets_training.md). The earlier data0 + data1 model remains available.

## Verified data2 contents

Data2 contains 1,323 site observations: 189 positions repeated across seven transcript identifiers. Each identifier has one constant label:

| Identifier | Label | Observations |
|---|---:|---:|
| tx_id_0 | 1.0 | 189 |
| tx_id_1 | 0.95 | 189 |
| tx_id_2 | 0.75 | 189 |
| tx_id_3 | 0.7 | 189 |
| tx_id_4 | 0.5 | 189 |
| tx_id_5 | 0.25 | 189 |
| tx_id_6 | 0.0 | 189 |

All 1,323 signal coordinates match the metadata, and observed read counts agree with metadata. Each position has the same 7-mer sequence context across all seven identifiers. The original 45-feature extractor accepts every observation. Extracted features and unmodified fractional labels are cached in `artifacts/all_datasets/data2_features.npz`.

## Related observations and splitting

Between the 1.0 and 0.95 conditions, 110,428 of 116,180 unique read feature vectors in the 0.95 condition also occur in the 1.0 condition: 95.05%. These conditions contain extensively reused measurements.

Consequently, splitting data2 by its seven transcript IDs would put near-duplicate observations into separate splits and separate one label condition from another. The identifiers behave like condition identifiers; they do not establish seven independent biological genes. All seven versions of each original position should stay in the same outer split and CV validation fold. This gives 189 position/context groups for data2. Whether different original positions belong to the same biological transcript or gene cannot be established from these files.

A comparison against data0 and data1 selected up to 30 sorted unique read vectors per condition/position, required identical sequence context, and searched for at least three matching read vectors per natural site. It found no matching natural sites. This limited anchor check does not prove that the complete datasets share no reads or biological origins.

Data0 and data1 should retain their existing conservative gene groups. The intended training protocol remains an 80/20 outer group split and five-fold grouped cross-validation within training, followed by one held-out evaluation. Data2's source-position groups are distinct from annotated gene groups and should be described accordingly.

## Target interpretations considered

Data0 and data1 provide binary site labels. Data2 provides fractional values whose biological definition is absent from the supplied handout. Three possible treatments have different meanings:

| Treatment | Effect on data2 | Training / reporting consequence |
|---|---|---|
| Preserve fractions as soft targets | Keep all seven numeric values | Requires fractional-target cross-entropy or a regression objective; ordinary binary ROC AUC/AP cannot directly evaluate fractional labels |
| Any fraction above zero is positive | 1,134 positives, 189 negatives | Binary site classifier detecting any modification; assumes fractions mean modification abundance |
| Fractions at least 0.5 are positive | 945 positives, 378 negatives | Binary classifier for a chosen majority-modified threshold; changes the endpoint from any modification |

Converting fractions with `int(label)` would incorrectly turn values such as 0.95 and 0.75 into zero. The original binary metadata reader rejects fractional labels. The all-datasets workflow now applies the user-selected inclusive 0.5 threshold before binary training and preserves the original fractional labels for audit.

Full audit counts and source SHA-256 hashes are published in `results/all_datasets/data2_audit.json`. The earlier combined model is packaged in `models/data0_data1_model.joblib`. Feature caches are local generated artifacts.
