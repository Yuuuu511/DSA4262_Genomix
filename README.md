# DSA4262_Genomix
DSA4262 Team Project - Prediction of m6A RNA modifications from direct RNA-Seq data

## Task 1: Random Forest and Logistic Regression

This branch contains site-level feature extraction, a Logistic Regression
baseline, and Random Forest training and evaluation.

### Installation

```bash
python -m pip install -r requirements.txt
```

### Data layout

Place the provided datasets next to the scripts:

```text
merge_datasets.py
train_model.py
requirements.txt
data0/
    dataset0.json.gz
    data.info.labelled
data1/
    dataset1.json.gz
    data.info
data2/
    dataset2.json.gz
    data.info
```

Raw datasets and trained model files are not included in this branch.

### Run

Run feature extraction first, then model training:

```bash
python merge_datasets.py
python train_model.py
```

Both scripts use their own directory as the base directory.

The merge script creates `merged/merged_site_features.csv`.
The training script saves models, metrics, plots, and split assignments
to `model_outputs/`. Selected evaluation results are stored in `results/`.

### Features and labels

Each site is represented by 28 features: read count, plus the mean,
standard deviation, and median of each of the nine read-level signal features.

The merge script preserves the original labels. For data2, labels greater
than or equal to 0.5 are converted to 1; smaller values are converted to 0.
This threshold is a preprocessing choice, not a specified project requirement.

Repeated sites from different datasets are retained as separate observations.
Dataset identifiers and original labels are not used as model features.

### Training and evaluation

The current experiment uses data0 and data1 observations with known gene IDs.
It excludes 15,137 data1 observations without gene mappings and all 1,323
data2 observations.

Genes are divided approximately 60%/20%/20% into training, validation,
and test partitions. Genes and transcript-position pairs do not overlap
between partitions.

Two Random Forest configurations are compared on the validation set.
The selected configuration uses 300 trees and `min_samples_leaf=5`.
The baseline and selected Random Forest are then fitted on training
and validation data before evaluation on the held-out test set.

| Model | Test ROC-AUC | Test PR-AUC | Test AP |
|---|---:|---:|---:|
| Logistic Regression | 0.7748 | 0.1882 | 0.1885 |
| Random Forest | 0.8577 | 0.4252 | 0.4255 |

PR-AUC uses trapezoidal integration of the precision-recall curve.
Average Precision (AP) is reported separately.

Combined metrics count repeated sites from different datasets as separate
observations. Per-dataset metrics are also available.

### Current status

This is an intermediate training and evaluation workflow.
The m6Anet comparison and a standalone prediction script are pending.

The saved evaluation models are fitted on training and validation data only;
the held-out test data are excluded from fitting.
