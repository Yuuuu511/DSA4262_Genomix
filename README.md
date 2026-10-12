# DSA4262 — Genomix

This repository contains our DSA4262 genomics project work.

## Task 1: m6A prediction

Our method uses an XGBoost classifier to predict m6A modification probabilities from m6Anet-processed direct RNA-sequencing data.

The implementation is in [`m6a_project/`](m6a_project/). It includes:

- Model training and prediction scripts.
- A pretrained data0-only XGBoost model.
- A reproducible gene-level train/validation/test split.
- A small example dataset for testing prediction.
- Evaluation results and comparisons against official m6Anet.

**For installation and commands, see [`m6a_project/README.md`](m6a_project/README.md).**

## Repository structure

```text
.
├── README.md
└── m6a_project/
    ├── README.md
    ├── requirements.txt
    ├── train.py
    ├── predict.py
    ├── xgb_features.py
    ├── artifacts/
    ├── splits/
    ├── example/
    └── evaluation/
```

Full course datasets are not included in the repository. For training and full-dataset evaluation, place the downloaded `data0/`, `data1/` and `data2/` folders beside `m6a_project/`.

## Verified environment

Saved-model prediction tests and official m6Anet benchmarking were successfully run on **AWS Ubuntu 24.04, x86_64**.

- XGBoost environment: Python 3.11.
- m6Anet environment: Python 3.8, m6Anet 2.0.2 and PyTorch 1.6.0.

Model training was reproduced on macOS. Full XGBoost retraining on AWS has not yet been verified.

## Evaluation summary

On the same 39,019 held-out data0/data1 sites:

| Method | ROC-AUC | PR-AUC | Average precision |
|---|---:|---:|---:|
| XGBoost | 0.876746 | 0.462438 | 0.462871 |
| Pretrained m6Anet | 0.878555 | 0.446030 | 0.446313 |

Data2 is evaluated separately using the label convention `label >= 0.5 → 1`, otherwise `0`. Detailed results and limitations are documented in [`m6a_project/RESULTS.md`](m6a_project/RESULTS.md).

These comparisons informed method selection and are exploratory. Pretrained m6Anet training-data overlap is unknown. A suitable simple-baseline comparison still needs to be included in the final report.

## References

Hendra et al. (2022), [Detection of m6A from direct RNA sequencing using a multiple instance learning framework](https://doi.org/10.1038/s41592-022-01666-1).

[Official m6Anet repository](https://github.com/GoekeLab/m6anet).
