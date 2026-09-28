# DSA4262 Task 1: first-pass pipeline

This is a **starting baseline**, not a final scientific result. It accepts the
course's m6Anet-style JSON Lines (`.json` or `.json.gz`) and site-level label CSV.
Each transcript position is a sample. Its reads are summarized by the mean,
standard deviation, median and quartiles of each of 9 signal values, plus read
count, central 5-mer motif and its two flanking bases. A group split by
`gene_id` reserves 20% of genes for evaluation.

## Run on course data

```bash
python -m pip install -r requirements.txt
python train.py \
  --json data/dataset0.json.gz \
  --labels data/data.info.labelled \
  --model-out model.joblib \
  --metrics-out metrics.json

python predict.py \
  --json data/new_dataset.json.gz \
  --model model.joblib \
  --output predictions.csv
```

`--model logistic` selects a simple linear baseline. Scores are probabilities
returned by the classifier; **their calibration has not been established**.
The trained model intentionally excludes the held-out genes, so validation stays
independent. Train a final model on all labelled genes only after model selection.

## Synthetic smoke test (not biological data)

```bash
python make_example.py
python train.py --json example/example.json --labels example/example.info \
  --model-out example/model.joblib --metrics-out example/metrics.json
python predict.py --json example/example.json --model example/model.joblib \
  --output example/predictions.csv
```

Inspect `metrics.json` for ROC AUC, PR AUC, and the random PR AUC (test-set
positive fraction). The demonstration predictions on `example/example.json`
include training sites and should **not** be reported as held-out performance.

Before final submission, benchmark against m6Anet on the same gene-disjoint test
set, examine cell-line shift, document any filtering/coverage rules, and test
the prediction script on the evaluator's Ubuntu environment.

## First run on the supplied Hct116 data

The included `real_model.joblib` was trained on the 80% training genes only.
`real_metrics.json` records an independent evaluation on the other 20% of genes:
23,832 test sites, positive fraction 0.04968, ROC AUC 0.91408, PR AUC 0.49550.
All 121,838 JSON positions matched the supplied label file. The included
`real_predictions.csv` contains one score for **every** site in that same JSON.
It mixes sites used for training with held-out sites and therefore must not be
used to estimate held-out performance. It is an output-format demonstration,
not a leaderboard submission for new data.
