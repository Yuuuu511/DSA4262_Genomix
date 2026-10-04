# HGB training with multiple datasets

Replace your original train.py with the train.py in this package. The original
features.py and predict.py remain compatible. This package also includes them
for convenience; it contains no real data or trained model.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place your folders next to train.py:

```text
train.py
features.py
predict.py
data/dataset0.json.gz
data/data.info.labelled
data1/dataset1.json.gz
data1/data.info.labelled
```

Train with both datasets:

```bash
python train.py \
  --json data/dataset0.json.gz data1/dataset1.json.gz \
  --labels data/data.info.labelled data1/data.info.labelled \
  --split-by transcript \
  --model hgb \
  --model-out model_combined.joblib \
  --metrics-out metrics_combined.json
```

JSON paths and label paths are paired by their order. Use your actual filenames.
One dataset is still supported using the original command syntax.

Each JSON is matched only with its own label file. Matched observations are then
concatenated before one grouped 80/20 split (seed 42). The command above groups
by transcript_id, because dataset1 labels lack gene_id. This keeps all sites
and datasets of one transcript together, but different transcripts of the same
gene can still appear on both sides: these metrics are NOT gene-level metrics. Repeated sites across
datasets are retained as separate observations, and all observations of a gene
remain on the same side when --split-by gene is used. Conflicting
transcript-to-gene mappings raise an error.
Both labels 0 and 1 must be present in the combined training and test partitions;
soft labels are not supported. Extra n_reads/start/end columns are ignored.
Read counts are obtained from the actual JSON reads, not from label metadata.

For strict gene-level evaluation, use --split-by gene (the default). Mapping
is collected across all input label files that contain gene_id, and can also
be supplied via --gene-map transcript_gene.csv. That CSV must have columns
transcript_id,gene_id and cover every matched transcript. Missing mappings
raise an error rather than silently dropping sites or changing split mode.

The features and classifier parameters are unchanged from the starter HGB.
Source identifiers are used for reporting only, not as model features.
metrics_combined.json includes matched counts, repeated-site counts, combined
test metrics and metrics for each dataset's held-out observations. PR AUC here
means sklearn average_precision_score, as in the original script. Repeated sites
contribute multiple observations to the combined metric; inspect per-dataset
results too. Exact copied datasets should not be intentionally included twice.

Adding groups may change the held-out set even with the same seed. Do not
interpret the new score versus the old 0.495 as an improvement measured on a
fixed test set. The saved model is fitted on training groups only. In transcript mode,
train_genes/test_genes count only known mapped genes and can overlap;
train_groups/test_groups are the actual split counts.

Predict with the existing prediction script:

```bash
python predict.py \
  --json data1/dataset1.json.gz \
  --model model_combined.joblib \
  --output predictions_data1.csv
```

Predictions on the input training datasets include training observations and
cannot be used as independent test scores.
