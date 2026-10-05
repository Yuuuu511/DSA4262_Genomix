# DSA4262_Genomix
DSA4262 Team Project - Prediction of m6A RNA modifications from direct RNA-Seq data

# m6A site prediction from direct RNA sequencing

A site-level gradient boosting model using mean, SD, quantiles and IQR of nine read features, plus seven-base sequence encoding. Dwell times use log1p. IDs, coordinates, labels, read depth and file offsets are not predictors. Input is streamed; training keeps the much smaller feature matrix in memory. Prediction uses batches of 1,024 sites.

## Install on AWS Ubuntu

Python 3.11 recommended (this package was tested with the Python version in REPORT.md). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Only load trusted joblib model files. Package versions should match those used for training.

## dataset 
Dataset0 and dataset1 were used combined and used to train the model in a 60/20/20 split for training, validation and testing respectively.

## Run predictions

```bash
python predict.py --data examples/data.json --model results/model.joblib --output predictions.csv
```

Input accepts newline-delimited JSON or gzip-compressed JSON. Output is exactly `transcript_id,transcript_position,score`, one row per input site, with scores in [0,1]. The small example contains ten real sites from dataset0 and is for execution testing, not accuracy estimation. `examples/expected_predictions.csv` gives expected output from the included model.

## Train a new model

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python train.py \
  --data 'raw_data/dataset0.json(1).gz' 'raw_data/dataset1.json(1).gz' \
  --labels 'raw_data/data.info(1).labelled' raw_data/data1.info \
  --out-dir results --seed 42
```

Each JSON file uses its own paired labels. Join by exact transcript ID and position. Gene mappings available in either label file apply to both datasets. Unmapped transcripts form their own groups. Shared sites keep replicate-specific labels, including disagreements. Start/end offsets from dataset1 must not be applied to dataset0 or compressed files.

Five stratified group folds produce approximately 60% training, 20% validation and 20% testing by assigning three folds to training and one each to validation/test. Proportions are approximate because genes remain intact. Transcript and known gene groups never cross partitions; unknown gene relationships cannot be excluded. Three hyperparameter choices are selected by validation average precision. Test data never choose model parameters. The delivered model is fit only to training rows; validation and test are not used for a final refit.

Outputs: `model.joblib`, `metrics.json`, `site_predictions.csv` with source, labels, groups, read counts and split. Labels with no signal are excluded and counted. Signal sites without labels are excluded from training but prediction scores all sites. Invalid matrices or duplicate matched sites fail explicitly.

## m6Anet comparison

Use a separate Python 3.8 conda environment, per the assignment, and install through conda, not pip:

```bash
conda create -n m6anet38 -c conda-forge -c bioconda python=3.8 m6anet=2.1.0
conda run -n m6anet38 m6anet inference --help
```

If solver availability changes, document the resolved m6Anet version and package build. In the custom model environment, regenerate byte offsets on UNCOMPRESSED held-out JSON:

```bash
python benchmark.py prepare --data 'raw_data/dataset0.json(1).gz' \
  --manifest results/site_predictions.csv --dataset 0 --out-dir benchmark_inputs/dataset0
conda run -n m6anet38 m6anet inference \
  --input_dir benchmark_inputs/dataset0 --out_dir benchmark_outputs/dataset0 \
  --n_processes 2 --num_iterations 1000
python benchmark.py compare --manifest results/site_predictions.csv --dataset 0 \
  --m6anet benchmark_outputs/dataset0/data.site_proba.csv --output comparison0.json
```

Repeat with dataset1 and `--dataset 1`. Compare identical eligible site intersections and disclose exclusions, including m6Anet coverage/motif filters. Noisy-OR multiple-instance learning and read sampling differ from our deterministic site summaries. The default human m6Anet model was trained on HCT116; its overlap with supplied sites is unknown. This is an external pretrained-model comparison, not a leakage-free comparison of learning algorithms. A stronger comparison retrains m6Anet using exactly our train/validation groups and evaluates the untouched test genes. Record runtime, peak RAM, versions and repeat sampling seeds in that experiment.

## results
The ROC-AUC was 0.874 and average precision was 0.414.
