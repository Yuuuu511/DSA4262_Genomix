# Original data0-only baseline

Historical training report. The packaged models are `models/data0_model.joblib` (all data0) and `models/data0_evaluation_model.joblib` (80% training genes). Numeric results and original audit checks are in `results/data0/`.

## Original data0-only workflow and what each step does

1. **Read the labels and signals.** The label CSV identifies a site by `(transcript_id, transcript_position)` and gives its gene and binary m6A label. The gzipped JSON-lines file contains a variable number of reads for each site. Every read has nine values: dwell time, current standard deviation, and mean current at the upstream, central, and downstream positions. We stream the file one record at a time, match the coordinates to labels, and reject missing labels, duplicate sites, missing signals, and invalid features. The source datasets are read without changing them.

2. **Turn each site into a fixed feature vector.** Logistic regression requires a fixed number of inputs. For each of the nine read-level measurements, calculate the mean and population standard deviation across reads (18 features). Add `log(1 + number of reads)` (one feature). Add one-hot indicators for the central DRACH 5-mer (18 possible motifs) and each outer flanking base (four possible bases per flank; eight features). Total: **45 features per site**. The across-read standard deviation is distinct from the current standard deviation already present in each read. Summaries and sequence encoding use no labels. IDs and transcript positions are used for matching and splitting, not model inputs.

3. **Split by gene before fitting anything.** Use a fixed random seed of 42 to put approximately 80% of unique genes in training and 20% in a held-out test. All sites from a gene stay in the same split. Splitting individual sites at random would allow related transcripts/sites from the same gene into both sets and could overstate generalisation. The percentages refer to genes, so site counts need not be exactly 80/20. Save every site's split in `artifacts/split.csv`. Verify that no genes or transcripts overlap. This is one fixed split, not cross-validation.

4. **Fit the scaler and train the model on training sites.** `StandardScaler` learns the training means and standard deviations and puts all features on comparable scales. Logistic regression learns weights and an intercept to minimise binary log loss with L2 regularisation (`C=1`). Its prediction is `sigmoid(intercept + weighted sum of scaled features)`. Positive weights increase the predicted probability; negative weights decrease it. The scaler and classifier are saved as one pipeline so inference applies the same transformation. Training stops with an error if the optimiser does not converge.

5. **Test once on unseen genes.** Transform test features using the training scaler, then obtain `predict_proba(...)[..., 1]`. Compare probabilities against test labels using ROC AUC and precision-recall metrics. Also evaluate a constant predictor that assigns the training positive fraction to every test site. No hyperparameters, threshold, or feature choices are selected using this held-out test. A separate validation split or gene-grouped cross-validation should be added before tuning future models; repeated tuning against this test would invalidate its independence.

6. **Retrain the fixed specification on all of data0.** After evaluation, fit a fresh scaler and logistic regression on all labelled sites. This uses all available training data for the intermediate submission. Save it as `artifacts/baseline_model.joblib`. The reported held-out metrics describe `evaluation_model.joblib`, which never saw test genes during fitting; they are not an independent evaluation of the refitted submission model.

7. **Predict the released leaderboard input.** `predict.py` reads only the new signal JSON, creates the same 45 features, loads the saved full-data0 model when explicitly selected with `--model models/data0_model.joblib`, and writes one probability per site in batches. Its current default uses the all-datasets model. The output has exactly `transcript_id,transcript_position,score`. It rejects duplicate sites and invalid probabilities, and replaces the output only after success. No labels are needed. Submit probabilities, not thresholded classes.

## Interpreting the evaluation

- **ROC AUC:** measures ranking across thresholds. A non-informative ranking has expected ROC AUC 0.5; a perfect ranking has 1.0.
- **Average precision (AP):** a non-interpolated summary of the precision-recall curve. The constant predictor's AP equals test prevalence. This is useful when positive labels are rare.
- **Trapezoidal PR AUC:** integrates the precision-recall curve with the trapezoidal rule. This differs from AP, so both are reported rather than assuming which definition the leaderboard uses. For a constant predictor, its trapezoidal PR AUC is not equal to prevalence because of interpolation to the endpoint.
- **Log loss and Brier score:** measure probability error; lower is better. These are additional diagnostics, not stated leaderboard metrics.

We keep the natural class frequencies and use no class weighting or oversampling. This avoids deliberately changing the class prior in a model whose output is intended as a probability. Imbalance is addressed here through appropriate evaluation, not by claiming it has been solved. Logistic output is not a guarantee of calibration on a different cell line or sequencing protocol. Accuracy and a default 0.5 threshold would be unhelpful primary measures for these rare positives.

Official references: [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), and [gene-group splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

## Measured results from the initial run

Training used **98,006 sites from 3,081 genes** (4,291 positive sites; 4.38%). The held-out test contained **23,832 sites from 771 genes** (1,184 positive sites; 4.97%). No genes or transcripts overlapped.

| Held-out metric | Logistic baseline | Constant probability predictor |
|---|---:|---:|
| ROC AUC | 0.902878 | 0.500000 |
| Average precision | 0.499075 | 0.049681 |
| Log loss | 0.124342 | 0.197974 |
| Brier score | 0.032791 | 0.047248 |

The logistic baseline's trapezoidal PR AUC was **0.498388**. ROC AUC is a ranking measure, not classification accuracy. These results evaluate unseen genes within data0; leaderboard performance has not been measured.

All four automated tests passed. Inference using the saved evaluation model on the twenty-site real-data sample reproduced the corresponding held-out probabilities to within `4.45e-16`. The evaluation scaler was confirmed to have seen only the 98,006 training sites; the submission scaler was confirmed to have seen all 121,838 sites. Results and checks are saved in `results/data0/metrics.json` and `results/data0/verification.json`.

## Limits of this first baseline

Mean and standard deviation lose information about individual reads and mixtures of modified and unmodified molecules. Logistic regression models additive effects on the log odds and cannot learn arbitrary interactions. Gene-separated evaluation tests new genes within data0, not generalisation to other cell lines or sequencing conditions. Site labels do not identify which reads are modified. The model outputs a site probability, not a per-read modification rate. Read depth could encode expression or experiment effects, so later models should check performance across coverage levels and with that feature removed. This baseline has not yet been benchmarked against m6Anet.

## Reproduction

```bash
python train.py --data data/data0/dataset0.json.gz \
  --labels data/data0/data.info.labelled --out artifacts
```

Training regenerates the historical `artifacts/` paths described above; generated caches and large diagnostic CSVs are ignored by Git. See the repository README for installation.
