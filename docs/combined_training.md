# Combined data0 + data1 training with cross-validation

Run: 2026-10-05T20:54:28+08:00. Model: standardised, L2-regularised logistic regression with the original 45 features. Training uses both supplied datasets. All source files remain unchanged. Data2 was not used.

## Workflow

1. **Combine site observations.** Read data0 and data1 signals and their respective binary labels. Keep dataset identity in audit files so repeated coordinates remain distinguishable. Dataset identity, transcript IDs, positions, labels and gene IDs are not input features.
2. **Resolve gene groups.** Use supplied data0 gene IDs and external annotation to map data1 transcripts. Conservatively join alternate gene identifiers linked by a transcript mapping or the same Ensembl gene name. This also keeps patch-contig versions together.
3. **Reserve the test set first.** Split 80%/20% of gene groups with seed 42. All observations belonging to a group stay in the same outer split. This yields approximately 80%/20% of observations; exact site percentages depend on gene sizes. The actual training fraction is 79.38%.
4. **Run five-fold validation inside training.** Divide only the outer training observations into five stratified group folds with seed 43. In each iteration, use four folds to fit a fresh scaler and logistic regression, and predict the fifth. Stratification seeks similar positive fractions while preserving groups. Each outer training observation is validated exactly once per candidate C. The outer test set is absent from every fold and parameter comparison.
5. **Select regularisation using validation scores.** Compare C = 0.01, 0.1, 1 and 10, using mean validation average precision. Smaller C means stronger regularisation. Use the same folds for every candidate. Choose the highest mean AP, with the smaller C winning an exact tie.
6. **Refit on the complete 80% training portion.** Train a fresh scaler and classifier using the selected C = 10. Save `artifacts/combined/baseline_model.joblib`.
7. **Test once on the reserved 20%.** Apply the saved pipeline to the test features. The saved model is fitted on the training portion only. Test observations have not been included in a subsequent full-data refit.

## Data and split counts

| Partition | Observations | Positives | Positive fraction | Gene groups | data0 observations | data1 observations |
|---|---:|---:|---:|---:|---:|---:|
| all | 212,648 | 12,068 | 5.68% | 4,379 | 121,838 | 90,810 |
| train | 168,802 | 9,370 | 5.55% | 3,503 | 96,462 | 72,340 |
| test | 43,846 | 2,698 | 6.15% | 876 | 25,376 | 18,470 |

There are **145,328 unique transcript-position coordinates** across **212,648 dataset observations**. The combined count is 121,838 + 90,810. No observations were discarded. The test set contains 30,190 distinct coordinates.

The datasets share **67,320 coordinates**, including **3,872 coordinates with different labels**. Each dataset's signal summary and supplied label is retained as a separate observation; labels are not averaged or replaced. Both occurrences always share the same outer split and validation fold. Consequently, a coordinate observed twice contributes twice to the objective and aggregate metrics. Dataset observations sharing a gene are correlated; these counts should not be interpreted as independent biological samples.

## Gene mapping and reference sources

All **6,352 transcripts** were mapped, with zero unresolved IDs:

- 5,333 transcripts use their supplied data0 gene IDs.
- 964 additional transcripts use the [SG-NEx Ensembl 91 transcript annotation](https://sg-nex-data.s3.amazonaws.com/data/annotations/txt_file/Homo_sapiens.GRCh38.91.annotations-transcripts.txt).
- 54 additional transcripts use a saved [Ensembl REST lookup](https://rest.ensembl.org/documentation/info/lookup_post) response, obtained for public transcript IDs on 5 October 2026.
- One retired transcript, ENST00000627066, is mapped through the [Ensembl 96 annotation including patch contigs](https://ftp.ensembl.org/pub/release-96/gtf/homo_sapiens/Homo_sapiens.GRCh38.96.chr_patch_hapl_scaff.gtf.gz). Its record appears on `CHR_HG2217_PATCH` with gene ID `ENSG00000281702` and gene name `HMBS`.

The supplied and SG-NEx mappings agree on all 5,185 transcripts present in both. The SG-NEx table has a header/data column mismatch, so its first transcript/gene fields were identified by ENST/ENSG prefixes rather than header names. Multiple reference versions can use different gene identifiers; linked identifiers and genes sharing an Ensembl gene name are conservatively merged for splitting. There are 4,486 preferred gene IDs and 4,379 resulting split groups. Name-based merging can join distinct loci with a shared name, making these conservative groups rather than a claim that every group is exactly one current annotated gene. Grouping uses identifiers and annotation, not modification labels.

The exact mapping is bundled in `reference/transcript_gene_map.csv`, with the saved lookup in `reference/ensembl_lookup.json`. `reference/provenance.json` records the source URLs and SHA-256 hashes. The large raw annotation snapshots remain local; they are not required by the default training command. The older Ensembl 96 GTF without patch contigs and the attempted BioMart HTML response were not used for the final mapping.

## Cross-validation results

These are means ± sample standard deviations across five validation folds, not confidence intervals.

| C | Average precision, mean ± SD | ROC AUC, mean ± SD |
|---:|---:|---:|
| 0.01 | 0.358878 ± 0.017995 | 0.856788 ± 0.003950 |
| 0.1 | 0.382623 ± 0.017935 | 0.861759 ± 0.004362 |
| 1 | 0.385004 ± 0.017812 | 0.861750 ± 0.004596 |
| 10 | 0.385143 ± 0.017793 | 0.861698 ± 0.004631 |

The pre-specified selection rule chose **C = 10**. Its mean AP exceeds C = 1 by only 0.000139, which is small relative to variation between folds. This does not establish a meaningful performance advantage of C = 10 over C = 1. The fold averages used for selection may be optimistic because the same folds were used to compare candidates; the outer test provides a separate evaluation for this run.

| Validation fold | Fitting observations | Validation observations | Validation positives | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| 1 | 135,058 | 33,744 | 1,865 | 0.861940 | 0.392972 |
| 2 | 135,010 | 33,792 | 1,891 | 0.866504 | 0.402799 |
| 3 | 135,049 | 33,753 | 1,870 | 0.854127 | 0.355515 |
| 4 | 135,058 | 33,744 | 1,866 | 0.861929 | 0.389068 |
| 5 | 135,033 | 33,769 | 1,878 | 0.863991 | 0.385363 |

Out-of-fold probabilities for the selected C are saved in `artifacts/combined/oof_predictions.csv`. Each row was predicted by a model that excluded its gene group during fitting. They are validation predictions from several models; pooled OOF metrics can differ from the average fold score.

## Held-out test results

| Metric | Combined trained model | Constant probability predictor |
|---|---:|---:|
| roc_auc | 0.848127 | 0.500000 |
| average_precision | 0.393503 | 0.061534 |
| log_loss | 0.171422 | 0.231501 |
| brier_score | 0.045469 | 0.057783 |

The model's trapezoidal PR AUC is **0.393139**. The constant predictor assigns the training positive fraction (0.055509) to every test observation. Its average precision equals test prevalence (0.061534). AP and trapezoidal PR AUC have different integration conventions; the latter's constant-score endpoint interpolation makes it unsuitable as a prevalence reference. Lower log loss and Brier score indicate better probability estimates. No classification threshold was selected.

Results separated by the source of test observations:

| Test source | Observations | ROC AUC | Average precision | Trapezoidal PR AUC |
|---|---:|---:|---:|---:|
| data0 | 25,376 | 0.890080 | 0.444016 | 0.443241 |
| data1 | 18,470 | 0.806047 | 0.360745 | 0.360102 |

These results evaluate gene groups withheld from the new combined model. Earlier project runs already examined data0 and data1, so this is a test held out for this training run, not a completely untouched project-level dataset. Changes in the training size, test membership, grouping and prevalence make direct comparisons against earlier scores imperfect. The hidden leaderboard and other cell lines remain unevaluated. Site-level summaries discard information about individual reads; this remains an additive logistic baseline.

## Verification

Six automated tests passed. A separate artifact audit confirmed:

- No gene groups, transcript IDs, or coordinates overlap across the outer train/test split.
- Shared coordinates across datasets have identical outer split and validation-fold assignments.
- All five folds exclude their validation groups from fitting, and every training observation has one OOF prediction.
- The saved scaler's fitted row count is 168,802 and its means equal the means of the outer training features only.
- The saved model's recorded training/test gene groups are disjoint.
- Test scores and each fold's AP recomputed from the exported CSVs match the reported results.
- The ordinary prediction script reproduces probabilities for 40 real held-out observations sampled across data0 and data1 to within numerical precision.

Historical numeric results and audit checks are published in `results/data0_data1/`.

## Reproduction and use

From the project directory, activate the existing environment and run:

```bash
source .venv/bin/activate
python train_combined.py \
  --data0 data/data0/dataset0.json.gz \
  --labels0 data/data0/data.info.labelled \
  --data1 data/data1/dataset1.json.gz \
  --labels1 data/data1/data.info
```

The default loads the bundled compact mapping and checks its checksum and the two label-file hashes. Install `requirements.txt` and supply the datasets under `data/` or adjust their paths. Five folds and four C values are the defaults. To rebuild the mapping from the original large reference files, supply `--annotation`, `--lookup` and `--patch-gtf` together; see `reference/README.md`. Source hashes and the feature version are checked before a feature cache is reused.

To make leaderboard-format predictions with the combined model:

```bash
python predict.py --data PATH_TO_LEADERBOARD_INPUT.json.gz \
  --model models/data0_data1_model.joblib \
  --output artifacts/combined/intermediate_predictions.csv
```

Replace the input placeholder with the released file's path. The inference script writes the required three probability columns. The diagnostic test and OOF CSVs contain extra metadata and repeated coordinates from different datasets; they are not leaderboard submission files.

| Output | Purpose |
|---|---|
| `artifacts/combined/baseline_model.joblib` | Selected model fitted on the 80% training portion |
| `artifacts/combined/metrics.json` | CV candidate/fold scores, test scores, versions and provenance |
| `artifacts/combined/split.csv` | Outer split and validation-fold assignment for every observation |
| `artifacts/combined/oof_predictions.csv` | Selected-C validation predictions within the training portion |
| `artifacts/combined/test_predictions.csv` | Predictions and true labels for the reserved test portion |
| `artifacts/combined/transcript_gene_map.csv` | Gene IDs, split groups and mapping sources |
| `artifacts/combined/verification.json` | Artifact audit checks |

The packaged combined model is `models/data0_data1_model.joblib`; specify it explicitly when predicting. The default prediction model now uses all three datasets. The table above describes regenerated local training outputs; the GitHub repository includes the model and result summaries, while split, feature and prediction files are regenerated locally. Algorithm references: [grouped stratified folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) and [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
