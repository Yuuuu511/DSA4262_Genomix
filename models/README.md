# Published models

Each file contains a fitted scikit-learn pipeline (`StandardScaler` and logistic regression), feature specification, library version and training provenance.

| File | Training scope | C |
|---|---|---:|
| `baseline_model.joblib` | Current data0 + data1 + data2 training portion; 169,525 observations; data2 label ≥ 0.5 | 10 |
| `data0_data1_model.joblib` | Data0 + data1 training portion; 168,802 observations | 10 |
| `data0_model.joblib` | Original submission baseline refitted on all 121,838 data0 sites | 1 |
| `data0_evaluation_model.joblib` | Original data0 training portion; 98,006 observations | 1 |

Combined models exclude their held-out test rows. The full-data0 model is a separate refit; its predecessor's test score is not an independent evaluation of that refit.

The latest model is the default for `predict.py`. Select an earlier file with `--model` explicitly. Install pinned requirements before loading. Only load joblib files from sources you trust.

`manifest.json` records distributed and original checksums, versions and scopes. Combined-model provenance paths were made relative for publication. Fitted scaler statistics, coefficients, intercepts and classes exactly match the local originals. Historic `reference/raw/` paths document omitted raw annotations; their URLs and checksums are in `reference/provenance.json`.

New training runs save under ignored `artifacts/`. Updating a published model also requires refreshing its manifest, results and example predictions.
