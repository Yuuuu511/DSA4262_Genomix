# Inference example

`test_data.json.gz` contains 20 real data0 sites, ten from each supplied class. Labels are not included or needed. This is an inference check, not an independent evaluation set.

`expected_predictions.csv` contains probabilities from `models/baseline_model.joblib`:

```bash
python predict.py --data example/test_data.json.gz \
  --output outputs/example_predictions.csv
```

The package test checks coordinates, the three-column submission schema, finite probabilities and agreement within numerical tolerance.
