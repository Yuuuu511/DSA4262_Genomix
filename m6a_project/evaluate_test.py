"""Evaluate the frozen selected model on all original data0/data1 test sites."""
import argparse
from pathlib import Path
import pandas as pd
from predict import load_model
from train import extract_selected, metrics, probabilities, read_manifest


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data0', type=Path, required=True)
    parser.add_argument('--data1', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=root / 'artifacts/selected_xgb.joblib')
    parser.add_argument('--manifest', type=Path, default=root / 'splits/split_manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'evaluation/recomputed')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new evaluation output directory')
    manifest = read_manifest(args.manifest)
    bundle = load_model(args.model)
    if bundle.get('training_data') != 'data0 train+validation; RF original manifest':
        raise ValueError('Unexpected model training scope')
    if not bundle['manifest'].equals(manifest):
        raise ValueError('Model and evaluation manifest differ')
    rows = manifest.loc[manifest.split.eq('test')].reset_index(drop=True)
    if set(rows.source_dataset) != {'data0', 'data1'}:
        raise ValueError('Expected both data0 and data1 test observations')
    X = extract_selected({'data0': args.data0, 'data1': args.data1}, rows)
    p = probabilities(bundle['model'], X, bundle['positive_weight'])
    results = [{'model': 'xgb_data0_rf_split', 'dataset': 'overall', **metrics(rows.label, p)}]
    for source in ['data0', 'data1']:
        mask = rows.source_dataset.eq(source).to_numpy()
        results.append({'model': 'xgb_data0_rf_split', 'dataset': source,
                        **metrics(rows.loc[mask, 'label'], p[mask])})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / 'test_metrics.csv', index=False)
    keys = ['sample_id', 'source_dataset', 'transcript_id', 'transcript_position', 'gene_id', 'label']
    rows[keys].assign(xgb_score=p).to_csv(out / 'heldout_test_predictions.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print('Reporting only; do not tune on test results.')


if __name__ == '__main__':
    main()
