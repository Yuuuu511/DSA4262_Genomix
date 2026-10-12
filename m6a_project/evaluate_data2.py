"""Evaluate data2 predictions with label >= 0.5 mapped to 1, else 0.

No training or inference is performed. Optionally compare an existing official
m6Anet prediction file on common data2 sites. Raw fractional labels are retained.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

KEYS = ['transcript_id', 'transcript_position']


def load_table(path, value_column, label=False):
    frame = pd.read_csv(path, dtype={'transcript_id': str})
    if not set(KEYS + [value_column]).issubset(frame.columns):
        raise ValueError(f'{path}: requires {KEYS + [value_column]}')
    frame = frame[KEYS + [value_column]].copy()
    if frame.isna().any().any() or frame.transcript_id.str.strip().eq('').any():
        raise ValueError(f'{path}: missing values or empty identifiers')
    positions = pd.to_numeric(frame.transcript_position, errors='raise')
    if not np.isfinite(positions).all() or (positions % 1 != 0).any():
        raise ValueError(f'{path}: positions must be finite integers')
    frame['transcript_position'] = positions.astype(np.int64)
    if frame.duplicated(KEYS).any():
        raise ValueError(f'{path}: duplicate sites')
    values = pd.to_numeric(frame[value_column], errors='raise')
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        kind = 'labels' if label else 'probability scores'
        raise ValueError(f'{path}: {kind} must be finite and in [0, 1]')
    frame[value_column] = values
    if frame.empty:
        raise ValueError(f'{path}: empty table')
    return frame


def metrics(name, dataset, y, scores):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f'{dataset}: both binary classes required for ROC/PR evaluation')
    precision, recall, _ = precision_recall_curve(y, scores)
    return {'model': name, 'dataset': dataset, 'n_sites': len(y),
            'positive_rate': float(np.mean(y)), 'roc_auc': float(roc_auc_score(y, scores)),
            'pr_auc': float(auc(recall, precision)),
            'average_precision': float(average_precision_score(y, scores))}


def evaluate(labels_path, predictions_path, output_dir, m6anet_path=None):
    out = Path(output_dir)
    if out.exists() and any(p.name != '.DS_Store' for p in out.iterdir()):
        raise FileExistsError('Data2 evaluation output directory is nonempty')
    labels = load_table(labels_path, 'label', label=True).rename(columns={'label': 'label_original'})
    labels['label_binary'] = (labels.label_original >= .5).astype(int)
    pred = load_table(predictions_path, 'score').rename(columns={'score': 'xgb_score'})
    aligned = labels.merge(pred, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all():
        raise ValueError('XGBoost predictions must cover exactly all data2 label sites')
    aligned = aligned.drop(columns='_merge')
    results = [metrics('xgboost', 'data2_all', aligned.label_binary, aligned.xgb_score)]
    report = {'label_rule': 'label >= 0.5 -> 1; label < 0.5 -> 0',
              'threshold': .5, 'n_sites': len(aligned),
              'n_original_labels_equal_0_5': int(labels.label_original.eq(.5).sum()),
              'positive_sites': int(labels.label_binary.sum()),
              'negative_sites': int(labels.label_binary.eq(0).sum()),
              'notes': ['Data2 is not used for training or tuning.',
                        'Metrics use thresholded fractional labels, unlike native data0/data1 binary labels.',
                        'Prediction scores remain probabilities; no score threshold is applied.']}
    missing = None
    if m6anet_path:
        m6anet = load_table(m6anet_path, 'probability_modified').rename(columns={'probability_modified': 'm6anet_score'})
        comparison = aligned.merge(m6anet, on=KEYS, how='outer', validate='one_to_one', indicator=True)
        if comparison['_merge'].eq('right_only').any():
            raise ValueError('m6Anet returned sites outside data2 labels')
        missing = comparison.loc[comparison['_merge'].eq('left_only'), KEYS].copy()
        common = comparison.loc[comparison['_merge'].eq('both')].copy()
        if common.empty:
            raise ValueError('No shared data2 sites for m6Anet comparison')
        for name, column in [('xgboost', 'xgb_score'), ('m6anet_pretrained', 'm6anet_score')]:
            results.append(metrics(name, 'data2_common_m6anet', common.label_binary.astype(int), common[column]))
        report['m6anet_common_sites'] = len(common)
        report['m6anet_missing_sites'] = len(missing)
        report['notes'].append('Pretrained m6Anet training overlap is unknown; compare on shared sites.')
        aligned = aligned.merge(m6anet, on=KEYS, how='left', validate='one_to_one')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / 'data2_metrics.csv', index=False)
    aligned.to_csv(out / 'data2_evaluated_predictions.csv', index=False)
    (out / 'data2_evaluation_report.json').write_text(json.dumps(report, indent=2))
    if missing is not None:
        missing.to_csv(out / 'missing_m6anet_sites.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(f'Original labels exactly 0.5: {report["n_original_labels_equal_0_5"]}; assigned to class 1.')
    print(f'Saved {out.resolve()}')
    return results, report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=root.parent / 'data2/data.info')
    parser.add_argument('--predictions', type=Path, default=root / 'submissions/Genomix_haribo_dataset2.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'evaluation/data2_evaluation')
    parser.add_argument('--m6anet-predictions', type=Path,
                        help='Optional data2 data.site_proba.csv; no m6Anet inference is run here')
    args = parser.parse_args()
    evaluate(args.labels, args.predictions, args.output_dir, args.m6anet_predictions)


if __name__ == '__main__':
    main()
