"""Train data0-only XGBoost from raw JSON and labels on a fixed gene manifest.

Four candidates are selected using data0 validation only. Final fit uses data0
train+validation. Original test genes are excluded. Default manifest is bundled.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier
from xgb_features import FEATURE_NAMES, VERSION, iter_sites, site_features

KEYS = ['source_dataset', 'transcript_id', 'transcript_position']
CANDIDATES = [(3, 1), (5, 1), (3, 4), (5, 4)]


def feature_hash():
    return hashlib.sha256(Path(__file__).with_name('xgb_features.py').read_bytes()).hexdigest()


def read_manifest(path):
    frame = pd.read_csv(path, dtype={'sample_id': str, 'source_dataset': str,
                                   'transcript_id': str, 'gene_id': str})
    required = ['sample_id'] + KEYS + ['gene_id', 'label', 'split']
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Manifest requires {required}')
    if frame[required].isna().any().any():
        raise ValueError('Missing manifest values')
    if frame.sample_id.duplicated().any() or frame.duplicated(KEYS).any():
        raise ValueError('Duplicate sample or source/site identifiers')
    if set(frame.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected unchanged train/validation/test split')
    if not frame.source_dataset.isin(['data0', 'data1']).all():
        raise ValueError('Expected data0/data1 only')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('Binary labels required')
    if frame.groupby('gene_id').split.nunique().max() > 1:
        raise ValueError('A gene occurs in multiple splits')
    if frame.groupby(['transcript_id', 'transcript_position']).split.nunique().max() > 1:
        raise ValueError('A site occurs in multiple splits')
    positions = pd.to_numeric(frame.transcript_position, errors='raise')
    if not np.isfinite(positions).all() or (positions % 1 != 0).any():
        raise ValueError('Positions must be finite integers')
    frame['transcript_position'] = positions.astype(np.int64)
    return frame


def extract_selected(paths, rows):
    """Compute features only for requested rows, keeping manifest order."""
    wanted = {key: i for i, key in enumerate(rows[KEYS].itertuples(index=False, name=None))}
    if len(wanted) != len(rows):
        raise ValueError('Duplicate requested sites')
    X = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    seen = set()
    for source in ['data0', 'data1']:
        if not rows.source_dataset.eq(source).any():
            continue
        path = Path(paths[source])
        print(f'Extracting selected XGBoost features from {source}...', flush=True)
        for count, (transcript, position, sequence, reads) in enumerate(iter_sites(path), 1):
            key = (source, transcript, position)
            if key in wanted:
                if key in seen:
                    raise ValueError(f'Duplicate raw site: {key}')
                X[wanted[key]] = site_features(sequence, reads)
                seen.add(key)
            if count % 20000 == 0:
                print(f'  Scanned {count:,} sites', flush=True)
    if seen != set(wanted):
        raise ValueError(f'Missing raw features for {len(set(wanted) - seen)} sites')
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite feature values')
    return X


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes required for evaluation')
    if len(p) != len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probability scores')
    precision, recall, _ = precision_recall_curve(y, p)
    return {'n_sites': len(y), 'positive_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, p)), 'pr_auc': float(auc(recall, precision)),
            'average_precision': float(average_precision_score(y, p))}


def probabilities(model, X, weight):
    p = model.predict_proba(X)[:, 1].astype(float)
    # Same analytic class-weight correction as the original XGBoost method.
    return p / (weight - (weight - 1) * p)


def model_parameters(depth, weight, args):
    return dict(max_depth=depth, scale_pos_weight=weight, learning_rate=.05,
                n_estimators=args.max_rounds, min_child_weight=15, subsample=.85,
                colsample_bytree=.85, reg_lambda=10, reg_alpha=.1, tree_method='hist',
                n_jobs=args.n_jobs, random_state=4262, eval_metric='aucpr',
                early_stopping_rounds=args.early_stopping_rounds)


def validate_labels(path, manifest):
    labels = pd.read_csv(path, dtype={'transcript_id': str, 'gene_id': str})
    keys = ['transcript_id', 'transcript_position']
    required = keys + ['gene_id', 'label']
    if not set(required).issubset(labels.columns) or labels[required].isna().any().any():
        raise ValueError('Training labels require transcript_id, transcript_position, gene_id, label')
    if labels.duplicated(keys).any() or not labels.label.isin([0, 1]).all():
        raise ValueError('Training labels must be unique and binary')
    expected = manifest.loc[manifest.source_dataset.eq('data0'), required]
    matched = expected.merge(labels[required], on=keys, how='left',
                             validate='one_to_one', suffixes=('', '_input'), indicator=True)
    if not matched['_merge'].eq('both').all():
        raise ValueError('Labels missing data0 manifest sites')
    for col in ['gene_id', 'label']:
        if not matched[col].eq(matched[col + '_input']).all():
            raise ValueError(f'Input labels disagree with manifest on {col}')


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Raw data0 .json or .json.gz')
    parser.add_argument('--labels', type=Path, required=True, help='data0 binary labels CSV')
    parser.add_argument('--manifest', type=Path, default=root / 'splits/split_manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'artifacts/retrained')
    parser.add_argument('--max-rounds', type=int, default=1200)
    parser.add_argument('--early-stopping-rounds', type=int, default=60)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()
    if min(args.max_rounds, args.early_stopping_rounds, args.n_jobs) < 1:
        parser.error('Round and worker counts must be positive')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output folder is nonempty; use a new folder')
    manifest = read_manifest(args.manifest)
    validate_labels(args.labels, manifest)
    rows = manifest.loc[manifest.source_dataset.eq('data0') &
                        manifest.split.isin(['train', 'validation'])].reset_index(drop=True)
    for split in ['train', 'validation']:
        if rows.loc[rows.split.eq(split), 'label'].nunique() != 2:
            raise ValueError(f'data0 {split} requires both classes')
    print(rows.groupby(['source_dataset', 'split']).size().to_string(), flush=True)
    X = extract_selected({'data0': args.data}, rows)
    y = rows.label.to_numpy().astype(int)
    train = rows.split.eq('train').to_numpy()
    val = rows.split.eq('validation').to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    best = None
    for depth, weight in CANDIDATES:
        model = XGBClassifier(**model_parameters(depth, weight, args))
        print(f'Training depth={depth}, positive_weight={weight}...', flush=True)
        model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
        m = metrics(y[val], probabilities(model, X[val], weight))
        result = dict(max_depth=depth, positive_weight=weight,
                      boosting_rounds=int(model.best_iteration + 1),
                      selection_score=(m['roc_auc'] + m['pr_auc']) / 2, **m)
        results.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(results).to_csv(out / 'validation_metrics.csv', index=False)
        if best is None or result['selection_score'] > best['selection_score']:
            best = result.copy()
    params = model_parameters(best['max_depth'], best['positive_weight'], args)
    params.pop('early_stopping_rounds')
    params['n_estimators'] = best['boosting_rounds']
    assert rows.source_dataset.eq('data0').all() and rows.split.isin(['train', 'validation']).all()
    print('Refitting on data0 train+validation only...', flush=True)
    final = XGBClassifier(**params).fit(X, y)
    if not manifest.equals(read_manifest(args.manifest)):
        raise ValueError('Manifest changed during training')
    bundle = {'model': final, 'positive_weight': best['positive_weight'],
              'feature_names': FEATURE_NAMES, 'feature_version': VERSION,
              'feature_hash': feature_hash(), 'manifest': manifest,
              'training_data': 'data0 train+validation; RF original manifest', 'seed': 4262}
    joblib.dump(bundle, out / 'selected_xgb.joblib', compress=3)
    report = {'selected': best, 'candidates': results, 'parameters': params,
              'training_scope': 'data0 train+validation only',
              'selection_rule': '(validation ROC-AUC + validation trapezoidal PR-AUC)/2',
              'manifest': str(args.manifest.resolve()),
              'versions': {'xgboost': xgboost.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__},
              'notes': ['Test excluded from all feature extraction, fitting and selection.',
                        'Input label file checked against the manifest for identity only.',
                        'Model is NOT refitted on all data0.']}
    (out / 'training_report.json').write_text(json.dumps(report, indent=2))
    print(f'Saved {out / "selected_xgb.joblib"}. Test was not evaluated.')


if __name__ == '__main__':
    main()
