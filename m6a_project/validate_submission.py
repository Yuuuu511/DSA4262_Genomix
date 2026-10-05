"""Check exact site coverage, CSV schema, score validity and filename."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from features import iter_sites

def validate(data,predictions):
    p=pd.read_csv(predictions)
    if p.columns.tolist()!=['transcript_id','transcript_position','score']:raise ValueError('Wrong columns/order')
    if p.isna().any().any() or p.duplicated(['transcript_id','transcript_position']).any():raise ValueError('Missing values or duplicate sites')
    if not np.isfinite(p.score).all() or not p.score.between(0,1).all():raise ValueError('Scores must be finite and in [0,1]')
    if not np.equal(p.transcript_position,p.transcript_position.astype(int)).all():raise ValueError('Noninteger positions')
    expected={(tx,pos) for tx,pos,_,_ in iter_sites(data)}
    actual=set(zip(p.transcript_id,p.transcript_position))
    if expected!=actual:raise ValueError(f'Site mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    return {'file':Path(predictions).name,'rows':len(p),'score_min':float(p.score.min()),'score_max':float(p.score.max()),'status':'PASS'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--submission-dir',default='submissions')
    ap.add_argument('--team-name',default='Genomix_haribo');ap.add_argument('--report');args=ap.parse_args()
    directory=Path(args.submission_dir);expected={f'{args.team_name}_dataset{i}.csv' for i in range(3)}
    if {p.name for p in directory.glob('*.csv')}!=expected:raise ValueError('Submission directory must contain exactly the three named CSV files')
    result=[validate(Path(args.data_root)/f'data{i}'/f'dataset{i}.json.gz',directory/f'{args.team_name}_dataset{i}.csv') for i in range(3)]
    if args.report:Path(args.report).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
