"""Predict from a JSON/JSON.GZ file; requires no labels or gene metadata."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from features import load_features,FEATURE_NAMES,VERSION

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--cache-dir');args=ap.parse_args()
    bundle=joblib.load(args.model)
    if bundle['feature_names']!=FEATURE_NAMES or bundle['feature_version']!=VERSION:raise ValueError('Model feature version mismatch')
    meta,X=load_features(args.data,args.cache_dir)
    p=bundle['model'].predict_proba(X)[:,1].astype(float);w=bundle['positive_weight'];p=p/(w-(w-1)*p)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid scores')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    meta.assign(score=p).to_csv(out,index=False)
    print(f'Saved {len(meta):,} predictions to {out}',flush=True)

if __name__=='__main__':main()
