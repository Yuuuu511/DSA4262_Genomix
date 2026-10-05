#!/usr/bin/env python3
"""Predict all sites, preserving their input transcript/position strings."""
import argparse,csv,joblib
import numpy as np
from features import sites,extract

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--model',default='model.joblib');p.add_argument('--output',required=True)
    a=p.parse_args();bundle=joblib.load(a.model)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['transcript_id','transcript_position','score'])
        batch=[];keys=[]
        def flush():
            if not batch:return
            scores=bundle['model'].predict_proba(np.asarray(batch))[:,1]
            for (t,pos),s in zip(keys,scores):w.writerow([t,pos,float(s)])
            batch.clear();keys.clear()
        for t,pos,seq,reads in sites(a.data):
            batch.append(extract(seq,reads));keys.append((t,pos))
            if len(batch)>=1024:flush()
        flush()
if __name__=='__main__':main()
