"""Isolated metric scoring, without loading PyTorch into the XGBoost process."""
import json,sys
import pandas as pd
from train import metrics

if __name__=='__main__':
    frame=pd.read_csv(sys.argv[1])
    print(json.dumps({'our_model':metrics(frame.label,frame.score),'m6anet':metrics(frame.label,frame.m6anet_score)}))
