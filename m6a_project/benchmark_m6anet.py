"""Adapter for official m6Anet 2.0.1 pretrained weights on course nine-column reads.

Uses m6Anet's own network, kmer mapping and normalization. Site pooling is the
exact expectation of upstream's 20-read sampling-with-replacement noisy-OR,
rather than a finite Monte Carlo approximation. See README for limitations.
"""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from features import iter_sites

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    ap.add_argument('--heldout-predictions',required=True);ap.add_argument('--output-dir',default='artifacts')
    ap.add_argument('--vendor-path');args=ap.parse_args()
    if args.vendor_path:sys.path.insert(0,args.vendor_path)
    import torch,toml
    from m6anet.model.model import MILModel
    from m6anet.utils.constants import DEFAULT_MODEL_CONFIG,DEFAULT_MODEL_WEIGHTS,DEFAULT_NORM_PATH,KMER_TO_INT
    torch.set_num_threads(2)
    model=MILModel(toml.load(DEFAULT_MODEL_CONFIG))
    model.load_state_dict(torch.load(DEFAULT_MODEL_WEIGHTS,map_location='cpu',weights_only=True))
    model.eval();norm=joblib.load(DEFAULT_NORM_PATH)
    target=pd.read_csv(args.heldout_predictions)
    keys=set(zip(target.transcript_id,target.transcript_position));pending=[];rows=[];checks=[]
    def flush():
        if not pending:return
        signals=[];codes=[];lengths=[]
        for tx,pos,seq,reads in pending:
            kmers=[seq[i:i+5] for i in range(3)]
            mean=np.concatenate([norm[k][0] for k in kmers]);sd=np.concatenate([norm[k][1] for k in kmers])
            signals.append((np.asarray(reads)-mean)/sd)
            codes.append(np.tile([KMER_TO_INT[k] for k in kmers],(len(reads),1)));lengths.append(len(reads))
        with torch.inference_mode():
            representation=model.get_read_representation({'X':torch.tensor(np.concatenate(signals),dtype=torch.float32),
                'kmer':torch.tensor(np.concatenate(codes),dtype=torch.long)})
            probs=model.pooling_filter.probability_layer(representation).flatten().numpy().astype(float)
        start=0
        for (tx,pos,_,_),n in zip(pending,lengths):
            p=probs[start:start+n];start+=n
            score=float(-np.expm1(20*np.log1p(-min(p.mean(),1-1e-15))))
            rows.append((tx,pos,score))
            if not checks:
                rng=np.random.default_rng(4262)
                mc=float(np.mean(1-np.prod(1-rng.choice(p,(100000,20),replace=True),axis=1)))
                checks.append({'exact_expectation':score,'monte_carlo_100000_samples':mc,'absolute_difference':abs(score-mc)})
                assert abs(score-mc)<.01
        pending.clear()
    for tx,pos,seq,reads in iter_sites(args.data):
        if (tx,pos) in keys:pending.append((tx,pos,seq,reads))
        if len(pending)>=256:flush()
    flush()
    predictions=pd.DataFrame(rows,columns=['transcript_id','transcript_position','m6anet_score'])
    compared=target.merge(predictions,on=['transcript_id','transcript_position'],validate='one_to_one')
    if len(compared)!=len(target):raise ValueError('Missing m6Anet benchmark predictions')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    compared.to_csv(out/'m6anet_heldout_comparison.csv',index=False)
    # Score in a fresh process: this Mac's torch/XGBoost OpenMP runtimes conflict.
    import subprocess
    scorer = subprocess.run([sys.executable,str(Path(__file__).with_name('score_benchmark.py')),
        str(out/'m6anet_heldout_comparison.csv')],capture_output=True,text=True,check=True)
    scores=json.loads(scorer.stdout)
    report={'m6anet_version':'2.0.1','weights':'official human_hct116.pt','torch_version':torch.__version__,
        **scores,
        'pooling_check':checks[0],
        'limitations':['Pretrained m6Anet training overlap with course data is unknown; this is not proof of superior generalization.',
            'Adapter uses official network and normalization, not the official CLI; course reads have no read-ID tenth column.',
            'Exact expectation replaces upstream 1000-iteration stochastic pooling; run uses modern Python/PyTorch rather than the handout Python 3.8 environment.']}
    (out/'m6anet_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
