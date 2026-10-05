#!/usr/bin/env python3
"""Train with gene-disjoint 60/20/20 partitions and validation-only selection."""
import argparse,json,os,joblib
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score,f1_score
from features import sites,extract
KEY=['transcript_id','transcript_position']
def metrics(y,p):
    return dict(n=len(y),positive=int(sum(y)),prevalence=float(np.mean(y)),roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),precision_05=float(precision_score(y,p>=.5,zero_division=0)),recall_05=float(recall_score(y,p>=.5,zero_division=0)),f1_05=float(f1_score(y,p>=.5,zero_division=0)))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',nargs='+',required=True);ap.add_argument('--labels',nargs='+',required=True);ap.add_argument('--out-dir',default='results');ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if len(args.data)!=len(args.labels):ap.error('One label file is required for each data file')
    os.makedirs(args.out_dir,exist_ok=True)
    tables=[pd.read_csv(p,dtype={'transcript_id':str,'transcript_position':str}) for p in args.labels]
    gene_map={}
    for d in tables:
        if d.duplicated(KEY).any() or not d.label.isin([0,1]).all():raise ValueError('Duplicate keys or invalid labels')
        if 'gene_id' in d:
            for t,g in zip(d.transcript_id,d.gene_id):
                if pd.isna(g):continue
                if t in gene_map and gene_map[t]!=g:raise ValueError('Inconsistent transcript/gene mapping')
                gene_map[t]=g
    rows=[];X=[];audit=[]
    for i,(path,d) in enumerate(zip(args.data,tables)):
        lookup=d.set_index(KEY).label.to_dict();matched=set();total=0
        for t,pos,seq,reads in sites(path):
            total+=1;k=(t,pos)
            if k not in lookup:continue
            if k in matched:raise ValueError('Duplicate site in JSON')
            matched.add(k);X.append(extract(seq,reads));rows.append([i,t,pos,int(lookup[k]),gene_map.get(t,'transcript:'+t),len(reads)])
        audit.append(dict(dataset=i,json_sites=total,matched=len(matched),unmatched_labels=len(d)-len(matched)))
        print('Extracted',audit[-1],flush=True)
    meta=pd.DataFrame(rows,columns=['dataset',*KEY,'label','group','n_reads']);X=np.asarray(X);y=meta.label.to_numpy();groups=meta.group.to_numpy()
    folds=np.full(len(y),-1)
    for i,(_,idx) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=args.seed).split(X,y,groups)):folds[idx]=i
    split=np.where(folds==0,'test',np.where(folds==1,'validation','train'));meta['split']=split
    tr=split=='train';va=split=='validation';te=split=='test'
    for name in ['train','validation','test']:
        if len(np.unique(y[split==name]))!=2:raise ValueError('Both classes required in every partition')
    trials=[];best=None;best_ap=-1
    for leaves,l2 in [(15,1.),(31,5.),(31,20.)]:
        model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=leaves,l2_regularization=l2,learning_rate=.08,early_stopping=False,random_state=args.seed)
        model.fit(X[tr],y[tr]);score=average_precision_score(y[va],model.predict_proba(X[va])[:,1]);trials.append(dict(leaves=leaves,l2=l2,validation_ap=float(score)))
        print('Candidate',trials[-1],flush=True)
        if score>best_ap:best_ap=score;best=model
    # Freeze the selected training-only model. Do not refit on validation or test.
    p=best.predict_proba(X)[:,1];meta['score']=p;meta.to_csv(args.out_dir+'/site_predictions.csv',index=False)
    report={'seed':args.seed,'audit':audit,'trials':trials,'metrics':{s:metrics(y[split==s],p[split==s]) for s in ['train','validation','test']},'test_by_dataset':{str(i):metrics(y[te&(meta.dataset==i)],p[te&(meta.dataset==i)]) for i in range(len(tables))},'unmapped_gene_rows':int(meta.group.str.startswith('transcript:').sum())}
    # Gene-cluster bootstrap intervals preserve dependence between sites.
    rng=np.random.default_rng(args.seed);test=meta[te];gg=test.group.unique();by={g:test[test.group==g] for g in gg};boot=[]
    for _ in range(200):
        b=pd.concat([by[g] for g in rng.choice(gg,len(gg),replace=True)])
        if b.label.nunique()==2:boot.append([roc_auc_score(b.label,b.score),average_precision_score(b.label,b.score)])
    report['test_95_ci']={k:np.quantile(np.asarray(boot)[:,i],[.025,.975]).tolist() for i,k in enumerate(['roc_auc','average_precision'])}
    with open(args.out_dir+'/metrics.json','w') as f:json.dump(report,f,indent=2)
    joblib.dump({'model':best,'feature_version':1,'seed':args.seed},args.out_dir+'/model.joblib')
    print(json.dumps(report['metrics'],indent=2),flush=True)
if __name__=='__main__':main()
