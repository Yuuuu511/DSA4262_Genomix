"""Train exclusively on data0, select on validation genes, evaluate once on test genes."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc,brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import load_features,read_labels,FEATURE_NAMES,VERSION

def metrics(y,p):
    precision,recall,_=precision_recall_curve(y,p)
    return {'roc_auc':float(roc_auc_score(y,p)), 'pr_auc_trapezoid':float(auc(recall,precision)),
        'average_precision':float(average_precision_score(y,p)), 'brier_score':float(brier_score_loss(y,p)),
        'sites':len(y),'positive_sites':int(np.sum(y)),'positive_fraction':float(np.mean(y))}

def probabilities(model,X,weight=1):
    p=model.predict_proba(X)[:,1].astype(float)
    # Analytic prior correction for positive class reweighting (not empirical calibration).
    return p/(weight-(weight-1)*p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--labels',required=True)
    ap.add_argument('--output-dir',default='artifacts');ap.add_argument('--cache-dir');ap.add_argument('--seed',type=int,default=4262)
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    meta,X=load_features(args.data,args.cache_dir);labels=read_labels(args.labels,meta)
    if 'gene_id' not in labels:raise ValueError('Training requires gene_id for leakage-safe splits')
    y=labels.label.to_numpy()
    if not np.isin(y,[0,1]).all():raise ValueError('Training labels must be binary; use data0')
    y=y.astype(int);genes=labels.gene_id.to_numpy()
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=args.seed).split(X,y,genes))
    test=folds[0][1];val=folds[1][1];train=np.concatenate([f[1] for f in folds[2:]])
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not set(genes[a]) & set(genes[b])
    split=np.full(len(y),'train',dtype='<U10');split[val]='validation';split[test]='test'
    labels.assign(split=split).to_csv(out/'gene_split.csv',index=False)
    report={'training_dataset':str(Path(args.data).resolve()),'seed':args.seed,
        'features':FEATURE_NAMES,'feature_version':VERSION,
        'versions':{'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'numpy':np.__version__},
        'split':{name:{'sites':int((split==name).sum()),'genes':len(set(genes[split==name]))} for name in ['train','validation','test']},
        'candidates':[]}
    baseline_idx=[FEATURE_NAMES.index(f'{s}_mean') for s in ['minus1_dwell','minus1_signal_sd','minus1_current','center_dwell','center_signal_sd','center_current','plus1_dwell','plus1_signal_sd','plus1_current']]
    baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=args.seed))
    baseline.fit(X[train][:,baseline_idx],y[train]);report['baseline_validation']=metrics(y[val],baseline.predict_proba(X[val][:,baseline_idx])[:,1])
    best=None
    for depth,weight in [(3,1),(5,1),(3,4),(5,4)]:
        params=dict(max_depth=depth,scale_pos_weight=weight,learning_rate=.05,n_estimators=1200,
            min_child_weight=15,subsample=.85,colsample_bytree=.85,reg_lambda=10,reg_alpha=.1,
            tree_method='hist',n_jobs=4,random_state=args.seed,eval_metric='aucpr',early_stopping_rounds=60)
        model=XGBClassifier(**params)
        model.fit(X[train],y[train],eval_set=[(X[val],y[val])],verbose=False)
        m=metrics(y[val],probabilities(model,X[val],weight));score=(m['roc_auc']+m['pr_auc_trapezoid'])/2
        result={'max_depth':depth,'positive_weight':weight,'boosting_rounds':int(model.best_iteration+1),'validation':m,'selection_score':score}
        report['candidates'].append(result);print(json.dumps(result),flush=True)
        if best is None or score>best[0]:best=(score,params,result)
    _,params,selected=best;params.pop('early_stopping_rounds');params['n_estimators']=selected['boosting_rounds']
    dev=np.concatenate([train,val]);heldout_model=XGBClassifier(**params).fit(X[dev],y[dev])
    test_p=probabilities(heldout_model,X[test],selected['positive_weight'])
    report['selected']=selected;report['heldout_test']=metrics(y[test],test_p)
    baseline.fit(X[dev][:,baseline_idx],y[dev]);report['baseline_heldout_test']=metrics(y[test],baseline.predict_proba(X[test][:,baseline_idx])[:,1])
    report['random_reference']={'roc_auc':.5,'pr_auc_reference':float(y[test].mean())}
    labels.iloc[test].assign(score=test_p).to_csv(out/'heldout_predictions.csv',index=False)
    # Save the evaluation model separately, then refit one submission model on ALL data0.
    def save(model,path):
        joblib.dump({'model':model,'positive_weight':selected['positive_weight'],'feature_names':FEATURE_NAMES,
            'feature_version':VERSION,'training_data':'data0 only','seed':args.seed},path)
    save(heldout_model,out/'heldout_model.joblib')
    final=XGBClassifier(**params).fit(X,y);save(final,out/'model.joblib')
    final.save_model(out/'model.ubj')
    pd.DataFrame({'feature':FEATURE_NAMES,'importance':final.feature_importances_}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    report['notes']=['Test genes were never used for tuning. Final submission model was refitted on all data0.',
        'Data0 submission performance is in-sample, not generalization performance.',
        'Dataset1 and dataset2 labels were not used for training or model selection.',
        'PR AUC uses trapezoidal integration; average precision is reported separately.',
        'Class-weight correction is analytic; probabilities are not empirically calibrated.']
    (out/'training_report.json').write_text(json.dumps(report,indent=2))
    print('Heldout test:',json.dumps(report['heldout_test']),flush=True)
    print('Baseline heldout:',json.dumps(report['baseline_heldout_test']),flush=True)

if __name__=='__main__':main()
