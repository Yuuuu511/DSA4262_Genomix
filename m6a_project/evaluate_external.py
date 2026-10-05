"""Post-selection checks only: external labels never tune the submitted model."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import spearmanr
from features import load_features,read_labels
from train import metrics,probabilities

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--project-dir',default='.')
    ap.add_argument('--cache-dir');args=ap.parse_args();root=Path(args.data_root);project=Path(args.project_dir)
    split=pd.read_csv(project/'artifacts/gene_split.csv');tx_gene=split[['transcript_id','gene_id']].drop_duplicates()
    assert not tx_gene.transcript_id.duplicated().any()
    gene_map=tx_gene.set_index('transcript_id').gene_id
    test_genes=set(split.loc[split.split=='test','gene_id'])
    heldout=joblib.load(project/'artifacts/heldout_model.joblib')
    report={}
    for i in [1,2]:
        meta,X=load_features(root/f'data{i}'/f'dataset{i}.json.gz',args.cache_dir)
        labels=read_labels(root/f'data{i}'/'data.info',meta)
        predictions=pd.read_csv(project/f'submissions/Genomix_haribo_dataset{i}.csv')
        joined=labels.merge(predictions,on=['transcript_id','transcript_position'],validate='one_to_one')
        if i==1:
            report['dataset1_all_sites']=metrics(joined.label,joined.score)
            mask=meta.transcript_id.map(gene_map).isin(test_genes).to_numpy()
            p=probabilities(heldout['model'],X[mask],heldout['positive_weight'])
            report['dataset1_data0_heldout_genes']=metrics(labels.label.to_numpy()[mask],p)
            report['dataset1_note']='All-site evaluation has overlapping training genes. The heldout-gene subset uses the evaluation model trained without these genes; neither result was used for tuning.'
        else:
            groups=joined.groupby('label').score.agg(['count','mean','median']).reset_index()
            report['dataset2_score_by_fractional_label']=groups.to_dict(orient='records')
            report['dataset2_spearman_with_fractional_label']=float(spearmanr(joined.label,joined.score).statistic)
            report['dataset2_note']='Descriptive association only. No binary conversion or ROC/PR AUC assumed; fractional label semantics and instructor evaluation convention are unspecified.'
    (project/'artifacts/external_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
