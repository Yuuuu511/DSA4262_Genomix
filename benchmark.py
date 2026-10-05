#!/usr/bin/env python3
"""Prepare indexed test JSON, or compare m6Anet and our frozen model."""
import argparse,csv,json,os
import pandas as pd
from features import sites
from train import metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('prepare');a.add_argument('--data',required=True);a.add_argument('--manifest',required=True);a.add_argument('--dataset',type=int,required=True);a.add_argument('--out-dir',required=True)
    a=s.add_parser('compare');a.add_argument('--manifest',required=True);a.add_argument('--dataset',type=int,required=True);a.add_argument('--m6anet',required=True);a.add_argument('--output',required=True)
    a=p.parse_args();d=pd.read_csv(a.manifest,dtype={'transcript_position':str});d=d[(d.split=='test')&(d.dataset==a.dataset)]
    if a.mode=='prepare':
        os.makedirs(a.out_dir,exist_ok=True);keys=set(zip(d.transcript_id,d.transcript_position))
        with open(a.out_dir+'/data.json','wb') as f,open(a.out_dir+'/data.info','w',newline='') as g:
            w=csv.writer(g);w.writerow(['transcript_id','transcript_position','n_reads','start','end'])
            for t,pos,seq,reads in sites(a.data):
                if (t,pos) not in keys:continue
                raw=(json.dumps({t:{pos:{seq:reads}}},separators=(',',':'))+'\n').encode();start=f.tell();f.write(raw);w.writerow([t,pos,len(reads),start,f.tell()])
    else:
        b=pd.read_csv(a.m6anet,dtype={'transcript_position':str});b=b.rename(columns={'probability_modified':'m6anet_score'})
        if b.duplicated(['transcript_id','transcript_position']).any():raise ValueError('Duplicate m6Anet predictions')
        z=d.merge(b[['transcript_id','transcript_position','m6anet_score']],on=['transcript_id','transcript_position'],validate='one_to_one')
        if z.empty or not z.m6anet_score.between(0,1).all():raise ValueError('No comparable predictions or invalid scores')
        result={'test_sites':len(d),'common_sites':len(z),'excluded_sites':len(d)-len(z),'ours':metrics(z.label,z.score),'m6anet':metrics(z.label,z.m6anet_score)}
        with open(a.output,'w') as f:json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
