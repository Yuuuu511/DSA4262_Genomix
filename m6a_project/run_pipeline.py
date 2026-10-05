"""Portable wrapper: train data0, predict three datasets, validate the CSVs."""
import argparse
from pathlib import Path
import subprocess,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--team-name',default='Genomix_haribo');ap.add_argument('--skip-training',action='store_true')
    args=ap.parse_args();project=Path(__file__).resolve().parent;data=Path(args.data_root).resolve()
    if any(c in args.team_name for c in '/\\') or not args.team_name:raise ValueError('Team name cannot contain a slash')
    def run(script,*arguments):
        subprocess.run([sys.executable,str(project/script),*map(str,arguments)],cwd=project,check=True)
    if not args.skip_training:
        run('train.py','--data',data/'data0/dataset0.json.gz','--labels',data/'data0/data.info.labelled',
            '--output-dir',project/'artifacts','--cache-dir',project/'.cache')
    for i in range(3):
        run('predict.py','--data',data/f'data{i}/dataset{i}.json.gz','--model',project/'artifacts/model.joblib',
            '--output',project/f'submissions/{args.team_name}_dataset{i}.csv','--cache-dir',project/'.cache')
    run('validate_submission.py','--data-root',data,'--submission-dir',project/'submissions','--team-name',args.team_name,
        '--report',project/'artifacts/submission_validation.json')

if __name__=='__main__':main()
