"""Generate and validate three prediction files using the frozen chosen model."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=root / 'artifacts/selected_xgb.joblib')
    parser.add_argument('--output-dir', type=Path, default=root / 'submissions')
    parser.add_argument('--team-name', default='Genomix_haribo')
    args = parser.parse_args()
    if not args.team_name or any(c in args.team_name for c in '/\\'):
        parser.error('Team name must be a nonempty filename component')
    pairs = [(args.data_root / f'data{i}/dataset{i}.json.gz',
              args.output_dir / f'{args.team_name}_dataset{i}.csv') for i in range(3)]
    for data, output in pairs:
        if not data.is_file():
            raise FileNotFoundError(data)
        if output.exists():
            raise FileExistsError(f'Existing prediction: {output}; choose another output folder')
    for data, output in pairs:
        subprocess.run([sys.executable, str(root / 'predict.py'), '--data', str(data),
                        '--model', str(args.model), '--output', str(output)], check=True)
        subprocess.run([sys.executable, str(root / 'validate_submission.py'), '--data', str(data),
                        '--predictions', str(output)], check=True)


if __name__ == '__main__':
    main()
