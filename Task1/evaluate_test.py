"""Report packaged joint-model performance on the original held-out genes."""
import sys
from pathlib import Path
import train


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    sys.argv.extend(['--evaluate-test'])
    if not any(arg == '--model' or arg.startswith('--model=') for arg in sys.argv):
        sys.argv.extend(['--model', str(root / 'artifacts/selected_xgb.joblib')])
    if not any(arg == '--evaluation-dir' or arg.startswith('--evaluation-dir=') for arg in sys.argv):
        sys.argv.extend(['--evaluation-dir', str(root / 'evaluation/recomputed')])
    train.main()
