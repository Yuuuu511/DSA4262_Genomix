# Push to your existing GitHub branch

The local project contains code, reports, pinned dependencies, four small trained models, a compact gene mapping and a 20-site inference example. Destination: [Yuuuu511/DSA4262_Genomix](https://github.com/Yuuuu511/DSA4262_Genomix), branch [`task1_ben`](https://github.com/Yuuuu511/DSA4262_Genomix/tree/task1_ben). A checkout of the existing branch is prepared locally at `outputs/github-checkout` relative to the original working folder. The original working folder has a separate, uncommitted Git repository; run Git commands in the branch checkout.

## Files to include

Include the root project files plus `.github/`, `docs/`, `example/`, `models/`, `reference/`, `results/` and `tests/`. Include `.gitignore` and `.gitattributes`; the latter preserves checksum-bound snapshot bytes across platforms.

Git ignores `.venv/`, `artifacts/`, `data/`, `outputs/`, caches, `.DS_Store`, `.Rhistory` and environment-secret files. Full training data is supplied separately. Large raw annotations are replaced by the compact map and provenance. Package checks are in `results/repository_verification.json`.

## Use the existing branch

The prepared checkout extends the destination branch's existing history. For a new checkout on another machine:

```bash
git clone --single-branch --branch task1_ben https://github.com/Yuuuu511/DSA4262_Genomix.git DSA4262_Genomix
cd DSA4262_Genomix
```

The prepared local checkout already contains the baseline files, including hidden Git configuration files. After the branch is pushed, a fresh clone contains them too. It retains the initial commit, the repository name and the original Python ignore rules, with project-specific exclusions appended.

Then, from the checkout directory:

```bash
git status --short --branch
git diff --stat
git diff
```

Use the installed Python environment to run the tests from this checkout:

```bash
python -m unittest discover -s tests -v
```

For a future update, review the changes and then:

```bash
git add .
git diff --cached --stat
git commit -m "Add reproducible m6A baseline, models and experiment reports"
git push -u origin task1_ben
```

The clone already configures `origin`. The destination branch is the branch you created on GitHub. No default-branch rename or new GitHub repository is needed. GitHub authentication uses your configured credentials.

## Verify the upload

Confirm the README renders, model/reference/example files are present and the Ubuntu workflow succeeds. From a fresh clone, install the pinned requirements and run:

```bash
python predict.py --data example/test_data.json.gz \
  --output outputs/example_predictions.csv
python -m unittest discover -s tests -v
```

Prediction needs no full training data. To train, supply the datasets and follow the root README. For future model updates, refresh the model manifest, numeric summaries and expected example predictions together. No software licence has been selected; source data and annotations retain their own terms.

Git command guidance: [GitHub's local repository instructions](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github). Workflow sources: [checkout](https://github.com/actions/checkout) and [Python setup](https://github.com/actions/setup-python).
