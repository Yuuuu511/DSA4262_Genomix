# DSA4262_Genomix
DSA4262 Team Project - Prediction of m6A RNA modifications from direct RNA-Seq data

**The interactive Task 2 Streamlit dashboard is also available online at [https://dsa4262genomix.streamlit.app/](https://dsa4262genomix.streamlit.app/).**

N6-methyladenosine (m6A) is a common mRNA modification that regulates gene expression through processes such as RNA degradation and translation. Nanopore direct RNA sequencing captures signals that can help identify these modifications.
This project develops a reproducible machine learning workflow to predict m6A sites in the SG-NEx dataset using sequencing signal and sequence features processed by m6Anet. It aims to:

- Distinguish modified from unmodified transcript sites and estimate modification probabilities for unseen data.
- Evaluate prediction accuracy and generalisation.
- Identify modification patterns that are reproducible within cell lines and differ across cell lines.
- Support exploratory biological analysis through an interactive visualisation platform.

## Project Documentation

### Task 1: m6A Prediction

Training, prediction and model evaluation using XGBoost.

See the [Task 1 documentation](Task1/README.md) for installation instructions, a step-by-step prediction example, input/output formats and model evaluation.

### Task 2: SG-NEx Analysis and Visualisation

Analysis of prediction patterns across SG-NEx runs and cell lines, with an interactive visualisation dashboard.

See the [Task 2 documentation](Task2/README.md) for the analysis workflow and dashboard instructions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.md) for details.
