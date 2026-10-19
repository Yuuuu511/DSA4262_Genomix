# Assistance and provenance

The team used OpenAI Codex assistance to organize and review code, implement input/output checks, and document training, prediction and benchmark workflows. The team should review this file and complete any required course AI-use disclosure in its own words.

The bundled main XGBoost is the user's previously trained joint data0+data1 model (depth 3, positive weight 1, 912 rounds), trained on the user's Mac. Package preparation did not retrain that main model. Label-bearing metadata was removed from its bundle while the fitted booster and feature implementation were verified unchanged.

During preparation, the assistant executed the fixed simple logistic baseline on the same supplied raw data/gene partitions and performed its separate test reporting. That baseline is explicitly auxiliary. The assistant also executed local prediction-contract checks, synthetic training/leakage guards, source/site split reconstruction, and real frozen-model test reporting.

Official pretrained m6Anet site probabilities were generated previously by the user on AWS using m6Anet 2.0.2/PyTorch 1.6.0/Python 3.8. Package preparation rescored those existing outputs on identical test keys; it did not run new m6Anet inference or retrain its weights. Original m6Anet training overlap is unknown.
