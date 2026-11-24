# Day 3 — FD001 RUL Regression Baseline

## Objective
Train and evaluate baseline Remaining Useful Life regression models using NASA C-MAPSS FD001 prepared training and validation datasets.

## Dataset
- Dataset: NASA C-MAPSS FD001
- Training rows: 16561
- Validation rows: 4070
- Features: 46
- Training engines: 80
- Validation engines: 20

## Models
1. DummyRegressor(strategy='mean')
2. RandomForestRegressor

### Random Forest Parameters
- n_estimators: 150
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42
- n_jobs: -1

## Validation Results

| Model | MAE | RMSE | NASA Score |
|---|---:|---:|---:|
| Dummy Regressor | 55.363 | 65.715 | 19351068.755 |
| Random Forest | 23.474 | 30.871 | 253571.045 |

## Evaluation
Random Forest outperformed the Dummy Regressor across all three evaluation metrics.

NASA score penalizes RUL overestimation more heavily than underestimation.

## Data Validation
- No missing feature values.
- No duplicate unit-cycle records.
- No overlap between training and validation engines.
- unit_number excluded from model features.
- RUL excluded from model input features.
- Test dataset was not used for training or model selection.

## Limitations
- Baseline model with limited hyperparameter tuning.
- Evaluation performed only on the validation split.
- Results are specific to FD001 and should not be generalized to other subsets without testing.
- No anomaly detection or SHAP analysis was performed.

## Saved Artifacts
- ml/models/rul_baseline.joblib
- ml/metrics/day3_rul_metrics.json
- ml/reports/day3_rul_baseline.md
