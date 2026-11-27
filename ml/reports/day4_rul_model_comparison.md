# Day 4 — FD001 RUL Model Comparison (Random Forest vs XGBoost)

## Objective
Compare the Day 3 Random Forest RUL baseline against an XGBoost Regressor on the NASA C-MAPSS FD001 dataset using the identical engine-wise validation split and 46-feature contract.

## Methodology
- **Dataset:** NASA C-MAPSS FD001
- **Split Strategy:** Engine-wise 80:20 split (`split_manifest.json`, `random_state=42`)
  - Training: 16,561 rows across 80 engines
  - Validation: 4,070 rows across 20 engines
  - Train-Validation Engine Overlap: 0
- **Features:** 46 features (`cycle`, `setting_1..3`, `sensor_01..21`, `sensor_01_delta..sensor_21_delta`); `unit_number` and `RUL` excluded.
- **Models Evaluated:**
  1. `DummyRegressor(strategy="mean")`
  2. `RandomForestRegressor` (`n_estimators=150`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`)
  3. `XGBRegressor` (`n_estimators=500`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=5`, `tree_method="hist"`, `random_state=42`)
- **Evaluation Metrics:** MAE, RMSE, and the asymmetric NASA C-MAPSS Score (which penalizes RUL overestimation more heavily than underestimation).
- **Test Set Isolation:** The test dataset (`test_prepared.csv`) was **not** used for training, tuning, or model selection.
- **Production Model Status:** The Day 3 production artifact (`ml/models/rul_baseline.joblib`) was **not** overwritten or replaced.

## Validation Metrics

| Model | MAE | RMSE | NASA Score | Training Time (s) |
|---|---:|---:|---:|---:|
| Dummy Regressor | 55.363 | 65.715 | 19351068.755 | - |
| Random Forest (Day 3 Baseline) | 23.474 | 30.871 | 253571.045 | 6.75 |
| XGBoost | 22.424 | 29.624 | 160813.110 | 1.02 |

## Comparison Summary
- **XGBoost vs. Random Forest:**
  - **MAE:** Improved from `23.474` to `22.424` (**4.48%** reduction).
  - **RMSE:** Improved from `30.871` to `29.624` (**4.04%** reduction).
  - **NASA Score:** Improved from `253571.045` to `160813.110` (**36.58%** reduction), indicating substantially fewer severe late-prediction / overestimation errors.
  - **Training Speed:** XGBoost (`tree_method="hist"`) trained in `1.02s` compared to `6.75s` for Random Forest.
- **Reproducibility:** The Day 3 Random Forest baseline metrics (`MAE=23.474`, `RMSE=30.871`, `NASA Score=253571.045`) were reproduced exactly.

## Feature Importance Interpretation (XGBoost)
Top 10 features ranked by XGBoost split gain importance:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `cycle` | 0.408937 |
| 2 | `sensor_11` | 0.151684 |
| 3 | `sensor_12` | 0.070482 |
| 4 | `sensor_04` | 0.068352 |
| 5 | `sensor_09` | 0.038533 |
| 6 | `sensor_07` | 0.032991 |
| 7 | `sensor_15` | 0.023860 |
| 8 | `sensor_21` | 0.017455 |
| 9 | `sensor_14` | 0.016838 |
| 10 | `sensor_20` | 0.016480 |

- **Important Note:** Tree-based feature importance measures each feature's relative contribution to reducing training loss within the fitted model. It reflects **model-based statistical association**, **not causal physical relationships**.
- 13 features received `0.0` importance in XGBoost (`sensor_01, sensor_10, sensor_05, setting_3, sensor_16, sensor_05_delta, sensor_01_delta, sensor_19, sensor_18, sensor_10_delta, sensor_16_delta, sensor_19_delta, sensor_18_delta`), consistent with constant or near-constant channels in C-MAPSS FD001.

## Limitations
- Evaluation was conducted strictly on the 20-engine FD001 validation split; held-out test set performance was not evaluated.
- Hyperparameters for XGBoost were selected as a clean regularized configuration without exhaustive cross-validated grid search.
- Results are specific to C-MAPSS FD001 (single operating condition, single fault mode) and do not generalize directly to FD002–FD004 without further validation.
- No SHAP local attribution or anomaly detection was performed in this comparison.

## Environment & Versions
- **Python:** `3.12.5`
- **scikit-learn:** `1.7.1`
- **xgboost:** `3.4.1`
- **pandas:** `2.3.1`
- **numpy:** `2.3.1`
- **Evaluation Timestamp (UTC):** `2026-11-27T21:20:03.594725+00:00`

## Artifacts
- `ml/notebooks/YantraDrishti_Day4_RUL_Model_Comparison.ipynb`
- `ml/metrics/day4_rul_comparison.json`
- `ml/reports/day4_rul_model_comparison.md`
