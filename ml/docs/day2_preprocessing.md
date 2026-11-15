
# YantraDrishti - Day 2 Data Preparation

## Dataset
- Source: NASA C-MAPSS Turbofan Dataset
- Subset: FD001
- Training engines: 100
- Test engines: 100
- Training records: 20,631
- Test records: 13,096

## RUL Generation

Training:
RUL = Maximum cycle of engine - Current cycle

Testing:
RUL = Official RUL + (Last observed cycle - Current cycle)

Test RUL labels are stored separately from test input features.

## Train-Validation Split

- Training engines: 80
- Validation engines: 20
- Random state: 42
- Split performed by unit_number.
- No engine overlap between training and validation.

## Feature Engineering

- Canonical zero-padded sensor names are used (sensor_01 to sensor_21) to match backend schemas.
- Cycle-to-cycle sensor differences are added (sensor_01_delta to sensor_21_delta).
- Delta features are calculated separately for each engine.
- cycle_norm is excluded because it uses the maximum observed cycle (future information leakage).
- unit_number is an entity identifier, excluded from model feature columns.
- cycle is retained as a causal model input feature.

No physical meanings are assigned to anonymous sensor channels.

## Validation

- Missing values checked.
- Duplicate rows checked.
- Negative RUL checked.
- Engine overlap checked.

## Output Files

- train_prepared.csv
- validation_prepared.csv
- test_prepared.csv
- test_rul_expanded.csv
- test_rul_official.csv
- split_manifest.json

## Important Decisions

- No synthetic data used.
- No model training performed.
- No random row-wise splitting.
- No fitted scaler applied yet.
- Test input features and test RUL labels are stored separately (no RUL in test_prepared.csv).
- split_manifest.json contains the final feature and target schema (46 features, RUL target).

## Next Phase

Model training, preprocessing fit on training data only,
model evaluation, and inference pipeline development.
