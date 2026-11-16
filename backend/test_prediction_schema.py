import copy
import json
from pathlib import Path
from pydantic import ValidationError

from app.schemas.prediction import (
    RULPredictionRequest,
    RULPredictionResponse,
    PredictionStatus,
    FEATURE_ORDER,
    EXPECTED_OPERATING_SETTINGS,
    EXPECTED_SENSORS,
    EXPECTED_SENSOR_DELTAS,
)

# Standard sample valid request payload
sample_prediction_request = {
    "machine_id": "FD001-ENGINE-001",
    "cycle": 100,
    "operating_settings": {
        "setting_1": -0.0007,
        "setting_2": -0.0004,
        "setting_3": 100.0,
    },
    "sensors": {f"sensor_{i:02d}": 500.0 + i for i in range(1, 22)},
    "sensor_deltas": {f"sensor_{i:02d}_delta": 0.05 * i for i in range(1, 22)},
}

# Standard sample valid response payload
sample_prediction_response = {
    "machine_id": "FD001-ENGINE-001",
    "predicted_rul_cycles": 112.5,
    "unit": "cycles",
    "model_name": "FD001_LinearBaseline",
    "model_version": "1.0.0",
    "prediction_status": PredictionStatus.SUCCESS,
}


def expect_validation_error(data, test_name, schema_cls=RULPredictionRequest):
    try:
        schema_cls.model_validate(data)
    except ValidationError:
        print(f"PASS: {test_name}")
        return

    raise AssertionError(f"FAIL: {test_name} - Expected ValidationError was not raised!")


def test_valid_request():
    req = RULPredictionRequest.model_validate(sample_prediction_request)
    assert req.machine_id == "FD001-ENGINE-001"
    assert req.cycle == 100
    assert len(req.operating_settings) == 3
    assert len(req.sensors) == 21
    assert len(req.sensor_deltas) == 21
    print("PASS: Valid prediction request accepted")


def test_missing_sensor():
    data = copy.deepcopy(sample_prediction_request)
    del data["sensors"]["sensor_21"]
    expect_validation_error(data, "Missing sensor channel rejected")


def test_missing_sensor_delta():
    data = copy.deepcopy(sample_prediction_request)
    del data["sensor_deltas"]["sensor_01_delta"]
    expect_validation_error(data, "Missing sensor delta channel rejected")


def test_missing_operating_setting():
    data = copy.deepcopy(sample_prediction_request)
    del data["operating_settings"]["setting_2"]
    expect_validation_error(data, "Missing operating setting rejected")


def test_unexpected_field():
    data = copy.deepcopy(sample_prediction_request)
    data["extra_field"] = "unexpected_data"
    expect_validation_error(data, "Unexpected top-level field rejected (extra=forbid)")


def test_unexpected_field_in_sensors():
    data = copy.deepcopy(sample_prediction_request)
    data["sensors"]["sensor_22"] = 500.0
    expect_validation_error(data, "Unexpected sensor channel rejected")


def test_invalid_sensor_name():
    data = copy.deepcopy(sample_prediction_request)
    # sensor_1 without zero padding
    del data["sensors"]["sensor_01"]
    data["sensors"]["sensor_1"] = 501.0
    expect_validation_error(data, "Unpadded sensor name rejected")


def test_nan_operating_setting():
    data = copy.deepcopy(sample_prediction_request)
    data["operating_settings"]["setting_1"] = float("nan")
    expect_validation_error(data, "NaN operating setting rejected")


def test_inf_sensor():
    data = copy.deepcopy(sample_prediction_request)
    data["sensors"]["sensor_05"] = float("inf")
    expect_validation_error(data, "Infinite sensor reading rejected")


def test_nan_sensor_delta():
    data = copy.deepcopy(sample_prediction_request)
    data["sensor_deltas"]["sensor_10_delta"] = float("nan")
    expect_validation_error(data, "NaN sensor delta rejected")


def test_zero_cycle():
    data = copy.deepcopy(sample_prediction_request)
    data["cycle"] = 0
    expect_validation_error(data, "Zero cycle rejected (gt=0)")


def test_negative_cycle():
    data = copy.deepcopy(sample_prediction_request)
    data["cycle"] = -5
    expect_validation_error(data, "Negative cycle rejected")


def test_empty_machine_id():
    data = copy.deepcopy(sample_prediction_request)
    data["machine_id"] = ""
    expect_validation_error(data, "Empty machine_id rejected (min_length=1)")


def test_feature_ordering_and_vector_conversion():
    req = RULPredictionRequest.model_validate(sample_prediction_request)
    feature_dict = req.to_feature_dict()
    feature_vector = req.to_feature_vector()

    # Exact 46-feature length
    assert len(feature_dict) == 46, f"Expected 46 features, got {len(feature_dict)}"
    assert len(feature_vector) == 46, f"Expected 46 vector elements, got {len(feature_vector)}"

    # Exact key ordering matches FEATURE_ORDER
    dict_keys = list(feature_dict.keys())
    assert dict_keys == list(FEATURE_ORDER), "Feature dict keys do not match FEATURE_ORDER tuple!"

    # Exclude machine_id from model features
    assert "machine_id" not in feature_dict, "machine_id must NOT be included in model features!"

    # Verify order structure:
    assert dict_keys[0] == "cycle"
    assert dict_keys[1:4] == ["setting_1", "setting_2", "setting_3"]
    assert dict_keys[4:25] == [f"sensor_{i:02d}" for i in range(1, 22)]
    assert dict_keys[25:46] == [f"sensor_{i:02d}_delta" for i in range(1, 22)]

    # Cross-reference with Day 2 split_manifest.json if available
    manifest_path = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "split_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_features = manifest.get("feature_columns", [])
        assert dict_keys == manifest_features, "Feature order deviates from Day 2 split_manifest.json!"
        print("PASS: Feature order matches Day 2 split_manifest.json perfectly")

    print("PASS: Exact 46-feature ordering verified")
    print("PASS: machine_id excluded from model features verified")


def test_valid_response():
    resp = RULPredictionResponse.model_validate(sample_prediction_response)
    assert resp.machine_id == "FD001-ENGINE-001"
    assert resp.predicted_rul_cycles == 112.5
    assert resp.unit == "cycles"
    assert resp.model_name == "FD001_LinearBaseline"
    assert resp.model_version == "1.0.0"
    assert resp.prediction_status == PredictionStatus.SUCCESS
    print("PASS: Valid prediction response accepted")


def test_response_negative_rul():
    data = copy.deepcopy(sample_prediction_response)
    data["predicted_rul_cycles"] = -1.0
    expect_validation_error(data, "Negative predicted RUL rejected", schema_cls=RULPredictionResponse)


def test_response_nan_rul():
    data = copy.deepcopy(sample_prediction_response)
    data["predicted_rul_cycles"] = float("nan")
    expect_validation_error(data, "NaN predicted RUL rejected", schema_cls=RULPredictionResponse)


def test_response_extra_field():
    data = copy.deepcopy(sample_prediction_response)
    data["shap_values"] = {"sensor_01": 0.25}
    expect_validation_error(data, "Unexpected field in response rejected (extra=forbid)", schema_cls=RULPredictionResponse)


if __name__ == "__main__":
    tests = [
        test_valid_request,
        test_missing_sensor,
        test_missing_sensor_delta,
        test_missing_operating_setting,
        test_unexpected_field,
        test_unexpected_field_in_sensors,
        test_invalid_sensor_name,
        test_nan_operating_setting,
        test_inf_sensor,
        test_nan_sensor_delta,
        test_zero_cycle,
        test_negative_cycle,
        test_empty_machine_id,
        test_feature_ordering_and_vector_conversion,
        test_valid_response,
        test_response_negative_rul,
        test_response_nan_rul,
        test_response_extra_field,
    ]

    for test in tests:
        test()

    print(f"\nAll {len(tests)} prediction schema tests passed successfully!")
