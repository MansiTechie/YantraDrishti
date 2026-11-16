import copy
import math
from pydantic import ValidationError

from app.schemas.prediction import (
    FEATURE_ORDER,
    RULPredictionRequest,
)
from app.services.feature_service import (
    FEATURE_COUNT,
    extract_feature_dict,
    extract_feature_vector,
)

# Reference sample payload
sample_request_data = {
    "machine_id": "FD001-ENGINE-042",
    "cycle": 85,
    "operating_settings": {
        "setting_1": 0.0015,
        "setting_2": -0.0002,
        "setting_3": 100.0,
    },
    "sensors": {f"sensor_{i:02d}": 450.0 + (i * 2.5) for i in range(1, 22)},
    "sensor_deltas": {f"sensor_{i:02d}_delta": -0.02 * i for i in range(1, 22)},
}


def test_feature_count():
    """Verify output contains exactly 46 values."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    feature_dict = extract_feature_dict(req)
    feature_vector = extract_feature_vector(req)

    assert len(feature_dict) == 46
    assert len(feature_vector) == 46
    assert FEATURE_COUNT == 46
    print("PASS: Output contains exactly 46 values")


def test_feature_order_matches_schema():
    """Verify feature ordering strictly matches FEATURE_ORDER tuple from the schema."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    feature_dict = extract_feature_dict(req)
    feature_vector = extract_feature_vector(req)

    dict_keys = list(feature_dict.keys())
    assert dict_keys == list(FEATURE_ORDER), "Feature dict keys mismatch FEATURE_ORDER!"

    # Verify 1-to-1 correspondence between vector indices and FEATURE_ORDER
    for idx, feature_name in enumerate(FEATURE_ORDER):
        assert feature_vector[idx] == feature_dict[feature_name]

    # Verify group block ordering
    assert dict_keys[0] == "cycle"
    assert dict_keys[1:4] == ["setting_1", "setting_2", "setting_3"]
    assert dict_keys[4:25] == [f"sensor_{i:02d}" for i in range(1, 22)]
    assert dict_keys[25:46] == [f"sensor_{i:02d}_delta" for i in range(1, 22)]
    print("PASS: Feature order matches schema perfectly")


def test_machine_id_excluded():
    """Verify machine_id is never present in the extracted feature representation."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    feature_dict = extract_feature_dict(req)

    assert "machine_id" not in feature_dict
    assert "machine_id" not in FEATURE_ORDER
    print("PASS: machine_id is excluded from feature representation")


def test_values_match_original_request():
    """Verify feature values match original request values without alteration or fabricated values."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    feature_dict = extract_feature_dict(req)
    feature_vector = extract_feature_vector(req)

    assert feature_dict["cycle"] == 85.0
    assert feature_dict["setting_1"] == 0.0015
    assert feature_dict["setting_2"] == -0.0002
    assert feature_dict["setting_3"] == 100.0

    for i in range(1, 22):
        s_name = f"sensor_{i:02d}"
        d_name = f"sensor_{i:02d}_delta"
        assert feature_dict[s_name] == sample_request_data["sensors"][s_name]
        assert feature_dict[d_name] == sample_request_data["sensor_deltas"][d_name]

    # Check vector values match dict values in order
    for idx, name in enumerate(FEATURE_ORDER):
        assert feature_vector[idx] == feature_dict[name]

    print("PASS: Extracted values match original request values")


def test_invalid_non_finite_values_rejected_at_schema():
    """Verify NaN and Inf cannot enter the feature pipeline via request validation."""
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        # Operating setting
        data = copy.deepcopy(sample_request_data)
        data["operating_settings"]["setting_1"] = bad_val
        try:
            RULPredictionRequest.model_validate(data)
            raise AssertionError("Expected ValidationError for bad operating setting")
        except ValidationError:
            pass

        # Sensor
        data = copy.deepcopy(sample_request_data)
        data["sensors"]["sensor_01"] = bad_val
        try:
            RULPredictionRequest.model_validate(data)
            raise AssertionError("Expected ValidationError for bad sensor")
        except ValidationError:
            pass

        # Delta
        data = copy.deepcopy(sample_request_data)
        data["sensor_deltas"]["sensor_01_delta"] = bad_val
        try:
            RULPredictionRequest.model_validate(data)
            raise AssertionError("Expected ValidationError for bad delta")
        except ValidationError:
            pass

    print("PASS: Non-finite values rejected at schema boundary")


def test_service_level_non_finite_safety_guard():
    """Verify extract_feature_dict detects non-finite values even if bypassing Pydantic."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    # Direct dictionary mutation on the instance
    req.operating_settings["setting_1"] = float("nan")

    try:
        extract_feature_dict(req)
        raise AssertionError("Expected ValueError from service guard on NaN")
    except ValueError as e:
        assert "Non-finite value" in str(e)

    req.operating_settings["setting_1"] = 0.0015
    req.sensors["sensor_05"] = float("inf")

    try:
        extract_feature_vector(req)
        raise AssertionError("Expected ValueError from service guard on Inf")
    except ValueError as e:
        assert "Non-finite value" in str(e)

    print("PASS: Service-level non-finite safety guard verified")


if __name__ == "__main__":
    tests = [
        test_feature_count,
        test_feature_order_matches_schema,
        test_machine_id_excluded,
        test_values_match_original_request,
        test_invalid_non_finite_values_rejected_at_schema,
        test_service_level_non_finite_safety_guard,
    ]

    for t in tests:
        t()

    print(f"\nAll {len(tests)} feature service tests passed successfully!")
