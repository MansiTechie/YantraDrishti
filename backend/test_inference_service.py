import copy
import math
import tempfile
from pathlib import Path

from app.schemas.prediction import (
    FEATURE_ORDER,
    PredictionStatus,
    RULPredictionRequest,
    RULPredictionResponse,
)
from app.services.feature_service import extract_feature_vector
from app.services.inference_service import (
    InvalidPredictionError,
    ModelUnavailableError,
    RULInferenceService,
)

# Reference sample payload
sample_request_data = {
    "machine_id": "FD001-ENGINE-007",
    "cycle": 120,
    "operating_settings": {
        "setting_1": -0.0002,
        "setting_2": 0.0001,
        "setting_3": 100.0,
    },
    "sensors": {f"sensor_{i:02d}": 510.0 + i for i in range(1, 22)},
    "sensor_deltas": {f"sensor_{i:02d}_delta": -0.05 * i for i in range(1, 22)},
}


class MockRULModel:
    """Test double implementing model.predict() for isolated unit testing."""

    def __init__(self, return_val=105.0):
        self.return_val = return_val
        self.received_input = None
        self.call_count = 0

    def predict(self, X):
        self.received_input = X
        self.call_count += 1
        return [self.return_val]


class FaultyPredictModel:
    """Test double that raises an error inside predict()."""

    def predict(self, X):
        raise RuntimeError("Internal numerical error in estimator")


def test_default_initialization_does_not_crash():
    """Verify instantiating the service without configured paths does not crash."""
    service = RULInferenceService()
    assert not service.is_available
    assert service.artifact_path is None
    print("PASS: Unconfigured service instantiates cleanly without crashing")


def test_missing_model_artifact_raises_error():
    """Verify invoking predict() when artifact is missing raises ModelUnavailableError."""
    req = RULPredictionRequest.model_validate(sample_request_data)

    # 1. No path configured
    service_no_path = RULInferenceService()
    try:
        service_no_path.predict(req)
        raise AssertionError("Expected ModelUnavailableError when path is unconfigured")
    except ModelUnavailableError as e:
        assert "not configured" in str(e).lower()

    # 2. Path points to non-existent file
    service_bad_path = RULInferenceService(
        model_artifact_path="non_existent_artifacts/model_fd001.pkl",
        model_name="FD001_LinearBaseline",
        model_version="1.0.0",
    )
    try:
        service_bad_path.predict(req)
        raise AssertionError("Expected ModelUnavailableError when artifact file is missing")
    except ModelUnavailableError as e:
        assert "not found" in str(e).lower()

    print("PASS: Missing model artifact correctly raises ModelUnavailableError")


def test_invalid_model_artifact_raises_error():
    """Verify loading a corrupt file or an object lacking predict() raises ModelUnavailableError."""
    req = RULPredictionRequest.model_validate(sample_request_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a corrupt non-pickle file
        corrupt_file = Path(tmp_dir) / "corrupt_model.pkl"
        corrupt_file.write_text("NOT_A_VALID_PICKLE_OR_JOBLIB_STREAM")

        service = RULInferenceService(
            model_artifact_path=corrupt_file,
            model_name="FD001_LinearBaseline",
            model_version="1.0.0",
        )

        try:
            service.predict(req)
            raise AssertionError("Expected ModelUnavailableError for corrupt artifact")
        except ModelUnavailableError as e:
            assert "Failed to load model artifact" in str(e)

        # Create a file containing an object without a predict() method
        import pickle
        non_model_file = Path(tmp_dir) / "non_model.pkl"
        with open(non_model_file, "wb") as f:
            pickle.dump(["not", "a", "model"], f)

        service_non_model = RULInferenceService(
            model_artifact_path=non_model_file,
            model_name="FD001_LinearBaseline",
            model_version="1.0.0",
        )

        try:
            service_non_model.predict(req)
            raise AssertionError("Expected ModelUnavailableError for object without predict()")
        except ModelUnavailableError as e:
            assert "does not provide a callable 'predict' method" in str(e)

    print("PASS: Invalid model artifact correctly raises ModelUnavailableError")


def test_valid_model_prediction_with_mock():
    """Verify prediction workflow using a controlled test double."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    mock_model = MockRULModel(return_val=105.0)

    service = RULInferenceService(
        model_instance=mock_model,
        model_name="FD001_LinearBaseline",
        model_version="1.0.0-test",
    )

    assert service.is_available

    response = service.predict(req)

    assert isinstance(response, RULPredictionResponse)
    assert response.machine_id == "FD001-ENGINE-007"
    assert response.predicted_rul_cycles == 105.0
    assert response.unit == "cycles"
    assert response.model_name == "FD001_LinearBaseline"
    assert response.model_version == "1.0.0-test"
    assert response.prediction_status == PredictionStatus.SUCCESS
    assert mock_model.call_count == 1
    print("PASS: Valid model prediction returns expected RULPredictionResponse")


def test_correct_46_feature_input_ordering():
    """Verify model.predict() receives 2D input with all 46 features in canonical order."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    expected_features = extract_feature_vector(req)

    mock_model = MockRULModel(return_val=50.0)
    service = RULInferenceService(
        model_instance=mock_model,
        model_name="FD001_LinearBaseline",
        model_version="1.0.0-test",
    )

    service.predict(req)

    received_X = mock_model.received_input
    assert received_X is not None, "Model predict() was not called"
    assert len(received_X) == 1, "Expected single-sample batch [1, 46]"
    sample_features = received_X[0]

    assert len(sample_features) == 46, f"Expected 46 features, got {len(sample_features)}"
    assert sample_features == expected_features, "Passed features deviate from canonical ordering!"
    print("PASS: Exact 46-feature input vector passed to model.predict() in canonical order")


def test_non_finite_prediction_rejection():
    """Verify non-finite model predictions (NaN, Inf) are strictly rejected."""
    req = RULPredictionRequest.model_validate(sample_request_data)

    for bad_pred in [float("nan"), float("inf"), float("-inf")]:
        mock_model = MockRULModel(return_val=bad_pred)
        service = RULInferenceService(
            model_instance=mock_model,
            model_name="FD001_LinearBaseline",
            model_version="1.0.0",
        )

        try:
            service.predict(req)
            raise AssertionError(f"Expected InvalidPredictionError for prediction: {bad_pred}")
        except InvalidPredictionError as e:
            assert "non-finite" in str(e).lower()

    print("PASS: Non-finite predictions strictly rejected with InvalidPredictionError")


def test_negative_prediction_rejection():
    """Verify negative RUL predictions are strictly rejected."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    mock_model = MockRULModel(return_val=-12.5)

    service = RULInferenceService(
        model_instance=mock_model,
        model_name="FD001_LinearBaseline",
        model_version="1.0.0",
    )

    try:
        service.predict(req)
        raise AssertionError("Expected InvalidPredictionError for negative prediction")
    except InvalidPredictionError as e:
        assert "negative" in str(e).lower()

    print("PASS: Negative RUL predictions strictly rejected with InvalidPredictionError")


def test_model_metadata_handling():
    """Verify model_name and model_version are strictly required and uninvented."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    mock_model = MockRULModel(return_val=80.0)

    # Missing model_name
    service_no_name = RULInferenceService(
        model_instance=mock_model,
        model_name=None,
        model_version="1.0.0",
    )
    try:
        service_no_name.predict(req)
        raise AssertionError("Expected ModelUnavailableError when model_name is missing")
    except ModelUnavailableError as e:
        assert "metadata" in str(e).lower()

    # Missing model_version
    service_no_ver = RULInferenceService(
        model_instance=mock_model,
        model_name="FD001_LinearBaseline",
        model_version=None,
    )
    try:
        service_no_ver.predict(req)
        raise AssertionError("Expected ModelUnavailableError when model_version is missing")
    except ModelUnavailableError as e:
        assert "metadata" in str(e).lower()

    print("PASS: Verified model metadata enforcement passed")


def test_prediction_execution_failure():
    """Verify model internal exception is wrapped in InvalidPredictionError."""
    req = RULPredictionRequest.model_validate(sample_request_data)
    service = RULInferenceService(
        model_instance=FaultyPredictModel(),
        model_name="FaultyModel",
        model_version="1.0.0",
    )

    try:
        service.predict(req)
        raise AssertionError("Expected InvalidPredictionError on model execution crash")
    except InvalidPredictionError as e:
        assert "failed" in str(e).lower()

    print("PASS: Internal model execution failure correctly wrapped in InvalidPredictionError")


if __name__ == "__main__":
    tests = [
        test_default_initialization_does_not_crash,
        test_missing_model_artifact_raises_error,
        test_invalid_model_artifact_raises_error,
        test_valid_model_prediction_with_mock,
        test_correct_46_feature_input_ordering,
        test_non_finite_prediction_rejection,
        test_negative_prediction_rejection,
        test_model_metadata_handling,
        test_prediction_execution_failure,
    ]

    for t in tests:
        t()

    print(f"\nAll {len(tests)} inference service tests passed successfully!")
