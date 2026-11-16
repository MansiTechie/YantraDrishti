import copy
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.api.predictions import get_inference_service
from app.schemas.prediction import (
    PredictionStatus,
    RULPredictionResponse,
)
from app.services.inference_service import (
    InvalidPredictionError,
    RULInferenceService,
)

# Reference sample payload
sample_prediction_payload = {
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

# Reference machine registration payload
sample_machine_payload = {
    "identity": {
        "machine_id": "FD001-TEST-MACHINE-999",
        "source_dataset": "NASA_CMAPSS_FD001",
        "engine_id": 999,
    },
    "telemetry": {
        "cycle": 50,
        "operating_settings": {
            "setting_1": 0.0,
            "setting_2": 0.0,
            "setting_3": 100.0,
        },
        "sensors": {
            "sensor_01": 518.67,
            "sensor_02": 642.0,
        },
    },
    "predictions": {
        "predicted_rul_cycles": None,
        "anomaly_score": None,
        "health_status": "unknown",
    },
    "replay": {
        "status": "idle",
        "sequence_number": 50,
        "timestamp": None,
    },
}


class MockRULModel:
    """Test double for isolated API testing."""
    def __init__(self, return_val=112.5):
        self.return_val = return_val

    def predict(self, X):
        return [self.return_val]


class CrashingRULModel:
    """Test double that simulates an internal prediction computation error."""
    def predict(self, X):
        raise RuntimeError("Low-level numerical crash during dot product")


def test_application_startup_without_model_artifact():
    """Verify application starts and endpoints respond without requiring a model on disk."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["project"] == "YantraDrishti"

    health_response = client.get("/api/v1/health")
    assert health_response.status_code == status.HTTP_200_OK
    assert health_response.json()["status"] == "healthy"
    print("PASS: Application startup succeeds without model artifact")


def test_missing_model_returns_503():
    """Verify unconfigured/missing model returns HTTP 503 Service Unavailable."""
    client = TestClient(app)
    # Ensure default unconfigured service is used (no model loaded)
    app.dependency_overrides.clear()

    response = client.post("/api/v1/predictions/rul", json=sample_prediction_payload)
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert "detail" in data
    assert "unavailable" in data["detail"].lower()
    # Confirm no stack traces or filesystem paths leaked
    assert "traceback" not in response.text.lower()
    print("PASS: Missing model returns HTTP 503 with clean client message")


def test_valid_request_with_mock_returns_200():
    """Verify valid request with mock inference service returns HTTP 200 and RULPredictionResponse."""
    mock_service = RULInferenceService(
        model_instance=MockRULModel(return_val=112.5),
        model_name="FD001_LinearBaseline",
        model_version="1.0.0-test",
    )

    app.dependency_overrides[get_inference_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post("/api/v1/predictions/rul", json=sample_prediction_payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Validate against response schema
    validated_resp = RULPredictionResponse.model_validate(data)
    assert validated_resp.machine_id == "FD001-ENGINE-001"
    assert validated_resp.predicted_rul_cycles == 112.5
    assert validated_resp.unit == "cycles"
    assert validated_resp.model_name == "FD001_LinearBaseline"
    assert validated_resp.model_version == "1.0.0-test"
    assert validated_resp.prediction_status == PredictionStatus.SUCCESS

    app.dependency_overrides.clear()
    print("PASS: Valid request returns HTTP 200 and valid RULPredictionResponse")


def test_invalid_request_returns_422():
    """Verify invalid payloads return HTTP 422 Unprocessable Entity."""
    client = TestClient(app)

    # 1. Missing sensor_21
    bad_payload = copy.deepcopy(sample_prediction_payload)
    del bad_payload["sensors"]["sensor_21"]
    resp1 = client.post("/api/v1/predictions/rul", json=bad_payload)
    assert resp1.status_code == 422

    # 2. Unpadded sensor name
    bad_payload = copy.deepcopy(sample_prediction_payload)
    del bad_payload["sensors"]["sensor_01"]
    bad_payload["sensors"]["sensor_1"] = 501.0
    resp2 = client.post("/api/v1/predictions/rul", json=bad_payload)
    assert resp2.status_code == 422

    # 3. Missing sensor delta
    bad_payload = copy.deepcopy(sample_prediction_payload)
    del bad_payload["sensor_deltas"]["sensor_01_delta"]
    resp3 = client.post("/api/v1/predictions/rul", json=bad_payload)
    assert resp3.status_code == 422

    # 4. Zero cycle
    bad_payload = copy.deepcopy(sample_prediction_payload)
    bad_payload["cycle"] = 0
    resp4 = client.post("/api/v1/predictions/rul", json=bad_payload)
    assert resp4.status_code == 422

    # 5. Unexpected field
    bad_payload = copy.deepcopy(sample_prediction_payload)
    bad_payload["unexpected_field"] = "malicious"
    resp5 = client.post("/api/v1/predictions/rul", json=bad_payload)
    assert resp5.status_code == 422

    print("PASS: Invalid requests correctly return HTTP 422")


def test_unexpected_inference_failure_returns_500():
    """Verify internal inference crash returns HTTP 500 without leaking stack traces."""
    faulty_service = RULInferenceService(
        model_instance=CrashingRULModel(),
        model_name="CrashingModel",
        model_version="1.0.0",
    )

    app.dependency_overrides[get_inference_service] = lambda: faulty_service

    client = TestClient(app)
    response = client.post("/api/v1/predictions/rul", json=sample_prediction_payload)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "detail" in data
    # Confirm no stack traces or private paths leaked
    assert "traceback" not in response.text.lower()
    assert "dot product" not in response.text.lower()

    app.dependency_overrides.clear()
    print("PASS: Unexpected inference failure returns HTTP 500 without leaking details")


def test_existing_machine_endpoints_continue_to_work():
    """Verify existing machine registration and retrieval endpoints are fully operational."""
    client = TestClient(app)

    # 1. Register a machine
    reg_resp = client.post("/api/v1/machines", json=sample_machine_payload)
    assert reg_resp.status_code == status.HTTP_201_CREATED
    assert reg_resp.json()["identity"]["machine_id"] == "FD001-TEST-MACHINE-999"

    # 2. Retrieve all machines
    list_resp = client.get("/api/v1/machines")
    assert list_resp.status_code == status.HTTP_200_OK
    machine_ids = [m["identity"]["machine_id"] for m in list_resp.json()]
    assert "FD001-TEST-MACHINE-999" in machine_ids

    # 3. Retrieve single machine
    get_resp = client.get("/api/v1/machines/FD001-TEST-MACHINE-999")
    assert get_resp.status_code == status.HTTP_200_OK
    assert get_resp.json()["identity"]["machine_id"] == "FD001-TEST-MACHINE-999"

    # 4. Duplicate registration 409
    dup_resp = client.post("/api/v1/machines", json=sample_machine_payload)
    assert dup_resp.status_code == status.HTTP_409_CONFLICT

    # 5. Non-existent machine 404
    not_found_resp = client.get("/api/v1/machines/DOES_NOT_EXIST")
    assert not_found_resp.status_code == status.HTTP_404_NOT_FOUND

    print("PASS: Existing machine registration and retrieval endpoints work seamlessly")


def test_health_and_root_endpoints_continue_to_work():
    """Verify health and root endpoints are fully operational."""
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == status.HTTP_200_OK
    assert root_resp.json()["status"] if "status" in root_resp.json() else root_resp.json()["project"] == "YantraDrishti"

    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == status.HTTP_200_OK
    assert health_resp.json()["status"] == "healthy"

    print("PASS: Health and root endpoints work seamlessly")


if __name__ == "__main__":
    tests = [
        test_application_startup_without_model_artifact,
        test_missing_model_returns_503,
        test_valid_request_with_mock_returns_200,
        test_invalid_request_returns_422,
        test_unexpected_inference_failure_returns_500,
        test_existing_machine_endpoints_continue_to_work,
        test_health_and_root_endpoints_continue_to_work,
    ]

    for t in tests:
        t()

    print(f"\nAll {len(tests)} prediction API tests passed successfully!")
