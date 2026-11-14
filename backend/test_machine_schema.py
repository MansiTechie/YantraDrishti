
import copy

from pydantic import ValidationError

from app.schemas.machine import (
    MachineState,
    HealthStatus,
    ReplayStatus,
)


sample_machine = {
    "identity": {
        "machine_id": "FD001-ENGINE-001",
        "source_dataset": "NASA_CMAPSS_FD001",
        "engine_id": 1,
    },
    "telemetry": {
        "cycle": 100,
        "operating_settings": {
            "setting_1": 0.1,
            "setting_2": 0.2,
            "setting_3": 0.3,
        },
        "sensors": {
            "sensor_01": 518.67,
            "sensor_02": 641.82,
        },
    },
    "predictions": {
        "predicted_rul_cycles": None,
        "anomaly_score": None,
        "health_status": HealthStatus.UNKNOWN,
    },
    "replay": {
        "status": ReplayStatus.IDLE,
        "sequence_number": 100,
        "timestamp": None,
    },
}


def expect_validation_error(data, test_name):
    try:
        MachineState.model_validate(data)
    except ValidationError:
        print(f"PASS: {test_name}")
        return

    raise AssertionError(f"FAIL: {test_name}")


def test_valid_machine_state():
    machine = MachineState.model_validate(sample_machine)

    assert machine.identity.engine_id == 1
    assert machine.telemetry.cycle == 100

    print("PASS: Valid machine state accepted")


def test_invalid_cycle():
    data = copy.deepcopy(sample_machine)
    data["telemetry"]["cycle"] = -1

    expect_validation_error(data, "Invalid cycle rejected")


def test_invalid_sensor_name():
    data = copy.deepcopy(sample_machine)
    data["telemetry"]["sensors"] = {
        "sensor_99": 500.0
    }

    expect_validation_error(data, "Invalid sensor name rejected")


def test_missing_operating_setting():
    data = copy.deepcopy(sample_machine)
    del data["telemetry"]["operating_settings"]["setting_3"]

    expect_validation_error(
        data,
        "Missing operating setting rejected",
    )


def test_nan_sensor_reading():
    data = copy.deepcopy(sample_machine)
    data["telemetry"]["sensors"]["sensor_01"] = float("nan")

    expect_validation_error(
        data,
        "NaN sensor reading rejected",
    )


def test_infinite_sensor_reading():
    data = copy.deepcopy(sample_machine)
    data["telemetry"]["sensors"]["sensor_01"] = float("inf")

    expect_validation_error(
        data,
        "Infinite sensor reading rejected",
    )


def test_negative_rul():
    data = copy.deepcopy(sample_machine)
    data["predictions"]["predicted_rul_cycles"] = -10

    expect_validation_error(data, "Negative RUL rejected")


def test_infinite_anomaly_score():
    data = copy.deepcopy(sample_machine)
    data["predictions"]["anomaly_score"] = float("inf")

    expect_validation_error(
        data,
        "Infinite anomaly score rejected",
    )


if __name__ == "__main__":
    tests = [
        test_valid_machine_state,
        test_invalid_cycle,
        test_invalid_sensor_name,
        test_missing_operating_setting,
        test_nan_sensor_reading,
        test_infinite_sensor_reading,
        test_negative_rul,
        test_infinite_anomaly_score,
    ]

    for test in tests:
        test()

    print(f"\nAll {len(tests)} schema tests passed!")
