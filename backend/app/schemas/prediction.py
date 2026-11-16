import math
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

EXPECTED_OPERATING_SETTINGS = (
    "setting_1",
    "setting_2",
    "setting_3",
)

EXPECTED_SENSORS = tuple(f"sensor_{i:02d}" for i in range(1, 22))

EXPECTED_SENSOR_DELTAS = tuple(f"sensor_{i:02d}_delta" for i in range(1, 22))

# Canonical 46-feature model input ordering
FEATURE_ORDER = (
    ("cycle",)
    + EXPECTED_OPERATING_SETTINGS
    + EXPECTED_SENSORS
    + EXPECTED_SENSOR_DELTAS
)


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class RULPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_id: str = Field(min_length=1)
    cycle: int = Field(gt=0)

    operating_settings: dict[str, float] = Field(
        min_length=3,
        max_length=3,
    )

    sensors: dict[str, float] = Field(
        min_length=21,
        max_length=21,
    )

    sensor_deltas: dict[str, float] = Field(
        min_length=21,
        max_length=21,
    )

    @field_validator("operating_settings")
    @classmethod
    def validate_operating_settings(cls, value):
        expected = set(EXPECTED_OPERATING_SETTINGS)

        if set(value.keys()) != expected:
            missing = expected - set(value.keys())
            extra = set(value.keys()) - expected
            raise ValueError(
                f"Operating settings must contain exactly setting_1, setting_2, and setting_3. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        for name, reading in value.items():
            if not math.isfinite(reading):
                raise ValueError(
                    f"Non-finite value in operating setting '{name}': {reading}"
                )

        return value

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, value):
        expected = set(EXPECTED_SENSORS)

        if set(value.keys()) != expected:
            missing = expected - set(value.keys())
            extra = set(value.keys()) - expected
            raise ValueError(
                f"Sensors must contain exactly sensor_01 through sensor_21. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        for name, reading in value.items():
            if not math.isfinite(reading):
                raise ValueError(
                    f"Non-finite reading in sensor '{name}': {reading}"
                )

        return value

    @field_validator("sensor_deltas")
    @classmethod
    def validate_sensor_deltas(cls, value):
        expected = set(EXPECTED_SENSOR_DELTAS)

        if set(value.keys()) != expected:
            missing = expected - set(value.keys())
            extra = set(value.keys()) - expected
            raise ValueError(
                f"Sensor deltas must contain exactly sensor_01_delta through sensor_21_delta. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        for name, reading in value.items():
            if not math.isfinite(reading):
                raise ValueError(
                    f"Non-finite reading in sensor delta '{name}': {reading}"
                )

        return value

    def to_feature_dict(self) -> dict[str, float]:
        """Convert request into canonical 46-feature ordered dictionary.

        Excludes machine_id. Order strictly matches Day 2 split manifest:
        1. cycle
        2. setting_1, setting_2, setting_3
        3. sensor_01 through sensor_21
        4. sensor_01_delta through sensor_21_delta
        """
        features: dict[str, float] = {"cycle": float(self.cycle)}

        for setting in EXPECTED_OPERATING_SETTINGS:
            features[setting] = float(self.operating_settings[setting])

        for sensor in EXPECTED_SENSORS:
            features[sensor] = float(self.sensors[sensor])

        for delta in EXPECTED_SENSOR_DELTAS:
            features[delta] = float(self.sensor_deltas[delta])

        return features

    def to_feature_vector(self) -> list[float]:
        """Convert request into canonical 46-feature ordered float list."""
        return list(self.to_feature_dict().values())


class RULPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_id: str = Field(min_length=1)
    predicted_rul_cycles: float = Field(ge=0, allow_inf_nan=False)
    unit: Literal["cycles"] = "cycles"
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prediction_status: PredictionStatus = PredictionStatus.SUCCESS
