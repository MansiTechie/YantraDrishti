
import math
import re

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class HealthStatus(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class ReplayStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class MachineIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_id: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    engine_id: int = Field(gt=0)


class MachineTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(gt=0)

    operating_settings: dict[str, float] = Field(
        min_length=3,
        max_length=3,
    )

    sensors: dict[str, float] = Field(
        min_length=1,
        max_length=21,
    )

    @field_validator("operating_settings")
    @classmethod
    def validate_operating_settings(cls, value):
        expected = {
            "setting_1",
            "setting_2",
            "setting_3",
        }

        if set(value.keys()) != expected:
            raise ValueError(
                "Operating settings must contain "
                "setting_1, setting_2 and setting_3"
            )

        for name, reading in value.items():
            if not math.isfinite(reading):
                raise ValueError(
                    f"Non-finite value in {name}"
                )

        return value

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, value):
        if not value:
            raise ValueError(
                "At least one sensor reading is required"
            )

        for name, reading in value.items():
            if not re.fullmatch(r"sensor_(0[1-9]|1[0-9]|2[01])", name):
                raise ValueError(
                    f"Invalid C-MAPSS sensor name: {name}"
                )

            if not math.isfinite(reading):
                raise ValueError(
                    f"Non-finite reading for {name}"
                )

        return value


class MachinePredictions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_rul_cycles: Optional[float] = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    anomaly_score: Optional[float] = Field(
        default=None,
        allow_inf_nan=False,
    )

    health_status: HealthStatus = HealthStatus.UNKNOWN


class ReplayMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReplayStatus = ReplayStatus.IDLE
    sequence_number: int = Field(ge=0)
    timestamp: Optional[datetime] = None


class MachineState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: MachineIdentity
    telemetry: MachineTelemetry
    predictions: MachinePredictions
    replay: ReplayMetadata
