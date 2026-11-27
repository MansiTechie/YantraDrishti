from datetime import datetime
from typing import Any, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    func,
)

from app.db.base import Base
from app.schemas.machine import MachineState


class MachineModel(Base):
    """
    SQLAlchemy ORM Model representing a monitored industrial asset (machine).
    Persists identity, telemetry, RUL predictions, and replay metadata.
    """

    __tablename__ = "machines"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Machine Identity
    machine_id = Column(String(128), unique=True, nullable=False, index=True)
    source_dataset = Column(String(64), nullable=False)
    engine_id = Column(Integer, nullable=False, index=True)

    # Telemetry
    cycle = Column(Integer, nullable=False)
    operating_settings = Column(JSON, nullable=False)
    sensors = Column(JSON, nullable=False)

    # Predictions
    predicted_rul_cycles = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    health_status = Column(String(32), nullable=False, default="unknown")

    # Replay Metadata
    replay_status = Column(String(32), nullable=False, default="idle")
    sequence_number = Column(Integer, nullable=False, default=0)
    replay_timestamp = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Maps ORM instance fields to the nested structure expected by MachineState schema."""
        return {
            "identity": {
                "machine_id": self.machine_id,
                "source_dataset": self.source_dataset,
                "engine_id": self.engine_id,
            },
            "telemetry": {
                "cycle": self.cycle,
                "operating_settings": self.operating_settings,
                "sensors": self.sensors,
            },
            "predictions": {
                "predicted_rul_cycles": self.predicted_rul_cycles,
                "anomaly_score": self.anomaly_score,
                "health_status": self.health_status,
            },
            "replay": {
                "status": self.replay_status,
                "sequence_number": self.sequence_number,
                "timestamp": self.replay_timestamp,
            },
        }

    def to_schema(self) -> MachineState:
        """Validates and returns a Pydantic MachineState instance."""
        return MachineState.model_validate(self.to_dict())

    @classmethod
    def from_schema(cls, schema: MachineState) -> "MachineModel":
        """Instantiates an ORM model instance from a validated Pydantic MachineState."""
        return cls(
            machine_id=schema.identity.machine_id,
            source_dataset=schema.identity.source_dataset,
            engine_id=schema.identity.engine_id,
            cycle=schema.telemetry.cycle,
            operating_settings=schema.telemetry.operating_settings,
            sensors=schema.telemetry.sensors,
            predicted_rul_cycles=schema.predictions.predicted_rul_cycles,
            anomaly_score=schema.predictions.anomaly_score,
            health_status=(
                schema.predictions.health_status.value
                if hasattr(schema.predictions.health_status, "value")
                else str(schema.predictions.health_status)
            ),
            replay_status=(
                schema.replay.status.value
                if hasattr(schema.replay.status, "value")
                else str(schema.replay.status)
            ),
            sequence_number=schema.replay.sequence_number,
            replay_timestamp=schema.replay.timestamp,
        )

    def update_from_schema(self, schema: MachineState) -> None:
        """Updates model fields in place from a validated Pydantic MachineState."""
        self.source_dataset = schema.identity.source_dataset
        self.engine_id = schema.identity.engine_id
        self.cycle = schema.telemetry.cycle
        self.operating_settings = schema.telemetry.operating_settings
        self.sensors = schema.telemetry.sensors
        self.predicted_rul_cycles = schema.predictions.predicted_rul_cycles
        self.anomaly_score = schema.predictions.anomaly_score
        self.health_status = (
            schema.predictions.health_status.value
            if hasattr(schema.predictions.health_status, "value")
            else str(schema.predictions.health_status)
        )
        self.replay_status = (
            schema.replay.status.value
            if hasattr(schema.replay.status, "value")
            else str(schema.replay.status)
        )
        self.sequence_number = schema.replay.sequence_number
        self.replay_timestamp = schema.replay.timestamp
