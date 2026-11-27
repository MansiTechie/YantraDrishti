import copy
import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure isolated SQLite in-memory database for tests
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import create_db_engine, get_db
from app.main import app
from app.models.machine import MachineModel
from app.repositories.machine_repository import (
    DatabaseUnavailableError,
    DuplicateMachineError,
    MachineNotFoundError,
    MachineRepository,
    get_machine_repository,
)
from app.schemas.machine import HealthStatus, MachineState, ReplayStatus

sample_machine_payload = {
    "identity": {
        "machine_id": "FD001-TEST-001",
        "source_dataset": "NASA_CMAPSS_FD001",
        "engine_id": 1,
    },
    "telemetry": {
        "cycle": 100,
        "operating_settings": {
            "setting_1": -0.0007,
            "setting_2": -0.0004,
            "setting_3": 100.0,
        },
        "sensors": {
            "sensor_01": 518.67,
            "sensor_02": 641.82,
            "sensor_03": 1589.70,
        },
    },
    "predictions": {
        "predicted_rul_cycles": 112.5,
        "anomaly_score": 0.02,
        "health_status": "normal",
    },
    "replay": {
        "status": "running",
        "sequence_number": 100,
        "timestamp": "2026-11-28T00:00:00Z",
    },
}


def create_isolated_test_db():
    """Creates a fresh, isolated in-memory SQLite engine and session factory."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    return test_engine, TestingSessionLocal


def test_db_engine_and_tables_creation():
    """Verify engine creation and table creation in database."""
    test_engine, _ = create_isolated_test_db()
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()

    assert "machines" in tables, f"Expected 'machines' table in {tables}"

    columns = {col["name"] for col in inspector.get_columns("machines")}
    expected_cols = {
        "id",
        "machine_id",
        "source_dataset",
        "engine_id",
        "cycle",
        "operating_settings",
        "sensors",
        "predicted_rul_cycles",
        "anomaly_score",
        "health_status",
        "replay_status",
        "sequence_number",
        "replay_timestamp",
        "created_at",
        "updated_at",
    }
    assert expected_cols.issubset(columns), f"Missing columns: {expected_cols - columns}"
    print("PASS: Engine and 'machines' table created with expected schema")


def test_machine_model_schema_conversions():
    """Verify bidirectional conversions between Pydantic MachineState and SQLAlchemy MachineModel."""
    state = MachineState.model_validate(sample_machine_payload)
    model = MachineModel.from_schema(state)

    assert model.machine_id == "FD001-TEST-001"
    assert model.engine_id == 1
    assert model.cycle == 100
    assert model.operating_settings["setting_3"] == 100.0
    assert model.sensors["sensor_01"] == 518.67
    assert model.predicted_rul_cycles == 112.5
    assert model.anomaly_score == 0.02
    assert model.health_status == "normal"
    assert model.replay_status == "running"
    assert model.sequence_number == 100

    # Test to_dict and to_schema
    state_converted = model.to_schema()
    assert state_converted.identity.machine_id == state.identity.machine_id
    assert state_converted.telemetry.cycle == state.telemetry.cycle
    assert state_converted.predictions.predicted_rul_cycles == 112.5
    assert state_converted.predictions.health_status == HealthStatus.NORMAL
    assert state_converted.replay.status == ReplayStatus.RUNNING

    print("PASS: MachineModel bidirectional conversion to/from MachineState verified")


def test_repository_create_and_get_by_id():
    """Verify repository creates and retrieves machine records."""
    _, SessionLocal = create_isolated_test_db()
    db = SessionLocal()
    try:
        repo = MachineRepository(db)
        state = MachineState.model_validate(sample_machine_payload)

        created = repo.create(state)
        assert created.id is not None
        assert created.machine_id == "FD001-TEST-001"

        retrieved = repo.get_by_id("FD001-TEST-001")
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.machine_id == "FD001-TEST-001"
        assert retrieved.cycle == 100
    finally:
        db.close()

    print("PASS: Repository create and get_by_id successfully persists and retrieves machine")


def test_repository_duplicate_machine_rejection():
    """Verify repository raises DuplicateMachineError when creating an existing machine_id."""
    _, SessionLocal = create_isolated_test_db()
    db = SessionLocal()
    try:
        repo = MachineRepository(db)
        state = MachineState.model_validate(sample_machine_payload)
        repo.create(state)

        # Attempt duplicate insertion
        try:
            repo.create(state)
            raise AssertionError("Duplicate insertion should have raised DuplicateMachineError")
        except DuplicateMachineError:
            pass
    finally:
        db.close()

    print("PASS: Repository duplicate machine_id insertion correctly raises DuplicateMachineError")


def test_repository_get_all():
    """Verify repository lists machines with pagination support."""
    _, SessionLocal = create_isolated_test_db()
    db = SessionLocal()
    try:
        repo = MachineRepository(db)
        for i in range(1, 6):
            payload = copy.deepcopy(sample_machine_payload)
            payload["identity"]["machine_id"] = f"FD001-TEST-{i:03d}"
            payload["identity"]["engine_id"] = i
            repo.create(MachineState.model_validate(payload))

        all_machines = repo.get_all()
        assert len(all_machines) == 5

        paginated = repo.get_all(skip=2, limit=2)
        assert len(paginated) == 2
        assert paginated[0].machine_id == "FD001-TEST-003"
    finally:
        db.close()

    print("PASS: Repository get_all returns expected records with pagination")


def test_repository_update():
    """Verify repository updates an existing machine record."""
    _, SessionLocal = create_isolated_test_db()
    db = SessionLocal()
    try:
        repo = MachineRepository(db)
        state = MachineState.model_validate(sample_machine_payload)
        repo.create(state)

        # Update telemetry and predictions
        updated_payload = copy.deepcopy(sample_machine_payload)
        updated_payload["telemetry"]["cycle"] = 150
        updated_payload["predictions"]["predicted_rul_cycles"] = 75.0
        updated_state = MachineState.model_validate(updated_payload)

        updated_model = repo.update("FD001-TEST-001", updated_state)
        assert updated_model.cycle == 150
        assert updated_model.predicted_rul_cycles == 75.0

        retrieved = repo.get_by_id("FD001-TEST-001")
        assert retrieved.cycle == 150
        assert retrieved.predicted_rul_cycles == 75.0
    finally:
        db.close()

    print("PASS: Repository update correctly updates machine in place")


def test_repository_delete():
    """Verify repository deletes a machine record and handles non-existent machine."""
    _, SessionLocal = create_isolated_test_db()
    db = SessionLocal()
    try:
        repo = MachineRepository(db)
        state = MachineState.model_validate(sample_machine_payload)
        repo.create(state)

        deleted = repo.delete("FD001-TEST-001")
        assert deleted is True
        assert repo.get_by_id("FD001-TEST-001") is None

        # Delete non-existent
        deleted_again = repo.delete("FD001-TEST-001")
        assert deleted_again is False
    finally:
        db.close()

    print("PASS: Repository delete removes record and safely returns False for missing machine")


def test_repository_missing_machine_handling():
    """Verify repository update on missing machine raises MachineNotFoundError."""
    _, SessionLocal = create_isolated_test_db()
    db = SessionLocal()
    try:
        repo = MachineRepository(db)
        state = MachineState.model_validate(sample_machine_payload)
        try:
            repo.update("NON-EXISTENT", state)
            raise AssertionError("Update on non-existent machine should have raised MachineNotFoundError")
        except MachineNotFoundError:
            pass
    finally:
        db.close()

    print("PASS: Repository missing machine update raises MachineNotFoundError")


def test_api_endpoints_with_sqlite_db():
    """Verify FastAPI machine endpoints persist and retrieve from SQLite in-memory DB."""
    test_engine, TestingSessionLocal = create_isolated_test_db()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        # 1. Register new machine (HTTP 201)
        resp1 = client.post("/api/v1/machines", json=sample_machine_payload)
        assert resp1.status_code == status.HTTP_201_CREATED
        data1 = resp1.json()
        assert data1["identity"]["machine_id"] == "FD001-TEST-001"
        assert data1["telemetry"]["cycle"] == 100

        # 2. Retrieve all machines (HTTP 200)
        resp2 = client.get("/api/v1/machines")
        assert resp2.status_code == status.HTTP_200_OK
        data2 = resp2.json()
        assert len(data2) == 1
        assert data2[0]["identity"]["machine_id"] == "FD001-TEST-001"

        # 3. Retrieve single machine (HTTP 200)
        resp3 = client.get("/api/v1/machines/FD001-TEST-001")
        assert resp3.status_code == status.HTTP_200_OK
        data3 = resp3.json()
        assert data3["identity"]["machine_id"] == "FD001-TEST-001"
        assert data3["predictions"]["predicted_rul_cycles"] == 112.5

        # 4. Duplicate registration returns HTTP 409
        resp4 = client.post("/api/v1/machines", json=sample_machine_payload)
        assert resp4.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in resp4.json()["detail"]

        # 5. Non-existent machine returns HTTP 404
        resp5 = client.get("/api/v1/machines/NON-EXISTENT-ID")
        assert resp5.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in resp5.json()["detail"].lower()

        # 6. Invalid payload returns HTTP 422
        bad_payload = copy.deepcopy(sample_machine_payload)
        bad_payload["telemetry"]["cycle"] = -5
        resp6 = client.post("/api/v1/machines", json=bad_payload)
        assert resp6.status_code == 422

    finally:
        app.dependency_overrides.pop(get_db, None)

    print("PASS: Machine API CRUD, duplicate 409, 404, and 422 validations verified with SQLite")


def test_database_unavailable_returns_503():
    """Verify API returns HTTP 503 Service Unavailable with clear message when DB fails."""
    class FailingRepo:
        def create(self, machine):
            raise DatabaseUnavailableError("Database connection failed. Verify service is running.")

        def get_all(self):
            raise DatabaseUnavailableError("Database connection failed. Verify service is running.")

        def get_by_id(self, machine_id):
            raise DatabaseUnavailableError("Database connection failed. Verify service is running.")

    app.dependency_overrides[get_machine_repository] = lambda: FailingRepo()
    client = TestClient(app)

    try:
        resp_post = client.post("/api/v1/machines", json=sample_machine_payload)
        assert resp_post.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "database connection failed" in resp_post.json()["detail"].lower()

        resp_get_all = client.get("/api/v1/machines")
        assert resp_get_all.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

        resp_get_one = client.get("/api/v1/machines/TEST")
        assert resp_get_one.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    finally:
        app.dependency_overrides.pop(get_machine_repository, None)

    print("PASS: Unreachable database properly maps to HTTP 503 without leaking stack trace")


def test_session_lifecycle_and_cleanup():
    """Verify that get_db dependency yields active session and closes it cleanly."""
    closed = False

    class MockSession:
        def close(self):
            nonlocal closed
            closed = True

    from app.db.session import get_db

    # Simulate dependency generator lifecycle
    def mock_session_local():
        return MockSession()

    import app.db.session as session_module
    orig_session_local = session_module.SessionLocal
    session_module.SessionLocal = mock_session_local

    try:
        gen = get_db()
        sess = next(gen)
        assert sess is not None
        assert closed is False
        try:
            next(gen)
        except StopIteration:
            pass
        assert closed is True
    finally:
        session_module.SessionLocal = orig_session_local

    print("PASS: Database session lifecycle correctly ensures session closure on exit")


def test_database_url_configuration_and_error_handling():
    """Verify DATABASE_URL configuration loading and actionable ConfigurationError when missing."""
    from app.core.config import ConfigurationError, Settings

    orig_url = os.environ.get("DATABASE_URL")
    try:
        # 1. Verify missing DATABASE_URL raises actionable ConfigurationError
        os.environ["DATABASE_URL"] = ""
        s = Settings()
        try:
            _ = s.DATABASE_URL
            raise AssertionError("Empty DATABASE_URL should have raised ConfigurationError")
        except ConfigurationError as exc:
            assert "DATABASE_URL is not configured" in str(exc)
            assert "postgresql+psycopg2" in str(exc)
            print("PASS: Missing DATABASE_URL correctly raises actionable ConfigurationError")

        # 2. Verify explicit environment DATABASE_URL is correctly loaded
        test_custom_url = "sqlite:///:memory:"
        os.environ["DATABASE_URL"] = test_custom_url
        assert s.DATABASE_URL == test_custom_url
        print("PASS: DATABASE_URL is correctly loaded when supplied through environment")

    finally:
        if orig_url is not None:
            os.environ["DATABASE_URL"] = orig_url


if __name__ == "__main__":
    tests = [
        test_database_url_configuration_and_error_handling,
        test_db_engine_and_tables_creation,
        test_machine_model_schema_conversions,
        test_repository_create_and_get_by_id,
        test_repository_duplicate_machine_rejection,
        test_repository_get_all,
        test_repository_update,
        test_repository_delete,
        test_repository_missing_machine_handling,
        test_api_endpoints_with_sqlite_db,
        test_database_unavailable_returns_503,
        test_session_lifecycle_and_cleanup,
    ]

    for t in tests:
        t()

    print(f"\nAll {len(tests)} database tests passed successfully!")
