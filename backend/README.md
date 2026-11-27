# YantraDrishti — Backend Service

FastAPI backend and machine learning inference engine for the YantraDrishti predictive maintenance digital twin platform.

---

## Architecture Overview

```text
backend/
├── app/
│   ├── api/
│   │   ├── machines.py              # Machine registration and query routes
│   │   └── predictions.py           # RUL inference API endpoint (/api/v1/predictions/rul)
│   ├── core/
│   │   └── config.py                # Environment configuration (DATABASE_URL, MODEL_ARTIFACT_PATH)
│   ├── db/
│   │   ├── base.py                  # DeclarativeBase SQLAlchemy 2.x class
│   │   └── session.py               # Engine creation, SessionLocal, get_db dependency, init_db()
│   ├── models/
│   │   └── machine.py               # MachineModel SQLAlchemy ORM model ('machines' table)
│   ├── repositories/
│   │   └── machine_repository.py    # Database query operations and exception handling
│   ├── schemas/
│   │   ├── machine.py               # Strict Pydantic v2 schemas for asset telemetry and state
│   │   └── prediction.py            # Strict Pydantic v2 schemas for RUL prediction requests/responses
│   ├── services/
│   │   ├── feature_service.py       # Canonical 46-feature extraction service
│   │   └── inference_service.py     # RUL model loading, verification, and prediction service
│   └── main.py                      # FastAPI application entrypoint with lifespan DB initialization
├── requirements.txt                 # Exact pinned backend dependencies
└── test_*.py                        # Standalone unit and integration test suites
```

---

## Database Persistence (Day 4)

The backend uses **SQLAlchemy 2.x** with a dedicated Repository pattern separating API routes from database operations.

### Supported Databases

1. **Production / Local Development:** PostgreSQL with `psycopg2-binary`.
2. **Automated Testing / CI:** In-memory SQLite (`sqlite:///:memory:`) using `StaticPool` for isolated, fast execution without external infrastructure dependencies.

### Database Configuration

Environment variables should be placed in a `.env` file at the repository root:

```env
# Production / Local PostgreSQL
# Format: postgresql+psycopg2://<user>:<password>@<host>:<port>/<database>
DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/yantradrishti

# Or SQLite In-Memory for Testing
# DATABASE_URL=sqlite:///:memory:

# Optional: Path to trained ML model artifact (relative to repository root)
# MODEL_ARTIFACT_PATH=ml/models/rul_baseline.joblib
```

> [!IMPORTANT]
> `DATABASE_URL` must be explicitly configured. To prevent insecure defaults, the application does not hardcode fallback credentials; if `DATABASE_URL` is missing, an actionable `ConfigurationError` is raised.

### Table Schema (`machines`)

| Column | Type | Constraints / Description |
|---|---|---|
| `id` | Integer | Primary key, autoincrement |
| `machine_id` | String(128) | Unique, indexed, not null |
| `source_dataset` | String(64) | Dataset identifier (e.g. `NASA_CMAPSS_FD001`) |
| `engine_id` | Integer | Indexed, not null |
| `cycle` | Integer | Monitored cycle number, not null |
| `operating_settings` | JSON | Exactly `setting_1`, `setting_2`, `setting_3` |
| `sensors` | JSON | Monitored sensor channels (`sensor_01` .. `sensor_21`) |
| `predicted_rul_cycles` | Float | Nullable predicted RUL |
| `anomaly_score` | Float | Nullable anomaly score |
| `health_status` | String(32) | Asset health status enum (`unknown`, `normal`, `warning`, `critical`) |
| `replay_status` | String(32) | Replay status enum (`idle`, `running`, `paused`, `completed`) |
| `sequence_number` | Integer | Sequence index in telemetry replay |
| `replay_timestamp` | DateTime(tz) | Timestamp of replay cycle |
| `created_at` | DateTime(tz) | Server default UTC timestamp |
| `updated_at` | DateTime(tz) | Server default UTC timestamp with auto-update |

### Migrations & Table Initialization

- **Development / Testing:** Tables are initialized during FastAPI application startup inside the `lifespan` handler via `init_db()` in `backend/app/db/session.py`. Connection errors are cleanly logged as warnings without hiding code errors.
- **Production:** Production deployments should execute schema migrations via Alembic.

---

## Setup & Local Execution

### 1. Prerequisites

- Python 3.12+
- PostgreSQL 14+ (optional for local testing; SQLite in-memory is supported for tests)

### 2. Environment Setup

```bash
# From repository root:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. PostgreSQL Database Setup (Local)

If running PostgreSQL locally:

```bash
# Create database
createdb yantradrishti

# Configure environment from repository template
cp .env.example .env
# Edit DATABASE_URL in .env with your local PostgreSQL credentials:
# DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/yantradrishti
```

### 4. Running the Backend Server

```bash
# From repository root:
uvicorn app.main:app --app-dir backend --reload --port 8000

# Or from backend/ directory:
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints

### Machine Asset Management

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `POST` | `/api/v1/machines` | Register an industrial machine asset | 201, 409, 422, 503 |
| `GET` | `/api/v1/machines` | List all registered machines | 200, 503 |
| `GET` | `/api/v1/machines/{machine_id}` | Retrieve specific machine by ID | 200, 404, 503 |

### Machine Learning Inference

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `POST` | `/api/v1/predictions/rul` | Predict Remaining Useful Life (RUL) | 200, 422, 503, 500 |

### System Health

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/` | API root metadata | 200 |
| `GET` | `/api/v1/health` | Service health status | 200 |

---

## Running the Automated Test Suite

All tests run cleanly with isolated in-memory SQLite and mock/real ML models without requiring external services:

```bash
# Run complete pytest suite from repository root:
pytest -v backend

# Or run individual test suites:
python backend/test_machine_schema.py
python backend/test_prediction_schema.py
python backend/test_feature_service.py
python backend/test_inference_service.py
python backend/test_prediction_api.py
python backend/test_database.py
```
