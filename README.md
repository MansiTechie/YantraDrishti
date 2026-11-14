# YantraDrishti

### AI-Powered Predictive Digital Twin for Smart Manufacturing

YantraDrishti is an AI-powered predictive maintenance platform designed to simulate, monitor, and analyze industrial machine health using historical sensor data, machine learning, and digital twin visualization.

The goal is to move beyond traditional reactive maintenance by estimating Remaining Useful Life (RUL), identifying abnormal machine behavior, and presenting machine health insights through an interactive digital environment.

> **Project Status:** Active Development 🚧

---

## 🎯 Project Objectives

- Build a predictive maintenance system using industrial machine sensor data.
- Estimate Remaining Useful Life (RUL) in operating cycles.
- Detect abnormal machine behavior using anomaly detection.
- Provide interpretable predictions using explainable AI.
- Simulate machine operating conditions through historical data replay.
- Visualize machine health and sensor behavior through a digital twin interface.

## ✨ Key Features

| Feature | Description | Status |
|---|---|---|
| Machine Management API | Register and retrieve machine information | Implemented |
| Data Validation | Validate machine telemetry and prediction schemas | Implemented |
| Dataset Inspection | Initial NASA C-MAPSS FD001 analysis | Completed |
| Data Preprocessing | Engine-wise data preparation and feature engineering | In Progress |
| RUL Prediction | Estimate remaining useful life | Planned |
| Anomaly Detection | Identify unusual machine behavior | Planned |
| Explainable AI | Understand important prediction features using SHAP | Planned |
| Historical Replay | Replay machine operating cycles | Planned |
| Digital Twin Dashboard | Interactive machine visualization | Planned |

## 🏗️ System Architecture

The intended system architecture consists of the following layers:

1. **Data Layer:** Historical industrial sensor data and preprocessing pipelines.
2. **Machine Learning Layer:** RUL prediction, anomaly detection, and explainability.
3. **Backend Layer:** FastAPI services for machine management, predictions, and replay.
4. **Database Layer:** Persistent storage for machine information, telemetry, and prediction history.
5. **Visualization Layer:** Interactive digital twin dashboard for machine monitoring.

### Technology Stack

| Layer | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost / LightGBM |
| Explainable AI | SHAP |
| Backend | FastAPI, Pydantic |
| Frontend | React, TypeScript |
| 3D Visualization | Three.js, React Three Fiber |
| Database | PostgreSQL |
| Communication | REST APIs, WebSockets |
| Development & Deployment | Git, GitHub, Docker, GitHub Actions |

*The technology stack represents the planned architecture; not every component has been integrated yet.*

## 📊 Dataset

The primary dataset used in this project is **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**.

Initial development focuses on the FD001 subset.

Dataset characteristics:

- 100 training engine trajectories.
- 100 test engine trajectories.
- 21 anonymous sensor channels.
- 3 operating settings.
- Historical run-to-failure training trajectories.
- Remaining Useful Life (RUL) measured in operating cycles.

**Important data considerations:**

- Sensor channels retain their original anonymous identifiers.
- RUL is represented in cycles, not hours or days.
- Data splitting is performed at the engine level to avoid data leakage.
- Preprocessing parameters are fitted using training data only.
- Synthetic replay data is not used as a substitute for real training data.

Dataset files are maintained under `datasets/raw/`.

## 🔧 Backend API

The backend is built using FastAPI and Pydantic.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API root |
| GET | `/api/v1/health` | Backend health check |
| POST | `/api/v1/machines` | Register a machine |
| GET | `/api/v1/machines` | Retrieve all registered machines |
| GET | `/api/v1/machines/{machine_id}` | Retrieve a machine by ID |

### API Documentation

Once the backend is running, interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### Current Backend Capabilities

- Machine registration with structured Pydantic validation.
- Machine identity and telemetry schemas.
- Operating settings and sensor validation.
- Prediction and replay metadata schemas.
- Duplicate machine detection (`409 Conflict`).
- Missing machine handling (`404 Not Found`).
- Schema validation test suite.

**Current limitation:** Machine records are stored in temporary in-memory storage. Database persistence and ML inference integration are future development tasks.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or compatible version.
- Git.
- pip.

### 1. Clone the repository

```bash
git clone https://github.com/MansiTechie/YantraDrishti.git
cd YantraDrishti
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it using Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the backend server

```bash
cd backend
python -m uvicorn app.main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Swagger documentation:

`http://127.0.0.1:8000/docs`

### 5. Run schema validation tests

From the `backend/` directory:

```bash
python test_machine_schema.py
```

## 📁 Repository Structure

```text
YantraDrishti/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── machines.py
│   │   ├── schemas/
│   │   │   └── machine.py
│   │   ├── services/
│   │   └── main.py
│   │
│   ├── requirements.txt
│   ├── test_machine_schema.py
│   └── README.md
│
├── datasets/
│   └── raw/
│       └── CMAPSSData.zip
│
├── ml/
│   └── notebooks/
│       └── YantraDrishti_Day1_Dataset_Inspection.ipynb
│
├── .gitignore
└── README.md
```

*The repository structure will evolve as the ML pipeline, database, replay engine, and frontend are developed.*

## 🧠 Machine Learning Roadmap

The planned machine learning workflow includes:

1. Dataset inspection and validation.
2. Engine-wise training and validation split.
3. RUL target preparation and feature engineering.
4. Baseline model development.
5. Anomaly detection using statistical baselines and Isolation Forest.
6. RUL regression using appropriate machine learning models.
7. Evaluation using MAE, RMSE, and suitable NASA scoring methodology.
8. Model explainability using SHAP.
9. Model integration with the FastAPI backend.

Model performance will be reported only after proper training and evaluation.

## 🗺️ Development Roadmap

- [x] Project repository and initial structure.
- [x] NASA C-MAPSS FD001 dataset inspection.
- [x] FastAPI backend initialization.
- [x] Machine state schemas and validation.
- [x] Initial machine management endpoints.
- [ ] Reproducible ML preprocessing pipeline.
- [ ] RUL prediction model.
- [ ] Anomaly detection model.
- [ ] Explainability integration.
- [ ] Database integration.
- [ ] Historical machine replay engine.
- [ ] Interactive digital twin dashboard.
- [ ] End-to-end integration and deployment.

## 👥 Contributors

- **Vivek Nigam** — Backend development, platform architecture, and integration.
- **Mansi** — Dataset analysis, machine learning pipeline, and model development.

## 🤝 Collaboration

This project is developed collaboratively using Git and GitHub.

Development follows a feature-branch workflow:

- `main` — Stable project milestones.
- `develop` — Shared integration branch.
- `feature/*` — Individual development branches.

Changes are reviewed through Pull Requests before integration.

---

**YantraDrishti — Bringing intelligence to industrial machine health.**