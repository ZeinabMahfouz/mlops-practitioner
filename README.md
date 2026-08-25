# 🚀 MLOps Practitioner — Production Machine Learning Service (prodml)

This repository provides an end-to-end engineering baseline to turn a machine learning model into a Production-Ready REST API built with **FastAPI** and **Docker**. It features modular packaging (`pip` installable), automated testing with high coverage (**pytest**), ONNX model export and parity testing, code quality tooling (**Ruff**), and a structured project layout.

---

## 📋 Prerequisites

Ensure you have the following installed on your system (working inside **WSL2 - Ubuntu 22.04** is recommended for Windows users):

* **Python 3.10+** (Python 3.13 supported)
* **Docker Engine / Docker Desktop**
* **Git**

---

## ⚡ 3-Command Quickstart

Run the containerized ML service directly using the pre-built Docker image:

```bash
# 1. Pull or run the container from Docker Hub
docker run -d -p 8000:8000 --name prodml-app zeinab1987/prodml-app:v0.1.0

# 2. Check service health
curl http://localhost:8000/health

🌐 Interactive Documentation: Visit http://localhost:8000/docs in your browser to test endpoints via Swagger UI.

🛠️ Local Development Setup

If you want to modify code, run training pipelines, or execute tests locally:

1️⃣ Environment Setup

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

2️⃣ Model Training & ONNX Export

Train the Ridge regression baseline model and generate both .pkl and .onnx artifacts:
python -m prodml.train

3️⃣ Run Local API Server

uvicorn prodml.api.app:app --host 0.0.0.0 --port 8000 --reload

🧪 Testing & Code QualityMaintain code reliability, linting, and coverage standards with these commands:
Run Test Suite & Coverage Report (Target $\ge 70\%$, Achieved $90\%$):
    pytest --cov=src/prodml tests/
Code Quality & Linting:
    ruff check src/

📡 API Endpoints & Usage

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check service health and availability |
| `GET` | `/metadata` | Inspect loaded model versions and environment details |
| `POST` | `/predict` | Predict single trip duration |
| `POST` | `/predict/batch` | Predict multiple trip durations in batch |
| `GET` | `/docs` | Interactive OpenAPI / Swagger UI |

Example Request (POST /predict)

curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "PU_DO": "130_205",
  "trip_distance": 2.5
}'

📂 Project Architecture

mlops-practitioner/
├── .github/                # CI/CD Workflows
├── data/                   # Raw/processed datasets
├── models/                 # Serialized model artifacts (model.pkl, model.onnx)
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── reports/                # Module documentations and metrics reports
├── src/
│   └── prodml/             # Main installable Python package
│       ├── api/            # FastAPI initialization, routing, and Pydantic schemas
│       ├── config.py       # Configuration and paths management
│       ├── data.py         # Data loading and splitting pipelines
│       ├── features.py     # Feature engineering & transformation functions
│       ├── predict.py      # Model loading and inference engines
│       └── train.py        # Model training and ONNX export pipeline
├── tests/                  # Pytest test suite (API, pipeline, and ONNX parity)
│   ├── test_api.py
│   ├── test_onnx_parity.py
│   ├── test_pipeline.py
│   └── test_predict.py
├── Dockerfile              # Non-root container runtime configuration
├── pyproject.toml          # Build configuration and dependency management
└── README.md               # Project documentation


🏷️ Release
Tagged under v0.1.0 representing Module 1: Production Machine Learning Baseline.
