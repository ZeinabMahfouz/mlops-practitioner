🚀 MLOps Practitioner — Production Machine Learning Service (prodml)


This repository provides an end-to-end engineering baseline to turn a machine learning model into a Production-Ready REST API built with FastAPI and Docker. It features modular packaging (pip installable), automated tests (pytest), and structured code layout.

📋 Prerequisites
Before getting started, ensure you have the following installed on your machine (working inside WSL2 - Ubuntu 22.04 is recommended for Windows users):

  * Python 3.10 or higher

  *   Docker Engine / Docker Desktop

  *  Git
  ⚡ 3-Command Quickstart

Run the containerized ML service without any local environment configuration:

# 1. Clone the repository
git clone https://github.com/ZeinabMahfouz/mlops-practitioner.git && cd mlops-practitioner

# 2. Build the Docker Image
docker build -t mlops-prodml:v0.1.0 .

# 3. Run the Container
docker run -d -p 8000:8000 --name prodml-app mlops-prodml:v0.1.0

🌐 Interactive Documentation: Visit http://localhost:8000/docs in your browser to access the Swagger UI and test endpoints directly.

🛠️ Local Development Setup
If you wish to modify the codebase, execute tests, or develop locally:

1️⃣ Create and Activate Virtual Environment

python3 -m venv .venv
source .venv/bin/activate

2️⃣ Install Package & Development Dependencies

pip install --upgrade pip
pip install -e ".[dev]"

3️⃣ Run Local API Server

uvicorn prodml.api.app:app --host 0.0.0.0 --port 8000 --reload


🧪 Testing & Code Quality
Maintain code reliability and style guidelines using the following commands:

* Run Tests with Coverage Report:
pytest --cov=src/prodml tests/

* Code Linting & Formatting:
ruff check .
black --check .

📡 API Endpoints & Usage

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check service health status |
| `POST` | `/predict` | Send input features to receive model predictions |
| `GET` | `/docs` | Interactive Swagger UI documentation |

Example Request (curl)

curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [1.0, 2.0, 3.0]
}'
📂 Project Architecture

mlops-practitioner/
├── .github/              # CI/CD Workflows
├── data/                 # Raw/processed datasets (e.g., parquet files)
├── models/               # Serialized model artifacts (.pkl)
├── notebooks/            # Jupyter notebooks for EDA and baseline modeling
├── reports/              # Module documentations and reports
├── src/
│   └── prodml/           # Main installable package
│       ├── api/          # FastAPI app, endpoints, and schemas
│       ├── config.py     # Configuration management
│       ├── data.py       # Data processing logic
│       ├── features.py   # Feature extraction & pipelines
│       ├── predict.py    # Model inference logic
│       └── train.py      # Model training script
├── tests/                # Automated pytest tests
├── Dockerfile            # Container definition
├── pyproject.toml        # Package dependencies & configuration
└── README.md             # Project documentation

🏷️ Release

This project is tagged under version v0.1.0 representing Module 1: From Notebook to Production Service.
