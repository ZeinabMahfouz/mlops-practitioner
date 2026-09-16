"""Task logic for the Module 3 Step 5 batch scoring DAG
(``batch_scoring_pipeline.py``).

Same split as the training DAG's tasks.py: the DAG file stays a readable
graph, business logic lives here and is testable without Airflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prodml.batch import run_batch_scoring

logger = logging.getLogger(__name__)

BATCH_INPUT_DIR = Path("/opt/airflow/data/raw/batch_input")
PREDICTIONS_DIR = Path("/opt/airflow/data/predictions")
MODEL_PATH = Path("/opt/airflow/models/model.pkl")


def score_task(**context) -> dict:
    summary = run_batch_scoring(BATCH_INPUT_DIR, PREDICTIONS_DIR, model_path=MODEL_PATH)
    logger.info("Nightly batch scoring finished: %s", summary)
    return summary
