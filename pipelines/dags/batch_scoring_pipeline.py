from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
import batch_tasks  # noqa: E402
from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

default_args = {
    "owner": "zeinab",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ride_duration_batch_scoring_pipeline",
    description="Nightly batch scoring against the current model artifact",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["module-3", "batch-scoring"],
) as dag:
    score_batch = PythonOperator(
        task_id="score_batch",
        python_callable=batch_tasks.score_task,
        execution_timeout=timedelta(minutes=30),
    )
