from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, os.path.dirname(__file__))
import tasks  # noqa: E402

default_args = {
    "owner": "zeinab",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ride_duration_training_pipeline",
    description=(
        "extract > validate > train > evaluate > [branch] > register/skip > notify"
    ),
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["module-3", "training"],
) as dag:
    wait_for_raw_data = FileSensor(
        task_id="wait_for_raw_data",
        fs_conn_id="fs_default",
        filepath=str(tasks.RAW_DATA_PATH),
        poke_interval=30,
        timeout=60 * 10,
        mode="reschedule",
    )

    extract = PythonOperator(
        task_id="extract",
        python_callable=tasks.extract_task,
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=tasks.validate_task,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=tasks.train_task,
        execution_timeout=timedelta(minutes=30),
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=tasks.evaluate_task,
    )

    decide_promotion = BranchPythonOperator(
        task_id="decide_promotion",
        python_callable=tasks.decide_promotion_task,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=tasks.register_task,
    )

    skip_registration = PythonOperator(
        task_id="skip_registration",
        python_callable=tasks.skip_task,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=tasks.notify_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    wait_for_raw_data >> extract >> validate >> train >> evaluate >> decide_promotion
    decide_promotion >> register_model >> notify
    decide_promotion >> skip_registration >> notify
