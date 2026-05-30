"""DAG de Airflow — pipeline de entrenamiento del modelo de stroke."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://api:8000")

default_args = {
    "owner": "mlops-fiuba",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def fetch_data() -> None:
    """Downloads a fresh dataset snapshot from the API."""
    import urllib.request

    data_path = os.getenv(
        "DATA_PATH", "/opt/airflow/data/healthcare-dataset-stroke-data.csv"
    )
    url = f"{DATA_SERVICE_URL}/dataset"

    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"/dataset returned HTTP {response.status}")
        content = response.read()

    Path(data_path).write_bytes(content)
    print(f"Dataset escrito en {data_path} ({len(content)} bytes)")


def validate_data() -> None:
    """Verifica que el dataset exista y tenga el esquema esperado."""
    import os

    import pandas as pd

    data_path = Path(
        os.getenv("DATA_PATH", "/opt/airflow/data/healthcare-dataset-stroke-data.csv")
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    df = pd.read_csv(data_path)

    expected_cols = {
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    if len(df) < 100:
        raise ValueError(f"Dataset demasiado pequeño: {len(df)} filas")

    print(f"Dataset validado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Stroke rate: {df['stroke'].mean():.2%}")


def train_model() -> None:
    """Entrena el Random Forest y loggea el run a MLflow."""
    from model.train import train

    train()


with DAG(
    dag_id="stroke_prediction_pipeline",
    default_args=default_args,
    description="Pipeline de entrenamiento del modelo de prediccion de stroke",
    schedule=timedelta(weeks=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["stroke", "mlops", "training"],
) as dag:
    task_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_fetch >> task_validate >> task_train
