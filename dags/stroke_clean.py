"""
DAG: stroke_clean
transformaciones sobre healthcare-dataset-stroke-data.csv
y sube los datasets resultantes (X_train, X_val, X_test, y_train, y_val, y_test)
a MinIO (S3) como archivos CSV.

Tareas
------
1.validate_source          — verifica que el CSV de entrada existe y es legible
2.load_and_split            — carga el CSV, drop de 'id', split estratificado 60/20/20
3.impute_bmi               — imputacion de BMI por mediana de grupo etario (solo en train)
4.encode_features          — encoding binario + OHE (solo en train)
5.scale_features           — StandardScaler sobre columnas numéricas (solo en train)
6.upload_to_minio          — sube los 6 CSVs a MinIO en s3
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import os

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator


DATA_PATH =  Path(os.getenv("DATA_PATH", "/opt/airflow/data/healthcare-dataset-stroke-data.csv"))
BUCKET     = "mlflow-artifacts"
S3_PREFIX  = "processed"
NUM_COLS   = ["age", "avg_glucose_level", "bmi"]
BIN_COLS   = ["hypertension", "heart_disease", "gender", "ever_married", "Residence_type"]
OHE_COLS   = ["work_type", "smoking_status"]
AGE_BINS   = [0, 10, 20, 30, 70, np.inf]
AGE_LABELS = ["0-10", "11-20", "21-30", "31-70", "71+"]

MINIO_URL  = os.environ["MLFLOW_S3_ENDPOINT_URL"]
#MINIO_ROOT_USER = Variable.get("MINIO_ROOT_USER",    default_var="minioadmin")
#MINIO_ROOT_PASSWORD = Variable.get("MINIO_ROOT_PASSWORD", default_var="minioadmin_secret")
MINIO_ROOT_USER = os.environ["AWS_ACCESS_KEY_ID"]
MINIO_ROOT_PASSWORD = os.environ["AWS_SECRET_ACCESS_KEY"]


from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log
logger.info("--/-/-/-> ----/---/")



default_args = {
    "owner": "mlops-fiuba",
    "retries": 0,
    "retry_delay": timedelta(minutes=0.5),
}



# helpers
def _get_s3_client():
    # conexion a s3
    return boto3.client(
        "s3",
        endpoint_url         = MINIO_URL,
        aws_access_key_id    = MINIO_ROOT_USER,
        aws_secret_access_key= MINIO_ROOT_PASSWORD,
        config               = Config(signature_version="s3v4"),
        region_name          = "us-east-1",
    )



def _df_to_s3(df: pd.DataFrame, s3_client, key: str) -> None:
    """Sube un DataFrame como CSV a S3/MinIO."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    logger.info("--/-/-/->Uploaded %s (%d rows, %d cols)", key, len(df), df.shape[1])


def _s3_to_df(s3_client, key: str) -> pd.DataFrame:
    """Descarga un CSV de S3/MinIO y lo devuelve como DataFrame."""
    obj = s3_client.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# Tarea 1: validate_source

def validate_source(**context) -> None:
    """
    Verifica que el CSV de entrada exista y tenga las columnas esperadas.
    Falla explícitamente antes de que alguna tarea posterior consuma tiempo.
    """
    import os

    data_path = context["params"].get(
        "data_path",
        "/opt/airflow/data/healthcare-dataset-stroke-data.csv",
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    df = pd.read_csv(data_path, nrows=5)

    required = {
        "id", "gender", "age", "hypertension", "heart_disease",
        "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status", "stroke",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")

    logger.info("--/-/-/->Validacion OK — path=%s", data_path)
    #context["ti"].xcom_push(key="data_path", value=data_path)
    context["ti"].xcom_push(key="data_path",value=str(data_path))

# Tarea 2  load_and_split

def load_and_split(**context) -> None:
    logger.info("--/-/-/->$$$$$$$$   Ejecutando load_and_split...")
    """
    - Carga el CSV completo.
    - Elimina la columna 'id' (identificador no informativo).
    - Split : 60% train, 20% val, 20% test.
    - Sube los 6 splits crudos a MinIO como CSV intermedios.
    - Pushea metadatos via XCom.
    """
    ti        = context["ti"]
    data_path = ti.xcom_pull(task_ids="validate_source", key="data_path")
    s3        = _get_s3_client()

    logger.info("--/-/-/->$$$$$$$$  Dataset cargado desde validate_source:", data_path)
    # Carga
    df = pd.read_csv(data_path)
    logger.info("--/-/-/->Dataset cargado: %s", df.shape)

    # Eliminar identificador
    df = df.drop(columns=["id"])

    # Separar features / target
    X = df.drop(columns=["stroke"])
    y = df["stroke"].astype(int)


    # Split 60/20/20 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    logger.info(
        "Split — train=%d  val=%d  test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    logger.info("--/-/-/->$$$$$$$$  Split realizado. Subiendo a MinIO...")
    # Subir a MinIO
    for name, frame in [
        ("X_train_raw", X_train), ("X_val_raw", X_val), ("X_test_raw", X_test),
        ("y_train",     y_train.to_frame()), ("y_val", y_val.to_frame()), ("y_test", y_test.to_frame()),
    ]:
        _df_to_s3(frame, s3, f"{S3_PREFIX}/splits/{name}.csv")

    logger.info("--/-/-/->$$$$$$$$  Splits subidos a MinIO.")

    ti.xcom_push(key="split_shapes", value={
        "train": len(X_train), "val": len(X_val), "test": len(X_test),
    })


# Tarea 3 — impute_bmi
def impute_bmi(**context) -> None:
    """
    Imputa BMI usando medianas por grupo etario calculadas SOLO en train
    (evita data leakage).

    Grupos: [0-10, 11-20, 21-30, 31-70, 71+]

    La mediana global actúa como fallback si algún grupo no tiene datos.
    """
    s3 = _get_s3_client()

    # Cargar splits
    X_train = _s3_to_df(s3, f"{S3_PREFIX}/splits/X_train_raw.csv")
    X_val   = _s3_to_df(s3, f"{S3_PREFIX}/splits/X_val_raw.csv")
    X_test  = _s3_to_df(s3, f"{S3_PREFIX}/splits/X_test_raw.csv")

    #  Fit: medianas por grupo etario (solo train) 
    X_train["_age_group"] = pd.cut(
        X_train["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    medians_by_group: dict = (
        X_train.groupby("_age_group", observed=False)["bmi"]
        .median()
        .dropna()
        .to_dict()
    )
    medians_by_group = {str(k): float(v) for k, v in medians_by_group.items()}
    global_median    = float(X_train["bmi"].median())

    logger.info("--/-/-/->BMI medians by age group: %s  |  global: %.2f", medians_by_group, global_median)

    #  Transform: aplica sobre train, val y test 
    def _apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
        )
        fill = df["_age_group"].astype(str).map(medians_by_group).fillna(global_median)
        df["bmi"] = df["bmi"].fillna(fill)
        df = df.drop(columns=["_age_group"])
        return df

    X_train = _apply_imputation(X_train)
    X_val   = _apply_imputation(X_val)
    X_test  = _apply_imputation(X_test)

    # Verificar que no quedan nulos en bmi
    for name, df in [("train", X_train), ("val", X_val), ("test", X_test)]:
        remaining = df["bmi"].isnull().sum()
        if remaining > 0:
            raise ValueError(f"Quedan {remaining} nulos en bmi ({name}) tras la imputacion")

    # Subir a MinIO
    logger.info("--/-/-/->$$$$$$$$  Subiendo datos imputados a MinIO...")
    for name, frame in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        _df_to_s3(frame, s3, f"{S3_PREFIX}/imputed/{name}.csv")

    # Guardar parámetros de imputación para reproducibilidad
    imputation_params = {"medians_by_group": medians_by_group, "global_median": global_median}
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{S3_PREFIX}/params/bmi_imputation.json",
        Body=json.dumps(imputation_params),
    )
    logger.info("--/-/-/->BMI imputado correctamente en los 3 splits")



# Tarea  4: encode_features

def encode_features(**context) -> None:
    """
    1. Elimina filas con gender='Other' 
    2. Encoding binario: gender, ever_married, Residence_type.
    3. Reemplaza smoking_status='Unknown' por 'never smoked'.
    4. OneHotEncoding (drop='first') sobre work_type y smoking_status (en train).
    5. Sube los DataFrames con features finales (sin escalar) a MinIO.
    """
    s3 = _get_s3_client()

    X_train = _s3_to_df(s3, f"{S3_PREFIX}/imputed/X_train.csv")
    X_val   = _s3_to_df(s3, f"{S3_PREFIX}/imputed/X_val.csv")
    X_test  = _s3_to_df(s3, f"{S3_PREFIX}/imputed/X_test.csv")
    y_train = _s3_to_df(s3, f"{S3_PREFIX}/splits/y_train.csv").squeeze()
    y_val   = _s3_to_df(s3, f"{S3_PREFIX}/splits/y_val.csv").squeeze()
    y_test  = _s3_to_df(s3, f"{S3_PREFIX}/splits/y_test.csv").squeeze()

    #  1. Eliminar gender='Other' ─
    def _drop_other_gender(X, y):
        mask = X["gender"] != "Other"
        return X[mask].copy(), y[mask]

    X_train, y_train = _drop_other_gender(X_train, y_train)
    X_val,   y_val   = _drop_other_gender(X_val,   y_val)
    X_test,  y_test  = _drop_other_gender(X_test,  y_test)

    logger.info("--/-/-/->Tras drop 'Other': train=%d  val=%d  test=%d",
             len(X_train), len(X_val), len(X_test))

    #  2. Smoking_status: reemplazar Unknown 
    for df in [X_train, X_val, X_test]:
        df["smoking_status"] = df["smoking_status"].replace("Unknown", "never smoked")

    #  3. Encoding binario manual ─
    gender_map       = {"Male": 0, "Female": 1}
    ever_married_map = {"Yes": 1, "No": 0}
    residence_map    = {"Urban": 1, "Rural": 0}

    for df in [X_train, X_val, X_test]:
        df["gender"]         = df["gender"].map(gender_map)
        df["ever_married"]   = df["ever_married"].map(ever_married_map)
        df["Residence_type"] = df["Residence_type"].map(residence_map)

    #  4. OneHotEncoding (fit solo en train) 
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=int, drop="first"
    )
    encoder.fit(X_train[OHE_COLS])

    def _apply_ohe(X):
        ohe_array = encoder.transform(X[OHE_COLS])
        ohe_cols  = encoder.get_feature_names_out(OHE_COLS)
        ohe_df    = pd.DataFrame(ohe_array, columns=ohe_cols, index=X.index)
        return pd.concat([X[NUM_COLS + BIN_COLS], ohe_df], axis=1)

    X_train_final = _apply_ohe(X_train)
    X_val_final   = _apply_ohe(X_val)
    X_test_final  = _apply_ohe(X_test)

    # Verificar no hay nulos
    for name, df in [("train", X_train_final), ("val", X_val_final), ("test", X_test_final)]:
        nulls = df.isnull().sum().sum()
        if nulls:
            raise ValueError(f"Nulos inesperados tras encoding ({name}): {nulls}")

    # Subir a MinIO
    for name, X, y in [
        ("train", X_train_final, y_train),
        ("val",   X_val_final,   y_val),
        ("test",  X_test_final,  y_test),
    ]:
        _df_to_s3(X, s3, f"{S3_PREFIX}/encoded/X_{name}.csv")
        _df_to_s3(y.to_frame(), s3, f"{S3_PREFIX}/encoded/y_{name}.csv")

    # Guardar feature names
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{S3_PREFIX}/params/feature_names.json",
        Body=json.dumps(list(X_train_final.columns)),
    )
    logger.info("--/-/-/->Encoding completado. Features: %s", list(X_train_final.columns))


# Tarea  5 escalado. se podrìa hacer antes, pero se separa para hacer una iteraccion intermedia con S3
# como ejemplo de como cada tarea puede ser independiente y reutilizable
def scale_features(**context) -> None:
    """
    Aplica StandardScaler sobre age, avg_glucose_level y bmi.
    El scaler se fita SOLO en train y se aplica a val y test.
    Las columnas binarias y OHE se pasan sin transformar (remainder='passthrough').
    """
    s3 = _get_s3_client()

    X_train = _s3_to_df(s3, f"{S3_PREFIX}/encoded/X_train.csv")
    X_val   = _s3_to_df(s3, f"{S3_PREFIX}/encoded/X_val.csv")
    X_test  = _s3_to_df(s3, f"{S3_PREFIX}/encoded/X_test.csv")

    scaler = StandardScaler()
    scaler.fit(X_train[NUM_COLS])

    def _scale(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[NUM_COLS] = scaler.transform(df[NUM_COLS])
        return df

    X_train_scaled = _scale(X_train)
    X_val_scaled   = _scale(X_val)
    X_test_scaled  = _scale(X_test)

    # Subir datasets finales
    for name, df in [("train", X_train_scaled), ("val", X_val_scaled), ("test", X_test_scaled)]:
        _df_to_s3(df, s3, f"{S3_PREFIX}/final/X_{name}.csv")

    # Guardar parámetros del scaler
    scaler_params = {
        "mean_" : scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "cols"  : NUM_COLS,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{S3_PREFIX}/params/scaler.json",
        Body=json.dumps(scaler_params),
    )
    logger.info(
        "Scaling completado — train=%s  val=%s  test=%s",
        X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape,
    )


# Tarea  6 upload_to_minio  (datasets finales + targets)
def upload_to_minio(**context) -> None:
    """
    Consolida y verifica los datasets finales en MinIO.
    Sube los targets (y_train, y_val, y_test) al prefijo final/.
    Genera un manifiesto JSON con shapes y checksums de cada archivo.
    """
    import hashlib

    s3 = _get_s3_client()

    manifest = {}

    for split in ["train", "val", "test"]:
        # X ya subido en scale_features, copiar y al prefijo final/
        y_key_src = f"{S3_PREFIX}/encoded/y_{split}.csv"
        y_key_dst = f"{S3_PREFIX}/final/y_{split}.csv"

        obj  = s3.get_object(Bucket=BUCKET, Key=y_key_src)
        body = obj["Body"].read()
        s3.put_object(Bucket=BUCKET, Key=y_key_dst, Body=body)

        # Verificar X
        x_key = f"{S3_PREFIX}/final/X_{split}.csv"
        x_obj = s3.get_object(Bucket=BUCKET, Key=x_key)
        x_body = x_obj["Body"].read()
        x_df = pd.read_csv(io.BytesIO(x_body))
        y_df = pd.read_csv(io.BytesIO(body))

        manifest[f"X_{split}"] = {
            "key"     : x_key,
            "rows"    : len(x_df),
            "cols"    : x_df.shape[1],
            "md5"     : hashlib.md5(x_body).hexdigest(),
        }
        manifest[f"y_{split}"] = {
            "key"     : y_key_dst,
            "rows"    : len(y_df),
            "stroke_rate": round(float(y_df.iloc[:, 0].mean()), 4),
            "md5"     : hashlib.md5(body).hexdigest(),
        }
        logger.info(
            "%s — X: %s  |  stroke_rate: %.4f",
            split, x_df.shape, y_df.iloc[:, 0].mean(),
        )

    # Subir manifiesto
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{S3_PREFIX}/final/manifest.json",
        Body=json.dumps(manifest, indent=2),
    )
    logger.info("--/-/-/->Manifiesto subido. Datasets disponibles en s3://%s/%s/final/", BUCKET, S3_PREFIX)



# DAG 
with DAG(
    dag_id            = "stroke_data_cleaning",
    description       = "Preprocesamiento del dataset de stroke y subida a MinIO",
    schedule_interval = None,           # disparo manual o desde otro DAG
    start_date        = datetime(2024, 1, 1),
    catchup           = False,
    default_args      = default_args,
    tags              = ["mlops", "preprocessing", "stroke"],
    params            = {"data_path":DATA_PATH},
    doc_md            = __doc__,
) as dag:

    t1_validate = PythonOperator(
        task_id         = "validate_source",
        python_callable = validate_source,
    )

    t2_split = PythonOperator(
        task_id         = "load_and_split",
        python_callable = load_and_split,
    )

    t3_impute = PythonOperator(
        task_id         = "impute_bmi",
        python_callable = impute_bmi,
    )

    t4_encode = PythonOperator(
        task_id         = "encode_features",
        python_callable = encode_features,
    )

    t5_scale = PythonOperator(
        task_id         = "scale_features",
        python_callable = scale_features,
    )

    t6_upload = PythonOperator(
        task_id         = "upload_to_minio",
        python_callable = upload_to_minio,
    )

    # Pipeline lineal — cada tarea depende de la anterior
    t1_validate >> t2_split >> t3_impute >> t4_encode >> t5_scale >> t6_upload
