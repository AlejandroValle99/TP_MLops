# TP MLOps — Predicción de Stroke

Integrantes del grupo:

Gabriela Sol Salazar
Christian Aballay
Leonardo Villalva
Alejandro A. Valle
Mariel Gaitán

Pipeline de MLOps para predecir riesgo de ACV (stroke) sobre el [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Todo el sistema corre en contenedores Docker con un único comando.

---

## Arquitectura

```
                        Docker Compose
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│   │   Airflow   │   │   MLflow    │   │      MinIO      │    │
│   │  :8080      │   │  :5001      │   │  :9000 / :9001  │    │
│   └──────┬──────┘   └──────┬──────┘   └────────┬────────┘    │
│          │                 │                    │             │
│          └────────┬────────┘                    │             │
│                   ▼                             │             │
│          ┌────────────────┐                     │             │
│          │   PostgreSQL   │◀────────────────────┘             │
│          │    :5432       │                                   │
│          └────────────────┘                                   │
│                                                                │
│   ┌─────────────┐                                             │
│   │   FastAPI   │  ◀── carga modelo champion desde MLflow    │
│   │   :8000     │  ◀── lee dataset desde MinIO               │
│   └─────────────┘                                             │
└────────────────────────────────────────────────────────────────┘
```

### Servicios

| Servicio              | Puerto      | Rol                                                                    |
| --------------------- | ----------- | ---------------------------------------------------------------------- |
| **FastAPI**           | 8000        | Sirve predicciones; carga el modelo `champion` desde MLflow al iniciar |
| **Airflow webserver** | 8080        | UI de orquestación; ejecuta los DAGs de entrenamiento                  |
| **MLflow**            | 5001        | Tracking de experimentos, métricas, parámetros y model registry        |
| **MinIO**             | 9000 / 9001 | Artifact store S3-compatible; guarda modelos entrenados y el dataset   |
| **PostgreSQL**        | 5432        | Backend de metadatos de Airflow y MLflow                               |

### Secuencia de arranque

```
postgres ──▶ minio ──▶ minio-init ──▶ mlflow ──▶ model-init ──▶ api
                                             └──▶ airflow-init ──▶ airflow-webserver
                                                               └──▶ airflow-scheduler
```

`minio-init` crea los buckets `mlflow-artifacts` y `datasets`, y sube el CSV original.
`model-init` entrena el primer modelo y lo registra como `champion` en MLflow para que la API tenga un modelo disponible desde el arranque.

---

## Estructura del proyecto

```
TP_MLops/
├── api/
│   ├── main.py          # FastAPI: /health, /predict, /model-info, /dataset
│   ├── schemas.py       # Esquemas Pydantic de entrada/salida
│   └── data.py          # Lectura del dataset desde MinIO y aplicación de mutaciones
├── dags/
│   ├── stroke_pipeline.py                  # DAG de producción (semanal)
│   ├── stroke_clean.py                     # DAG de limpieza/ETL (manual)
│   ├── etl_train_models_process_taskflow.py # DAG de comparación de modelos (manual)
│   └── utils/
│       ├── model_utils.py  # AgeBaselineClassifier, cv_f2, métricas
│       ├── plots.py        # Confusion matrix, ROC, Precision-Recall
│       └── s3_utils.py     # Helpers para leer/escribir en MinIO
├── model/
│   ├── preprocess.py    # Pipeline sklearn: limpieza, imputación BMI, encoding, scaling
│   ├── train.py         # Entrenamiento del Random Forest y registro en MLflow
│   └── s3_utils.py      # Lectura del dataset desde S3/local
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── docker/
│   ├── airflow/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── mlflow/
│   │   └── Dockerfile
│   └── postgres/
│       └── init.sql         # Crea la base mlflow_db al iniciar
├── docker-compose.yml
├── Dockerfile               # Imagen compartida por api y model-init
└── .env.example
```

---

## Cómo levantar el sistema

### Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo

### 1. Configurar variables de entorno

```bash
cp .env.example .env
```

El `.env` viene con valores funcionales por defecto. Solo modificarlo para cambiar credenciales.

### 2. Levantar todos los servicios

```bash
docker compose up -d
```

La primera vez tarda varios minutos porque construye las imágenes de Airflow y MLflow. Una vez que todos los servicios están `healthy`:

| Interfaz      | URL                        | Usuario    | Contraseña        |
| ------------- | -------------------------- | ---------- | ----------------- |
| API docs      | http://localhost:8000/docs | —          | —                 |
| Airflow       | http://localhost:8080      | admin      | admin             |
| MLflow        | http://localhost:5001      | —          | —                 |
| MinIO Console | http://localhost:9001      | minioadmin | minioadmin_secret |

### 3. Verificar el estado

```bash
docker compose ps
```

Los servicios `minio-init`, `model-init` y `airflow-init` finalizan solos (no quedan corriendo). El resto debe aparecer como `healthy`.

---

## API

La API carga el modelo con alias `champion` desde MLflow al iniciar. Si no hay modelo disponible, `/health` devuelve 503.

### Endpoints

| Método | Ruta          | Descripción                                                      |
| ------ | ------------- | ---------------------------------------------------------------- |
| GET    | `/health`     | Estado del servicio y si el modelo está cargado                  |
| GET    | `/model-info` | Versión activa, run_id y métricas del modelo champion            |
| POST   | `/predict`    | Recibe datos de un paciente y devuelve predicción y probabilidad |
| GET    | `/dataset`    | Descarga el dataset desde MinIO con mutaciones aplicadas         |

### Campos de entrada para `/predict`

| Campo               | Tipo          | Valores válidos                                                              |
| ------------------- | ------------- | ---------------------------------------------------------------------------- |
| `gender`            | string        | `"Male"`, `"Female"`                                                         |
| `age`               | float         | > 0                                                                          |
| `hypertension`      | int           | `0`, `1`                                                                     |
| `heart_disease`     | int           | `0`, `1`                                                                     |
| `ever_married`      | string        | `"Yes"`, `"No"`                                                              |
| `work_type`         | string        | `"Private"`, `"Self-employed"`, `"Govt_job"`, `"children"`, `"Never_worked"` |
| `Residence_type`    | string        | `"Urban"`, `"Rural"`                                                         |
| `avg_glucose_level` | float         | > 0                                                                          |
| `bmi`               | float \| null | > 0, acepta null                                                             |
| `smoking_status`    | string        | `"never smoked"`, `"formerly smoked"`, `"smokes"`, `"Unknown"`               |

### Ejemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

```json
{
  "stroke_prediction": 1,
  "stroke_probability": 0.626
}
```

---

## DAGs de Airflow

El sistema tiene un DAG orquestador (`0_stroke_full_pipeline`) que encadena las dos etapas de selección de modelo, más un DAG de producción independiente.

### DAG orquestador: `0_stroke_full_pipeline`

Encadena **limpieza → comparación** con `TriggerDagRunOperator` + `ExternalTaskSensor`:

```
stroke_data_cleaning ──▶ etl_train_models_process_taskflow
   (split de datos)          (compara 5 modelos y promueve el champion)
```

La comparación deja directamente el modelo `champion` que sirve la API, así que no hay una etapa de "producción" separada dentro del orquestador.

### DAG 1: `stroke_prediction_pipeline` — producción

**Schedule**: semanal automático.

```
fetch_data ──▶ validate_data ──▶ train_model
```

| Tarea           | Qué hace                                                                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `fetch_data`    | Llama al endpoint `/dataset` de la API, que devuelve el CSV desde MinIO con mutaciones aplicadas, y lo escribe en disco |
| `validate_data` | Verifica columnas correctas y mínimo 100 filas                                                                          |
| `train_model`   | Entrena el Random Forest, loggea métricas en MLflow y promueve el modelo como `champion`                                |

Este DAG es el pipeline "vivo". Para triggerearlo manualmente: **Airflow UI → stroke_prediction_pipeline → ▶ Trigger DAG**.

### DAG 2: `stroke_data_cleaning` — partición de datos

**Schedule**: ninguno (disparo manual o vía orquestador).

```
validate_source ──▶ split_and_upload
```

Carga el CSV crudo con `model.preprocess.load_data` (la misma carga que usa el entrenamiento: drop de `id` y de filas `gender='Other'`), hace un split estratificado 60/20/20 y sube los 6 CSV **crudos** (X/y train/val/test) a `s3://mlflow-artifacts/processed/final/`.

Tras unificar el preprocessing, este DAG ya **no** imputa/encoda/escala: esa lógica vive en un único lugar (`model/preprocess.py`) y se aplica dentro del Pipeline de cada modelo.

### DAG 3: `etl_train_models_process_taskflow` — comparación y promoción

**Schedule**: ninguno (disparo manual o vía orquestador). **Requiere que el DAG 2 haya corrido antes.**

```
check_data_to_process ──▶ create_base_model  ──┐
                      ──▶ create_knn_model   ──┤
                      ──▶ create_decision_tree ─┤──▶ set_champion_model
                      ──▶ create_xgboost_model ─┤   (promueve el ganador por F2)
                      ──▶ create_random_forest ─┘
```

Lee los splits **crudos** de `processed/final/` y entrena cinco modelos en paralelo, cada uno como un `Pipeline([build_feature_pipeline(), estimador])`, con búsqueda de hiperparámetros vía **Optuna** (50 trials cada uno). Loggea métricas, parámetros y gráficos en MLflow. Está implementado con la **TaskFlow API**.

Al final, `set_champion_model` compara los cinco por **F2**, registra al ganador como nueva versión de **`stroke-model`** y le asigna el alias **`champion`** — el mismo modelo que sirve la API. Como cada modelo es un Pipeline autónomo (lleva el preprocessing adentro), la API lo carga sin cambios aunque gane un algoritmo distinto.

Los modelos que entrena son: baseline (solo edad), KNN, Decision Tree, XGBoost y Random Forest.

### Flujo

```
[Orquestador] stroke_data_cleaning  →  etl_train_models_process_taskflow
                  (split de datos)         (compara 5 modelos y promueve champion)

[DAG 1] stroke_pipeline                 →  bootstrap/retrain del champion (RF fijo)
         (manual)                             usado también por model-init al arranque
```

El orquestador es la herramienta para **seleccionar y promover** el mejor modelo. El DAG 1 (`stroke_pipeline`) queda como camino rápido de reentrenamiento de un único Random Forest, y es lo que usa `model-init` para tener un `champion` disponible desde el arranque.

---

## Modelo

- **Algoritmo**: Random Forest Classifier
- **Hiperparámetros**: `n_estimators=100`, `max_depth=10`, `min_samples_leaf=19`, `max_features=log2`, `class_weight=balanced`
- **Split**: 60/20/20 estratificado por clase
- **Métrica principal**: F2-score (prioriza recall, apropiado para diagnóstico médico donde un falso negativo es más costoso que un falso positivo)
- **Métricas registradas**: F2, Recall, Precision, ROC-AUC, PR-AUC en validación y test

### Preprocesamiento (`model/preprocess.py`)

Implementado como un `Pipeline` de sklearn para garantizar que las mismas transformaciones (con los mismos parámetros ajustados en train) se apliquen al momento de predicción:

1. **`_StrokeCleaner`**: `smoking_status = Unknown` → `never smoked`; encoding binario de `gender`, `ever_married`, `Residence_type`
2. **`_BMIGroupImputer`**: imputa BMI faltante con la mediana del grupo etario (bins: 0-10, 11-20, 21-30, 31-70, 71+), fiteada solo sobre train para evitar data leakage
3. **`ColumnTransformer`**: `StandardScaler` sobre variables numéricas; OHE con `drop='first'` sobre `work_type` y `smoking_status`

### Endpoint `/dataset` y mutaciones

El endpoint `/dataset` de la API no devuelve el CSV original crudo. Aplica mutaciones antes de entregarlo:

- Muestrea un 80% de las filas aleatoriamente (`MUTATION_SAMPLE_RATE`)
- Agrega ruido gaussiano a `age`, `avg_glucose_level` y `bmi` (std = 3% de la desviación estándar de cada columna)
- Imputa los BMI nulos con la mediana

Esto simula drift en los datos para que el re-entrenamiento semanal no sea idéntico al anterior.

---

## Desarrollo local

```bash
# Instalar dependencias (requiere uv)
uv sync

# Instalar pre-commit hooks
uv run pre-commit install

# Linter
uv run ruff check .

# Formatter
uv run ruff format .

# Type checker
uv run mypy .
```

---

## Apagar el sistema

```bash
# Apagar contenedores (conserva los datos en volúmenes)
docker compose down

# Apagar y borrar todos los datos (MLflow, Airflow, MinIO)
docker compose down -v
```

---

## Próximos pasos

### Integración de los DAGs ✅

El pipeline ya es cohesivo de extremo a extremo:

- **Encadenado limpieza → comparación** mediante el DAG orquestador `0_stroke_full_pipeline`.
- **Champion automático**: `set_champion_model` (DAG 3) promueve el ganador por F2 directamente sobre `stroke-model@champion`, que es lo que sirve la API.
- **Preprocessing unificado**: hay una única implementación (`model/preprocess.py::build_feature_pipeline`), usada tanto por el entrenamiento de producción como dentro del Pipeline de cada modelo del DAG 3. El DAG 2 ya no preprocesa: solo particiona datos crudos.
