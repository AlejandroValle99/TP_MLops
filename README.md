# TP MLOps — Predicción de Stroke

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

| Servicio | Puerto | Rol |
|---|---|---|
| **FastAPI** | 8000 | Sirve predicciones; carga el modelo `champion` desde MLflow al iniciar |
| **Airflow webserver** | 8080 | UI de orquestación; ejecuta los DAGs de entrenamiento |
| **MLflow** | 5001 | Tracking de experimentos, métricas, parámetros y model registry |
| **MinIO** | 9000 / 9001 | Artifact store S3-compatible; guarda modelos entrenados y el dataset |
| **PostgreSQL** | 5432 | Backend de metadatos de Airflow y MLflow |

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

| Interfaz | URL | Usuario | Contraseña |
|---|---|---|---|
| API docs | http://localhost:8000/docs | — | — |
| Airflow | http://localhost:8080 | admin | admin |
| MLflow | http://localhost:5001 | — | — |
| MinIO Console | http://localhost:9001 | minioadmin | minioadmin_secret |

### 3. Verificar el estado

```bash
docker compose ps
```

Los servicios `minio-init`, `model-init` y `airflow-init` finalizan solos (no quedan corriendo). El resto debe aparecer como `healthy`.

---

## API

La API carga el modelo con alias `champion` desde MLflow al iniciar. Si no hay modelo disponible, `/health` devuelve 503.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y si el modelo está cargado |
| GET | `/model-info` | Versión activa, run_id y métricas del modelo champion |
| POST | `/predict` | Recibe datos de un paciente y devuelve predicción y probabilidad |
| GET | `/dataset` | Descarga el dataset desde MinIO con mutaciones aplicadas |

### Campos de entrada para `/predict`

| Campo | Tipo | Valores válidos |
|---|---|---|
| `gender` | string | `"Male"`, `"Female"` |
| `age` | float | > 0 |
| `hypertension` | int | `0`, `1` |
| `heart_disease` | int | `0`, `1` |
| `ever_married` | string | `"Yes"`, `"No"` |
| `work_type` | string | `"Private"`, `"Self-employed"`, `"Govt_job"`, `"children"`, `"Never_worked"` |
| `Residence_type` | string | `"Urban"`, `"Rural"` |
| `avg_glucose_level` | float | > 0 |
| `bmi` | float \| null | > 0, acepta null |
| `smoking_status` | string | `"never smoked"`, `"formerly smoked"`, `"smokes"`, `"Unknown"` |

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

El sistema tiene tres DAGs con propósitos distintos. Actualmente son independientes entre sí (ver [Próximos pasos](#próximos-pasos)).

### DAG 1: `stroke_prediction_pipeline` — producción

**Schedule**: semanal automático.

```
fetch_data ──▶ validate_data ──▶ train_model
```

| Tarea | Qué hace |
|---|---|
| `fetch_data` | Llama al endpoint `/dataset` de la API, que devuelve el CSV desde MinIO con mutaciones aplicadas, y lo escribe en disco |
| `validate_data` | Verifica columnas correctas y mínimo 100 filas |
| `train_model` | Entrena el Random Forest, loggea métricas en MLflow y promueve el modelo como `champion` |

Este DAG es el pipeline "vivo". Para triggerearlo manualmente: **Airflow UI → stroke_prediction_pipeline → ▶ Trigger DAG**.

### DAG 2: `stroke_data_cleaning` — ETL exploratorio

**Schedule**: ninguno (disparo manual).

```
validate_source ──▶ load_and_split ──▶ impute_bmi ──▶ encode_features ──▶ scale_features ──▶ upload_to_minio
```

Procesa el CSV local en 6 pasos independientes y persiste cada etapa en MinIO (`s3://mlflow-artifacts/processed/`). La particularidad es que cada paso escribe su resultado en S3 y el siguiente lo lee desde ahí, lo que permite inspeccionar intermedios y hace cada tarea reutilizable de forma independiente.

Produce los splits finales en `processed/final/` que usa el DAG 3.

### DAG 3: `etl_train_models_process_taskflow` — comparación de modelos

**Schedule**: ninguno (disparo manual). **Requiere que el DAG 2 haya corrido antes.**

```
check_data_to_process ──▶ create_base_model  ──┐
                      ──▶ create_knn_model   ──┤
                      ──▶ create_decision_tree ─┤──▶ (resultados en MLflow)
                      ──▶ create_xgboost_model ─┤
                      ──▶ create_random_forest ─┘
```

Lee los splits de `processed/final/`, entrena cinco modelos en paralelo con búsqueda de hiperparámetros via **Optuna** (50 trials cada uno) y loggea métricas, parámetros y gráficos de evaluación en MLflow. Está implementado con la **TaskFlow API** de Airflow.

Los modelos que entrena son: baseline (solo edad), KNN, Decision Tree, XGBoost y Random Forest.

### Flujo de los tres DAGs

```
[DAG 2] stroke_data_cleaning   →   [DAG 3] etl_train_models   →  comparar en MLflow
         (manual, una vez)               (manual, una vez)           y elegir modelo

[DAG 1] stroke_pipeline                                         →  mantiene el sistema
         (automático, semanal)                                       actualizado
```

Los DAGs 2 y 3 son herramientas de investigación para seleccionar el mejor modelo. El DAG 1 es el pipeline de producción que mantiene el modelo `champion` actualizado cada semana.

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

### Integración de los DAGs

Los tres DAGs son actualmente independientes. El DAG 1 tiene su propio preprocessing inline (`model/preprocess.py`) y no usa los datos que produce el DAG 2. El DAG 3 compara modelos pero ninguno de sus resultados impacta en el modelo `champion`.

Para que el pipeline sea cohesivo de extremo a extremo:

- **Encadenar DAG 2 → DAG 3**: agregar un `TriggerDagRunOperator` al final del DAG 2 para que el DAG 3 corra automáticamente cuando los datos estén listos
- **Conectar DAG 3 con producción**: agregar una tarea al DAG 3 que evalúe el mejor modelo encontrado por Optuna y, si supera al `champion` actual en F2-score, lo promueva en el MLflow registry
- **Unificar el preprocessing**: hacer que el DAG 1 consuma los datos procesados por el DAG 2 en lugar de tener su propio pipeline inline, de modo que todos los flujos usen exactamente la misma lógica de preprocesamiento

### Otras mejoras

- **Recarga del modelo en la API sin restart**: implementar un endpoint `/reload-model` o un mecanismo de polling para que la API detecte automáticamente cuando hay un nuevo `champion` en MLflow
- **Tests de integración**: agregar tests que verifiquen el contrato del endpoint `/predict` y la consistencia entre el preprocessing del pipeline y el de la API
- **Monitoreo de drift**: integrar una herramienta como Evidently para detectar drift entre el dataset con el que se entrenó y los datos que llegan vía `/dataset` en cada ciclo semanal
