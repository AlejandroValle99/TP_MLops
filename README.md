# TP MLOps — Predicción de Stroke

Pipeline completo de MLOps para predecir riesgo de ACV (stroke) sobre el [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Todo el sistema corre en contenedores Docker con un único comando.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                         │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Airflow  │───▶│  MLflow  │───▶│  MinIO   │             │
│  │ :8080    │    │  :5000   │    │  :9000   │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│       │               │                │                   │
│       │         ┌─────┴──────┐         │                   │
│       └────────▶│ PostgreSQL │◀────────┘                   │
│                 │  :5432     │                             │
│                 └────────────┘                             │
│                                                             │
│  ┌──────────┐                                              │
│  │ FastAPI  │  ◀── consume modelo champion desde MLflow   │
│  │  :8000   │                                              │
│  └──────────┘                                              │
└─────────────────────────────────────────────────────────────┘
```

### Qué hace cada servicio

| Servicio | Puerto | Rol |
|---|---|---|
| **Airflow** | 8080 | Orquesta el pipeline: valida el CSV y ejecuta el entrenamiento cada semana |
| **MLflow** | 5000 | Registra experimentos, métricas, parámetros y versiones del modelo |
| **MinIO** | 9000 / 9001 | Almacena los artefactos de MLflow (archivos del modelo entrenado) |
| **PostgreSQL** | 5432 | Base de datos de Airflow y backend de MLflow |
| **FastAPI** | 8000 | API REST que sirve predicciones usando el modelo `champion` de MLflow |

---

## Estructura del proyecto

```
TP_MLops/
├── api/
│   ├── main.py          # FastAPI: endpoints /health y /predict
│   └── schemas.py       # Esquemas de entrada/salida de la API
├── dags/
│   └── stroke_pipeline.py  # DAG de Airflow con las tareas del pipeline
├── data/
│   └── healthcare-dataset-stroke-data.csv  # Dataset original
├── docker/
│   ├── airflow/
│   │   ├── Dockerfile       # Imagen de Airflow con dependencias ML
│   │   └── requirements.txt
│   ├── mlflow/
│   │   └── Dockerfile       # Imagen de MLflow con soporte a PostgreSQL y S3
│   └── postgres/
│       └── init.sql         # Crea la base de datos de MLflow al iniciar
├── model/
│   ├── preprocess.py    # Pipeline de sklearn: limpieza, imputación, encoding
│   └── train.py         # Entrenamiento del Random Forest y registro en MLflow
├── docker-compose.yml   # Orquestación de todos los servicios
├── Dockerfile           # Imagen de la API FastAPI
└── .env.example         # Variables de entorno necesarias (copiar a .env)
```

---

## Cómo levantar el sistema

### 1. Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

El archivo `.env` ya viene con valores por defecto funcionales. Solo modificarlo si se quiere cambiar alguna credencial.

### 3. Levantar todos los servicios

```bash
docker compose up -d
```

La primera vez tarda unos minutos porque construye las imágenes. Una vez que todos los servicios están `healthy`, los accesos son:

| Interfaz | URL | Usuario | Contraseña |
|---|---|---|---|
| Airflow | http://localhost:8080 | admin | admin |
| MLflow | http://localhost:5000 | — | — |
| MinIO Console | http://localhost:9001 | minioadmin | minioadmin_secret |
| API docs | http://localhost:8000/docs | — | — |

### 4. Verificar que todo levantó bien

```bash
docker compose ps
```

Todos los servicios deben mostrar `healthy` (excepto `airflow-init` y `minio-init` que son tareas de inicialización que terminan solas).

---

## Cómo funciona el pipeline

### Entrenamiento (Airflow)

El DAG `stroke_prediction_pipeline` en Airflow tiene dos tareas que se ejecutan en orden:

```
validate_data  ──▶  train_model
```

1. **`validate_data`**: verifica que el CSV existe, tiene las columnas correctas y al menos 100 filas.
2. **`train_model`**: ejecuta el entrenamiento completo y registra el modelo en MLflow.

El DAG está configurado para correr automáticamente una vez por semana. También se puede ejecutar manualmente desde la UI de Airflow con el botón **Trigger DAG** (▶).

### Preprocesamiento (`model/preprocess.py`)

Replica exactamente el notebook `01_data_processing_.ipynb`:

- **Limpieza**: `smoking_status = Unknown` → `never smoked`; se eliminan las filas con `gender = Other`
- **Encoding binario**: `gender` (Male→0, Female→1), `ever_married` (Yes→1, No→0), `Residence_type` (Urban→1, Rural→0)
- **Imputación de BMI**: los valores faltantes se imputan con la mediana del grupo de edad correspondiente (bins: 0-10, 11-20, 21-30, 31-70, 71+), calculada solo sobre el set de entrenamiento para evitar data leakage
- **Encoding categórico**: OHE con `drop='first'` para `work_type` y `smoking_status`
- **Escalado**: `StandardScaler` sobre `age`, `avg_glucose_level` y `bmi`

Todo esto está encapsulado en un `Pipeline` de sklearn, lo que garantiza que las mismas transformaciones (con los mismos parámetros ajustados en train) se aplican al hacer predicciones.

### Entrenamiento (`model/train.py`)

- Split 60/20/20 estratificado por clase (igual que el notebook)
- **Modelo**: Random Forest con los hiperparámetros óptimos encontrados por Optuna en el notebook `02_training_and_tuning.ipynb`:
  - `n_estimators=100`, `max_depth=10`, `min_samples_leaf=19`, `max_features=log2`, `class_weight=balanced`
- **Métrica principal**: F2-score (prioriza recall sobre precisión, adecuado para diagnóstico médico)
- Registra parámetros y métricas en MLflow
- Guarda el modelo en el registro de MLflow y le asigna el alias **`champion`** automáticamente

### Servicio de predicciones (FastAPI)

Al iniciar, la API carga automáticamente el modelo con alias `champion` desde MLflow. Expone dos endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Devuelve estado del servicio y si el modelo está cargado |
| POST | `/predict` | Recibe datos de un paciente y devuelve la predicción |

#### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
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

Respuesta:
```json
{
  "stroke_prediction": 1,
  "stroke_probability": 0.79
}
```

---

## Flujo completo de un re-entrenamiento

Cuando se quiere reentrenar el modelo (por ejemplo, con nuevos datos):

1. Reemplazar el CSV en `data/healthcare-dataset-stroke-data.csv`
2. En Airflow (http://localhost:8080), hacer **Trigger DAG** en `stroke_prediction_pipeline`
3. Esperar a que ambas tareas queden en verde
4. Reiniciar la API para que cargue el nuevo modelo champion:
   ```bash
   docker compose restart api
   ```

MLflow guarda el historial de todos los entrenamientos. Se puede comparar métricas entre versiones desde http://localhost:5000.

---

## Apagar el sistema

```bash
# Apagar contenedores (conserva los datos)
docker compose down

# Apagar y borrar todos los datos (MLflow, Airflow, MinIO)
docker compose down -v
```
