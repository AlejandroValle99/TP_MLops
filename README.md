# TP MLOps — Arquitectura del proyecto

Resumen breve
- **Propósito:** Proyecto de MLOps que agrupa una API de inferencia, orquestación con Airflow, trazabilidad con MLflow y servicios de infraestructura como Postgres, MinIO y broker de tareas.
- **Nota:** Ignorar la carpeta `NO_MERGE`, contiene ejemplos alternativos que no forman parte de la arquitectura principal.

Estructura principal
- **`main.py`**: entrada de la API FastAPI.
- **`docker-compose.yml`**: orquestación de todos los servicios.
- **`Dockerfile`**: imagen principal para la API.
- **`airflow/`**: configuración de Airflow y Dockerfile.
  - `airflow/config/requirements.txt`: dependencias adicionales para la imagen de Airflow.
  - `airflow/dags/`: DAGs de Airflow.
  - `airflow/logs/`, `airflow/plugins/`: volúmenes para Airflow.
  - `airflow/secrets/`: secretos de Airflow montados en `/opt/secrets`.
- **`mlflow/`**: servidor de MLflow, Dockerfile y `requirements.txt`.
- **`postgres/`**: imagen de Postgres y `mlflow.sql` para crear la DB `mlflow_db`.
- **`notebooks/`**: notebooks de análisis, entrenamiento y evaluación.

Servicios de `docker-compose.yml`
- **postgres**: base de datos principal.
- **valkey**: broker de tareas actual configurado en lugar de Redis.
- **s3 / minio**: almacenamiento de artefactos y buckets.
- **mlflow**: servidor de MLflow con backend en Postgres y artefactos en MinIO.
- **airflow-***: despliegue completo de Airflow (api-server, scheduler, worker, triggerer, init, cli, dag-processor).

Cambios aplicados
- Se movieron los secretos de Airflow a `airflow/secrets/`.
- Se validó que `docker-compose.yml` monta `airflow/secrets` en `/opt/secrets`.
- Se dejó constancia de que `AIRFLOW__CORE__FERNET_KEY` está vacío y debe completarse si se requiere cifrado de secretos.
- Se agregaron perfiles faltantes a `copy_data_to_s3` para que coincida con su dependencia `create_s3_buckets`.

Estado actual de la configuración
- Airflow carga secretos desde:
  - `/opt/secrets/connections.yaml`
  - `/opt/secrets/variables.yaml`
- La clave Fernet aún no está configurada y debe generarse si se usa cifrado.
- El broker de tareas está definido como `valkey`, pero la URL de Airflow todavía apunta a `redis://:@redis:6379/0`. Esto debe corregirse para que Airflow use el mismo servicio de broker.

Recomendaciones
1. Actualizar el broker de Airflow y el servicio de tareas para que coincidan.
2. Generar y fijar `AIRFLOW__CORE__FERNET_KEY` si se espera cifrar datos sensibles.
3. Mantener `airflow/secrets/` como la ubicación activa de secretos para Airflow.

Cómo arrancar el entorno
```bash
docker-compose --profile all up --build
```

Acceso a servicios
- Airflow webserver: `http://localhost:8080`
- MLflow: `http://localhost:5000`
- API FastAPI: `http://localhost:8000`

TODO:
 - README: add imagen arquitectura 
 - dags
 - batch processing
 - notebooks with mlflow

