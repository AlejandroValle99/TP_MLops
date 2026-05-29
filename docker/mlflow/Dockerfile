FROM python:3.13-slim

RUN pip install --no-cache-dir \
    "mlflow>=2.15.0" \
    psycopg2-binary \
    boto3

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "*"
