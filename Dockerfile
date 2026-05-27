FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi[standard]>=0.115.0" \
    "scikit-learn>=1.5.0" \
    "pandas>=2.2.0" \
    "numpy>=1.26.0" \
    "mlflow>=2.15.0" \
    boto3

COPY api/ ./api/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
