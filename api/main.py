"""FastAPI app — stroke prediction service."""

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import StrokeInput, StrokePrediction

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "stroke-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        _model = mlflow.sklearn.load_model(model_uri)
        print(f"Modelo cargado desde {model_uri}")
    except Exception as exc:
        print(f"Advertencia: no se pudo cargar el modelo: {exc}")
    yield
    _model = None


app = FastAPI(title="Stroke Prediction API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=StrokePrediction)
def predict(data: StrokeInput) -> StrokePrediction:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Entrená primero el pipeline de Airflow.",
        )
    df = pd.DataFrame([data.model_dump()])
    prob = float(_model.predict_proba(df)[0, 1])
    pred = int(_model.predict(df)[0])
    return StrokePrediction(stroke_prediction=pred, stroke_probability=prob)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
