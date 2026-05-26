import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_tuned.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "data/encoder.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "data/preprocessor.pkl")

CAT_COLS = ["work_type", "smoking_status"]

store: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store["model"] = joblib.load(MODEL_PATH)
    store["encoder"] = joblib.load(ENCODER_PATH)
    store["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    yield
    store.clear()


class StrokeInput(BaseModel):
    gender: Literal["Male", "Female"]
    age: Annotated[float, Field(ge=0, le=120)]
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal[
        "children", "Govt_job", "Never_worked", "Private", "Self-employed"
    ]
    Residence_type: Literal["Rural", "Urban"]
    avg_glucose_level: Annotated[float, Field(gt=0)]
    bmi: Annotated[float, Field(gt=0)]
    smoking_status: Literal[
        "formerly smoked", "never smoked", "smokes", "Unknown"
    ]


class StrokeOutput(BaseModel):
    stroke_probability: float
    stroke_prediction: int


app = FastAPI(lifespan=lifespan)


def preprocess(data: StrokeInput) -> np.ndarray:
    smoking = (
        "never smoked"
        if data.smoking_status == "Unknown"
        else data.smoking_status
    )

    encoded = store["encoder"].transform(
        pd.DataFrame(
            [{"work_type": data.work_type, "smoking_status": smoking}]
        )
    )
    encoded_cols = store["encoder"].get_feature_names_out(CAT_COLS)
    encoded_vals = dict(zip(encoded_cols, encoded[0], strict=True))

    row = pd.DataFrame(
        [
            {
                "age": data.age,
                "avg_glucose_level": data.avg_glucose_level,
                "bmi": data.bmi,
                "hypertension": data.hypertension,
                "heart_disease": data.heart_disease,
                "gender": 0 if data.gender == "Male" else 1,
                "ever_married": 1 if data.ever_married == "Yes" else 0,
                "Residence_type": 1 if data.Residence_type == "Urban" else 0,
                **encoded_vals,
            }
        ]
    )

    return store["preprocessor"].transform(row)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=StrokeOutput)
def predict(data: StrokeInput) -> StrokeOutput:
    if not store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = preprocess(data)
    proba = store["model"].predict_proba(features)
    stroke_prob = float(proba[0][1])
    stroke_pred = int(stroke_prob >= 0.5)

    return StrokeOutput(
        stroke_probability=stroke_prob,
        stroke_prediction=stroke_pred,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
