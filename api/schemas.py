from pydantic import BaseModel, Field


class StrokeInput(BaseModel):
    gender: str = Field(..., examples=["Male"])
    age: float = Field(..., examples=[67.0])
    hypertension: int = Field(..., ge=0, le=1, examples=[0])
    heart_disease: int = Field(..., ge=0, le=1, examples=[1])
    ever_married: str = Field(..., examples=["Yes"])
    work_type: str = Field(..., examples=["Private"])
    Residence_type: str = Field(..., examples=["Urban"])
    avg_glucose_level: float = Field(..., examples=[228.69])
    bmi: float | None = Field(None, examples=[36.6])
    smoking_status: str = Field(..., examples=["formerly smoked"])


class StrokePrediction(BaseModel):
    stroke_prediction: int
    stroke_probability: float
