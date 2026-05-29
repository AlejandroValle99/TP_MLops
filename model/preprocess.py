import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "stroke"

_NUM_COLS = ["age", "avg_glucose_level", "bmi"]
_BIN_COLS = [
    "hypertension",
    "heart_disease",
    "gender",
    "ever_married",
    "Residence_type",
]
_WORK_COL = ["work_type"]
_SMOKE_COL = ["smoking_status"]

_AGE_BINS = [0, 10, 20, 30, 70, np.inf]
_AGE_LABELS = ["0-10", "11-20", "21-30", "31-70", "71+"]


class _StrokeCleaner(BaseEstimator, TransformerMixin):
    """Applies deterministic cleaning and binary encoding to the raw stroke DataFrame."""

    def fit(self, X: pd.DataFrame, y=None) -> "_StrokeCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # "Unknown" smoking maps to "never smoked" per notebook 01 decision
        df["smoking_status"] = df["smoking_status"].replace("Unknown", "never smoked")
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)
        df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0}).astype(int)
        df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0}).astype(int)
        return df


class _BMIGroupImputer(BaseEstimator, TransformerMixin):
    """Imputes missing BMI using age-group medians fitted only on training data."""

    def fit(self, X: pd.DataFrame, y=None) -> "_BMIGroupImputer":
        df = X.copy()
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["_age_group"] = pd.cut(
            df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, include_lowest=True
        )
        medians = df.groupby("_age_group", observed=False)["bmi"].median()
        self.group_medians_: dict[str, float] = {str(k): float(v) for k, v in medians.items()}
        self.global_median_: float = float(df["bmi"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["_age_group"] = pd.cut(
            df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, include_lowest=True
        )
        fill = df["_age_group"].astype(str).map(self.group_medians_).astype(float)
        df["bmi"] = df["bmi"].fillna(fill.fillna(self.global_median_))
        return df.drop(columns=["_age_group"])


def _build_column_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), _NUM_COLS),
            ("pass", "passthrough", _BIN_COLS),
            (
                "work",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                _WORK_COL,
            ),
            (
                "smoke",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                _SMOKE_COL,
            ),
        ],
        remainder="drop",
    )


def build_feature_pipeline() -> Pipeline:
    """Returns an unfitted sklearn Pipeline that replicates the notebook 01 preprocessing."""
    return Pipeline(
        [
            ("cleaner", _StrokeCleaner()),
            ("bmi_imputer", _BMIGroupImputer()),
            ("features", _build_column_transformer()),
        ]
    )


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads raw CSV, drops id and gender='Other' rows, returns (X, y)."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"]
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y
