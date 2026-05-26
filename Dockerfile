FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY main.py .
COPY models/ models/
COPY data/ data/

EXPOSE 8000

ENV MODEL_PATH="models/random_forest_tuned.pkl"
ENV ENCODER_PATH="data/encoder.pkl"
ENV PREPROCESSOR_PATH="data/preprocessor.pkl"
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_NAME=""
ENV MLFLOW_MODEL_STAGE="Develop"

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
