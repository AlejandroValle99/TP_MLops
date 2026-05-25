# TP MLOps — Stroke Prediction

API de predicción de riesgo de ACV (stroke) construida con FastAPI y desplegada con Docker. El modelo subyacente es un Random Forest entrenado sobre el [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

## Requisitos

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- Docker

## Instalación local

```bash
# Clonar el repositorio
git clone <repo-url>
cd TP_MLops

# Instalar dependencias
uv sync
```

## Ejecución

### Local

```bash
uv run uvicorn main:app --reload
```

La API queda disponible en `http://localhost:8000`. La documentación interactiva en `http://localhost:8000/docs`.

### Docker

```bash
# Construir imagen
docker build -t tp-mlops .

# Correr contenedor
docker run -p 8000:8000 tp-mlops
```

## Endpoints

| Método | Ruta      | Descripción                         |
| ------ | --------- | ----------------------------------- |
| GET    | `/health` | Verificación de estado del servicio |

## Desarrollo

```bash
# Instalar dependencias de desarrollo (linters, hooks)
uv sync --group dev

# Instalar pre-commit hooks
uv run pre-commit install

# Correr linter
uv run ruff check .

# Correr type checker
uv run mypy .
```
