# FraudShield scoring service.
#
# The image contains the code, not the model. The model is resolved from the
# MLflow registry at startup, which is the point: baking a model into an
# image means the running container and the registry can disagree about what
# is promoted, and nothing would notice. Mount the store, or point
# MLFLOW_TRACKING_URI at a shared one.
#
#   docker build -t fraudshield:latest .
#   docker run --rm -p 8080:8080 \
#     -v "$PWD/mlflow.db:/app/mlflow.db:ro" \
#     -v "$PWD/mlartifacts:/app/mlartifacts:ro" \
#     -v "$PWD/reports:/app/reports:ro" \
#     fraudshield:latest

FROM python:3.12-slim AS base

# libgomp is required by both xgboost and lightgbm; the slim image omits it
# and the failure is an unhelpful import error at startup rather than a
# build-time one.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: they change far less often than the source, so this
# layer stays cached across ordinary code edits.
COPY requirements.txt requirements.lock.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.lock.txt \
    && pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"

COPY src/ ./src/
COPY examples/ ./examples/

# Non-root: the service reads a model and writes nothing.
RUN useradd --create-home --uid 10001 fraudshield \
    && chown -R fraudshield:fraudshield /app
USER fraudshield

# Pinned inside the image too, so a container scheduled onto a larger host
# does not silently retrain or rescore with a different thread count.
ENV OMP_NUM_THREADS=2 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

EXPOSE 8080

# /health resolves the loaded model version, so a container that started but
# failed to resolve the registry reports unhealthy rather than accepting
# traffic it cannot score.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
