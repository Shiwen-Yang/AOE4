FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY backend ./backend

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENV AOE4_DB_PATH=/artifacts/aoe4.duckdb
ENV AOE4_MODEL_PATH=/artifacts/models/aoe4_predict/lgbm_s10s11s12.txt
ENV AOE4_MODEL_META_PATH=/artifacts/models/aoe4_predict/lgbm_s10s11s12_meta.json
ENV AOE4_MODEL_VERSION=lgbm_s10s11s12
ENV AOE4_DELTA_MODEL_PATH=/artifacts/models/ratings_delta/lgbm_delta.txt
ENV AOE4_DELTA_PARAMETRIC_PATH=/artifacts/models/p3_parametric.json

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
