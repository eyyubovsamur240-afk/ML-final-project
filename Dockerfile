# syntax=docker/dockerfile:1
# Price estimator web interface.
#
#   docker build -t binaaz-estimator .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" binaaz-estimator
#
# The model bundle is NOT baked in: train it once with `python -m app.train`
# (or inside the image, see README) and mount models/. Without it the page
# loads and /api/health reports model_loaded=false.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/predictor.pkl

WORKDIR /app
COPY requirements.txt requirements-app.txt ./
# Behind a TLS-inspecting proxy, pass its CA without baking it into the image:
#   docker build --secret id=pip_ca,src=/path/to/ca.crt -t binaaz-estimator .
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements-app.txt

COPY src ./src
COPY app ./app

RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/models /app/data \
    && chown -R appuser /app/models /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
