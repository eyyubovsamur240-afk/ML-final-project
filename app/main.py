"""
The web interface: one HTML page plus a small JSON API.

    uvicorn app.main:app --port 8000        # then open http://localhost:8000

Endpoints
    GET  /                 the form (static/index.html)
    GET  /api/health       liveness + whether a model is loaded
    GET  /api/options      dropdown values and input ranges for the form
    GET  /api/model        model card: held-out test metrics, training metadata
    POST /api/predict      ListingIn -> PredictionOut
    GET  /docs             interactive API docs (OpenAPI)

The model bundle path comes from the MODEL_PATH environment variable
(default models/predictor.pkl); create it with ``python -m app.train``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .predictor import DEFAULT_MODEL_PATH, Listing, PricePredictor
from .schemas import ListingIn, PredictionOut

log = logging.getLogger("app")
STATIC_DIR = Path(__file__).parent / "static"


def create_app(model: PricePredictor | None = None, model_path: str | Path | None = None) -> FastAPI:
    """App factory. Pass ``model`` directly (tests) or let it load from disk."""
    path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.model is None and path.exists():
            app.state.model = PricePredictor.load(path)
            log.info("Loaded model bundle %s", path)
        elif app.state.model is None:
            log.warning("No model at %s; run `python -m app.train` first", path)
        yield

    app = FastAPI(title="bina.az price estimator", version="1.0",
                  description="Price and premium-tier estimates from the project's "
                              "from-scratch random forest and RFF Pegasos SVM.",
                  lifespan=lifespan)
    app.state.model = model

    def get_model() -> PricePredictor:
        if app.state.model is None:
            raise HTTPException(503, f"No model loaded. Train one with `python -m app.train` "
                                     f"(expected at {path}).")
        return app.state.model

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.get("/api/options")
    def options(model: PricePredictor = Depends(get_model)):
        return model.options()

    @app.get("/api/model")
    def model_card(model: PricePredictor = Depends(get_model)):
        return model.card()

    @app.post("/api/predict", response_model=PredictionOut)
    def predict(listing: ListingIn, model: PricePredictor = Depends(get_model)):
        if listing.location is not None and listing.location not in model.gazetteer_:
            raise HTTPException(422, f"Unknown location {listing.location!r}; see /api/options")
        return asdict(model.predict(Listing(**listing.model_dump())))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
