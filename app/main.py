"""FastAPI backend for the eye-disease detection app.

Flow: upload photo -> CNN classifier (src.predict) -> LLM explanation (app.llm)
-> JSON response. Serves a single-page frontend from app/static/.

Run:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app import llm
from src import config

app = FastAPI(title="Eye Disease Screening")

STATIC_DIR = Path(__file__).parent / "static"

# Load the classifier once at startup. If no model is trained yet, we defer the
# error to request time so the app can still boot and serve the frontend.
_predictor = None
_predictor_error: str | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _predictor, _predictor_error
    try:
        from src.predict import Predictor

        _predictor = Predictor()
        print(f"[startup] model loaded, classes: {_predictor.class_names}")
    except Exception as exc:  # noqa: BLE001
        _predictor_error = str(exc)
        print(f"[startup] no model available: {exc}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "model_loaded": _predictor is not None,
        "model_error": _predictor_error,
        "classes": _predictor.class_names if _predictor else None,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model available. Train one first: `python -m src.train`. ({_predictor_error})",
        )

    raw = await file.read()
    try:
        # Validate it's a real image before touching the model.
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, Exception):  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Predictor.predict() takes a path; write to a temp file the model can open.
    with tempfile.NamedTemporaryFile(suffix=".img", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        result = _predictor.predict(tmp.name)

    explanation = llm.explain(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )

    return JSONResponse(
        {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "explanation": explanation["explanation"],
            "explanation_source": explanation["source"],
        }
    )


# Static assets (CSS/JS) served under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
