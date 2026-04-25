"""
FastAPI Backend for Tabla Taal & Bol Identification.

Production-grade REST API for classifying Tabla audio recordings.
Designed for integration with mobile apps and web frontends.

Run with:
    python -m api.main
    # or
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import TablaPredictor


# =============================================================================
# Pydantic Response Models
# =============================================================================

class PredictionItem(BaseModel):
    label: str
    display_name: str
    confidence: float
    beats: Optional[int] = None
    vibhag: Optional[str] = None

class TaskResult(BaseModel):
    label: str
    display_name: str
    confidence: float
    top_k: List[PredictionItem]

class AudioInfo(BaseModel):
    duration_seconds: float
    sample_rate: int
    source: str

class PredictionResponse(BaseModel):
    status: str
    task: str
    bol: Optional[TaskResult] = None
    taal: Optional[TaskResult] = None
    audio_info: AudioInfo
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str

class ModelInfo(BaseModel):
    architecture: str
    accuracy: float
    classes: List[str]
    input_duration_seconds: float

class ModelsResponse(BaseModel):
    bol_model: ModelInfo
    taal_model: ModelInfo
    device: str
    sample_rate: int


# =============================================================================
# FastAPI Application
# =============================================================================

# Global predictor instance
predictor: Optional[TablaPredictor] = None

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@asynccontextmanager
async def lifespan(app):
    """Load models on startup, cleanup on shutdown."""
    global predictor
    try:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
        predictor = TablaPredictor(models_dir=models_dir)
    except Exception as e:
        print(f"ERROR loading models: {e}")
        traceback.print_exc()
    yield
    predictor = None


app = FastAPI(
    title="Tabla Taal & Bol Identifier",
    description=(
        "Deep Learning API for identifying Tabla Taals (rhythmic patterns) "
        "and Bols (strokes) from audio recordings. "
        "Powered by ResidualCNN and CRNN models trained with 5-fold cross-validation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests (mobile apps, web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (web demo)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Serve the web demo page."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"message": "Tabla API is running. Visit /docs for API documentation."}
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if predictor else "models_not_loaded",
        models_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "unknown"
    )


@app.get("/api/v1/models", response_model=ModelsResponse, tags=["System"])
async def get_models_info():
    """Get information about loaded models."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    info = predictor.get_model_info()
    return ModelsResponse(**info)


@app.get("/api/v1/classes", tags=["System"])
async def get_classes():
    """Get available Bol and Taal class names."""
    return {
        "bol_classes": TablaPredictor.BOL_CLASSES,
        "bol_display_names": TablaPredictor.BOL_DISPLAY,
        "taal_classes": TablaPredictor.TAAL_CLASSES,
        "taal_display_names": TablaPredictor.TAAL_DISPLAY,
        "taal_info": TablaPredictor.TAAL_INFO,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A)"),
    task: str = Query(
        "both",
        description="What to predict: 'bol', 'taal', or 'both'",
        pattern="^(bol|taal|both)$"
    )
):
    """
    Predict Tabla Taal and/or Bol from an uploaded audio file.
    
    Upload a recording of Tabla playing and get:
    - **Bol prediction**: Which stroke (Dha, Dhin, Ga, Ka, T, Ta, Tin, Tit)
    - **Taal prediction**: Which rhythm (Trital, Ektal, Jhaptal, Rupak, etc.)
    - **Confidence scores**: How sure the model is about each prediction
    - **Top-k results**: All class probabilities ranked
    
    Supports WAV, MP3, OGG, FLAC, M4A audio formats.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file content
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
        )

    # Save to temp file for librosa (handles format detection better with file path)
    tmp_path = None
    try:
        suffix = ext if ext else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Run prediction
        result = predictor.predict_from_file(tmp_path, task=task)

        return PredictionResponse(
            status=result["status"],
            task=result["task"],
            bol=TaskResult(**result["bol"]) if "bol" in result else None,
            taal=TaskResult(**result["taal"]) if "taal" in result else None,
            audio_info=AudioInfo(**result["audio_info"]),
            processing_time_ms=result["processing_time_ms"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/v1/predict/sequence", tags=["Prediction"])
async def predict_sequence(
    file: UploadFile = File(..., description="Audio file with Tabla playing"),
):
    """
    Predict the full Bol sequence in a Tabla recording (Combined Approach).
    
    Uses a hybrid pipeline:
    1. **Taal Detection** — Identifies the rhythmic pattern (Dadra, Trital, etc.)
    2. **Template Mapping** — Maps onsets to expected Bols from the Taal's theka
    3. **Model Classification** — Classifies each audio segment independently
    4. **Cross-Validation** — Combines both sources for maximum accuracy
    
    Returns a timeline of Bols with timestamps, source attribution,
    and confidence scores that can be synced with audio playback.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    tmp_path = None
    try:
        suffix = ext if ext else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = predictor.predict_combined_from_file(tmp_path)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"}
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
