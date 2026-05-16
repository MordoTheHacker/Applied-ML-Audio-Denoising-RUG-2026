import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io
import time
import logging
import numpy as np
import soundfile as sf
import librosa
import torch
import tempfile, os
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from contextlib import asynccontextmanager

from models.spectral_subtraction import SpectralSubtraction
from models.geometric_subtraction import GeometricSpectralSubtraction
from models.mlp import SpeechMLP
from models.mlp import enhance_file as mlp_enhance
from models.unet import UNet
from models.unet import enhance_file as unet_enhance

from evaluate import evaluate_all

"""
Audio Denoising API — FastAPI Application

Provides RESTful endpoints for speech enhancement using trained
MLP and U-Net models as well as statistical spectral subtraction methods.
Accepts noisy audio files and returns enhanced (denoised) audio along with quality
metrics.

Endpoints:
    GET  /                    — API info and health check
    GET  /models              — List available models and their status
    GET  /models/{model_name} — Info and metadata for a specific model
    POST /enhance             — Enhance a noisy audio file
    POST /evaluate            — Enhance + compute quality metrics
    GET  /docs                — Interactive Swagger UI documentation
    GET  /redoc               — ReDoc documentation
"""

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SR = 16000
MAX_DURATION_SECONDS = 60
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".ogg"}
MODELS_DIR = Path("outputs/models")
RESULTS_DIR = Path("outputs/results")

# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────

class ModelName(str, Enum):
    spectral_subtraction = "spectral_subtraction"
    geometric_subtraction = "geometric_subtraction"
    mlp = "mlp"
    unet = "unet"

# Lazy-loaded model cache
_model_cache = {}

def load_spectral_subtraction():
    """Load spectral subtraction model"""
    return SpectralSubtraction(
        sr=SR,
        frame_len=0.025,
        frame_shift=0.010,
        noise_frames=20,
        lambda_n=0.95,
        alpha=1.5,
        beta=0.002,
    )

def load_geometric_subtraction():
    """Load geometric spectral subtraction model"""
    return GeometricSpectralSubtraction(sr=SR)

def load_mlp():
    """Load trained MLP model and normalization stats."""

    model_dir = MODELS_DIR / "mlp"
    weights_path = model_dir / "best_model.pt"
    mean_path    = model_dir / "norm_mean.npy"
    std_path     = model_dir / "norm_std.npy"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"MLP weights not found at {weights_path}. "
            "Train the model first: python src/models/mlp.py"
        )

    mean = np.load(mean_path)
    std  = np.load(std_path)

    # Reconstruct model architecture
    context_frames = 5
    freq_bins = 257
    window_size = 2 * context_frames + 1
    input_dim = window_size * freq_bins

    model = SpeechMLP(
        input_dim=input_dim,
        output_dim=freq_bins,
        hidden_dim=1024,
        n_layers=4,
        dropout=0.2,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    return model, mean, std

def load_unet():
    """Load trained U-Net model and normalization stats."""

    model_dir = MODELS_DIR / "unet"
    weights_path = model_dir / "best_model.pt"
    mean_path    = model_dir / "norm_mean.npy"
    std_path     = model_dir / "norm_std.npy"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"U-Net weights not found at {weights_path}. "
            "Train the model first: python src/models/unet.py"
        )

    mean = float(np.load(mean_path))
    std  = float(np.load(std_path))

    model = UNet(
        base_filters=32,
        dropout_enc=0.1,
        dropout_bottleneck=0.5,
        dropout_dec=0.1,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    return model, mean, std

def get_model(model_name: ModelName):
    """Get model from cache or load it."""
    if model_name not in _model_cache:
        logger.info(f"Loading model: {model_name}")
        try:
            if model_name == ModelName.spectral_subtraction:
                _model_cache[model_name] = load_spectral_subtraction()
            elif model_name == ModelName.geometric_subtraction:
                _model_cache[model_name] = load_geometric_subtraction()
            elif model_name == ModelName.mlp:
                _model_cache[model_name] = load_mlp()
            elif model_name == ModelName.unet:
                _model_cache[model_name] = load_unet()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' is not available: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model '{model_name}': {str(e)}"
            )
    return _model_cache[model_name]

def check_model_available(model_name: ModelName) -> bool:
    """Check if model weights exist on disk."""
    if model_name in (ModelName.spectral_subtraction, ModelName.geometric_subtraction):
        return True  # No weights needed
    weights = MODELS_DIR / model_name.value / "best_model.pt"
    return weights.exists()

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    logger.info("Starting Audio Denoising API...")
    logger.info("Pre-loading classical algorithms...")

    classical_models = [
        ModelName.spectral_subtraction,
        ModelName.geometric_subtraction
    ]

    for model in classical_models:
        try:
            get_model(model)
            logger.info(f" -> {model.value} baseline ready.")
        except Exception as e:
            logger.warning(f"Could not pre-load {model.value}: {e}")

    for model_name in [ModelName.mlp, ModelName.unet]:
        available = check_model_available(model_name)
        status = "available" if available else "weights not found"
        logger.info(f"  {model_name.value}: {status}")

    logger.info("API ready. Visit http://localhost:8000/docs")
    yield
    # Shutdown (runs after yield when server stops)
    logger.info("Shutting down Audio Denoising API...")
    _model_cache.clear()

app = FastAPI(
    title="Audio Denoising API",
    lifespan=lifespan,
    description="""
## Speech Enhancement API

This API provides monaural speech enhancement using classical signal processing 
and lightweight machine learning models trained on the **VoiceBank+DEMAND** dataset.

### Background

Background noise — such as traffic, crowd chatter, or mechanical interference — 
degrades the intelligibility and quality of audio recordings. This API removes 
such noise using one of four available methods:

| Model | Type | Description |
|-------|------|-------------|
| `spectral_subtraction` | Classical | Boll (1979). Fast, no GPU required |
| `geometric_subtraction` | Classical |
| `mlp` | ML | Frame-level MLP with IRM masking |
| `unet` | ML | U-Net with skip connections |

### How to Use

1. Upload a noisy `.wav` audio file to `/enhance`
2. Choose a model (`spectral_subtraction`, `geometric_subtraction`, `mlp`, or `unet`)
3. Receive the enhanced audio file as a downloadable `.wav`
4. Optionally use `/evaluate` to also receive quality metrics (PESQ, STOI, etc.)

### Audio Requirements

- **Format**: WAV (16-bit PCM recommended)
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Channels**: Mono or Stereo (automatically converted to mono)
- **Maximum Duration**: 60 seconds
- **Maximum File Size**: 50 MB
    """,
    version="1.0.0",
    contact={
        "name": "Applied ML Group 2 — RUG 2026",
        "url": "https://github.com/MordoTheHacker/Applied-ML-Audio-Denoising-RUG-2026",
    },
    license_info={
        "name": "MIT",
    },
)

# ─────────────────────────────────────────────
# Audio Processing
# ─────────────────────────────────────────────

def validate_and_load_audio(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Validate and load audio from bytes.

    Raises HTTPException for:
        - Unsupported file format
        - File too large
        - Audio too long
        - Corrupt/unreadable audio
    """
    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB. "
                   f"Received: {len(file_bytes) / (1024*1024):.1f} MB"
        )

    # Check file extension
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{suffix}'. "
                   f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Load audio
    try:
        audio_io = io.BytesIO(file_bytes)
        y, file_sr = sf.read(audio_io)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read audio file. Ensure it is a valid audio file. Error: {str(e)}"
        )

    # Convert stereo to mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        logger.info(f"Converted stereo to mono: {filename}")

    # Check duration
    duration = len(y) / file_sr
    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=f"Audio too long. Maximum duration is {MAX_DURATION_SECONDS} seconds. "
                   f"Received: {duration:.1f} seconds"
        )

    if duration < 0.1:
        raise HTTPException(
            status_code=422,
            detail=f"Audio too short. Minimum duration is 0.1 seconds. "
                   f"Received: {duration:.3f} seconds"
        )

    # Resample to 16kHz
    if file_sr != SR:
        y = librosa.resample(y.astype(np.float64), orig_sr=file_sr, target_sr=SR)
        logger.info(f"Resampled {file_sr}Hz → {SR}Hz: {filename}")

    return y.astype(np.float64)

def enhance_audio(y: np.ndarray, model_name: ModelName) -> np.ndarray:
    """Run enhancement using the specified model."""

    if model_name == ModelName.spectral_subtraction:
        model = get_model(model_name)
        return model.enhance(y)
    
    elif model_name == ModelName.geometric_subtraction:
        model = get_model(model_name)
        return model.enhance(y)

    elif model_name == ModelName.mlp:
        model, mean, std = get_model(model_name)
        
        # 1. Create the temp file name, write to it, and close it immediately
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        # 2. Write numpy audio array directly to path
        sf.write(tmp_path, y, SR, format='WAV')
        
        # 3. Process it and clean up
        try:
            enhanced = mlp_enhance(tmp_path, model, mean, std, sr=SR)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()  # Deletes the file cleanly using the Path object
        return enhanced

    elif model_name == ModelName.unet:
        model, mean, std = get_model(model_name)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        sf.write(tmp_path, y, SR, format='WAV')
        
        try:
            enhanced = unet_enhance(tmp_path, model, mean, std, sr=SR)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return enhanced

def audio_to_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# Pydantic Models (Request/Response Schemas)
# ─────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type: 'classical' or 'ml'")
    description: str = Field(..., description="Human-readable description")
    available: bool = Field(..., description="Whether model weights are loaded and ready")
    parameters: Optional[int] = Field(None, description="Number of trainable parameters (ML models only)")
    reference: Optional[str] = Field(None, description="Academic reference for this method")

class EnhanceResponse(BaseModel):
    model_used: str = Field(..., description="Name of the model used for enhancement")
    input_duration_seconds: float = Field(..., description="Duration of the input audio in seconds")
    output_duration_seconds: float = Field(..., description="Duration of the enhanced audio in seconds")
    sample_rate: int = Field(..., description="Sample rate of the output audio (always 16000 Hz)")
    processing_time_seconds: float = Field(..., description="Time taken to process the audio")
    message: str = Field(..., description="Human-readable status message")

class MetricsResponse(BaseModel):
    model_used: str = Field(..., description="Name of the model used for enhancement")
    input_duration_seconds: float = Field(..., description="Duration of the input audio")
    processing_time_seconds: float = Field(..., description="Time taken to process")
    metrics: dict = Field(..., description="""
Quality metrics comparing enhanced audio against provided clean reference:
- **PESQ**: Perceptual Evaluation of Speech Quality (-0.5 to 4.5, higher is better)
- **STOI**: Short-Time Objective Intelligibility (0 to 1, higher is better)
- **CSIG**: Signal distortion MOS predictor (1 to 5)
- **CBAK**: Background noise MOS predictor (1 to 5)
- **COVL**: Overall quality MOS predictor (1 to 5)
- **SSNR**: Segmental Signal-to-Noise Ratio (dB, higher is better)
- **SI_SDR**: Scale-Invariant SDR (dB, higher is better)
    """)
    above_random_baseline: dict = Field(..., description="Comparison against noisy baseline (no denoising)")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing what went wrong")

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get(
    "/",
    summary="API Health Check",
    description="Returns API status and basic information.",
    tags=["Info"],
)
def root():
    """
    Health check endpoint. Returns API name, version, and available endpoints.
    """
    return {
        "name": "Audio Denoising API",
        "version": "1.0.0",
        "status": "running",
        "description": "Speech enhancement using classical and ML-based denoising models",
        "endpoints": {
            "GET  /":                "This endpoint — health check",
            "GET  /models":          "List all available models",
            "GET  /models/{name}":   "Get info for a specific model",
            "POST /enhance":         "Enhance a noisy audio file",
            "POST /evaluate":        "Enhance + compute quality metrics",
            "GET  /docs":            "Interactive API documentation (Swagger UI)",
            "GET  /redoc":           "Alternative API documentation (ReDoc)",
        },
        "supported_audio_formats": list(SUPPORTED_FORMATS),
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "sample_rate": SR,
    }

@app.get(
    "/models",
    summary="List Available Models",
    description="Returns a list of all models with their availability status and descriptions.",
    response_model=list[ModelInfo],
    tags=["Models"],
)
def list_models():
    """
    Returns all available denoising models with their metadata.

    A model is **available** if its weights have been trained and saved to disk.
    The `spectral_subtraction` model is always available as it requires no training.
    """
    models = [
        ModelInfo(
            name="spectral_subtraction",
            type="classical",
            description=(
                "Classical spectral subtraction (Boll, 1979). Estimates noise power "
                "from initial silent frames and subtracts it from each frame. "
                "Fast and requires no GPU, but can produce musical noise artifacts."
            ),
            available=True,
            parameters=None,
            reference="Boll (1979). Suppression of acoustic noise in speech using spectral subtraction. "
                      "IEEE Trans. Acoustics, Speech, Signal Process.",
        ),
        ModelInfo(
            name="geometric_subtraction",
            type="classical",
            description=(
                "Geometric spectral subtraction (Lu & Loizou, 2008). Improves on standard "
                "spectral subtraction by accounting for the phase relationship between speech "
                "and noise using a geometric gain function. Reduces musical noise artifacts."
            ),
            available=True,
            parameters=None,
            reference="Lu & Loizou (2008). A geometric approach to spectral subtraction. "
                    "Speech Communication.",
        ),
        ModelInfo(
            name="mlp",
            type="ml",
            description=(
                "Frame-level Multilayer Perceptron with Ideal Ratio Mask (IRM). "
                "Uses an 11-frame temporal context window (5 past + current + 5 future frames) "
                "and 4 hidden layers of 1024 neurons. Predicts a soft mask per frequency bin."
            ),
            available=check_model_available(ModelName.mlp),
            parameters=6_316_289,
            reference="Wang et al. (2014). Towards scaling up classification-based speech separation. "
                      "IEEE Trans. Audio, Speech, Language Process.",
        ),
        ModelInfo(
            name="unet",
            type="ml",
            description=(
                "U-Net convolutional encoder-decoder with skip connections. "
                "Treats the spectrogram as a 2D image and predicts an IRM mask. "
                "Skip connections preserve fine harmonic detail lost in the bottleneck. "
                "Best quality among the three models."
            ),
            available=check_model_available(ModelName.unet),
            parameters=7_760_961,
            reference="Ronneberger et al. (2015). U-Net: Convolutional Networks for "
                      "Biomedical Image Segmentation. MICCAI.",
        ),
    ]
    return models

@app.get(
    "/models/{model_name}",
    summary="Get Model Info",
    description="Returns detailed information about a specific model.",
    tags=["Models"],
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
def get_model_info(model_name: ModelName):
    """
    Returns detailed metadata for a specific model including:
    - Architecture description
    - Training hyperparameters (if trained)
    - Evaluation metrics on VoiceBank+DEMAND test set (if available)
    - Academic reference
    """
    # Load training log if available
    training_log = {}
    log_path = MODELS_DIR / model_name.value / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            training_log = json.load(f)

    # Load evaluation results if available
    eval_results = {}
    results_path = RESULTS_DIR / f"{model_name.value}.json"
    if results_path.exists():
        with open(results_path) as f:
            eval_results = json.load(f)

    return {
        "name": model_name.value,
        "available": check_model_available(model_name),
        "training_log": training_log if training_log else None,
        "evaluation_results": eval_results if eval_results else None,
    }

@app.post(
    "/enhance",
    summary="Enhance Noisy Audio",
    description="""
Upload a noisy audio file and receive a denoised version.

**Request:**
- `file`: Audio file (WAV, FLAC, MP3, or OGG)
- `model`: Denoising model to use (`spectral_subtraction`, `geometric_subtraction`, `mlp`, or `unet`)

**Response:**
- Enhanced audio file as a downloadable WAV (16kHz, 16-bit PCM)
- Response headers contain processing metadata

**Notes:**
- Audio is automatically converted to mono and resampled to 16kHz
- Maximum duration: 60 seconds
- Maximum file size: 50 MB
    """,
    tags=["Enhancement"],
    responses={
        200: {"description": "Enhanced WAV audio file"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported audio format"},
        422: {"model": ErrorResponse, "description": "Invalid audio file or duration"},
        503: {"model": ErrorResponse, "description": "Model weights not available"},
    },
)
def enhance(
    file: UploadFile = File(
        ...,
        description="Noisy audio file to enhance (WAV, FLAC, MP3, or OGG)"
    ),
    model: ModelName = Query(
        default=ModelName.spectral_subtraction,
        description="Denoising model to use"
    ),
):
    """
    Enhance a noisy audio file using the specified denoising model.

    The response is a downloadable WAV file containing the enhanced audio.
    Processing metadata is included in the response headers.
    """
    t0 = time.time()

    # Read and validate file
    file_bytes = file.file.read()
    y = validate_and_load_audio(file_bytes, file.filename)
    input_duration = len(y) / SR

    logger.info(f"Enhancing {file.filename} ({input_duration:.2f}s) with {model}")

    # Enhance
    try:
        enhanced = enhance_audio(y, model)
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhancement failed: {str(e)}"
        )

    processing_time = time.time() - t0
    output_duration = len(enhanced) / SR

    # Convert to WAV bytes
    wav_bytes = audio_to_bytes(enhanced)

    logger.info(f"Enhancement complete in {processing_time:.2f}s")

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="enhanced_{file.filename}"',
            "X-Model-Used": model.value,
            "X-Input-Duration": str(round(input_duration, 3)),
            "X-Output-Duration": str(round(output_duration, 3)),
            "X-Processing-Time": str(round(processing_time, 3)),
            "X-Sample-Rate": str(SR),
        }
    )

@app.post(
    "/evaluate",
    summary="Enhance and Evaluate Quality",
    description="""
Upload a noisy audio file **and** its clean reference to receive the enhanced audio
along with objective quality metrics.

**Request:**
- `noisy_file`: Noisy audio file to enhance
- `clean_file`: Clean reference audio (ground truth) for metric computation
- `model`: Denoising model to use

**Response (JSON):**
- Quality metrics: PESQ, STOI, CSIG, CBAK, COVL, SSNR, SI-SDR
- Comparison against noisy baseline (unprocessed audio metrics)
- Processing time and metadata

**Use this endpoint when:**
- You have access to the clean reference and want to measure improvement
- You are benchmarking or comparing models
    """,
    response_model=MetricsResponse,
    tags=["Enhancement"],
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported audio format"},
        422: {"model": ErrorResponse, "description": "Invalid audio or duration mismatch"},
        503: {"model": ErrorResponse, "description": "Model weights not available"},
    },
)
def evaluate(
    noisy_file: UploadFile = File(
        ...,
        description="Noisy audio file to enhance"
    ),
    clean_file: UploadFile = File(
        ...,
        description="Clean reference audio file for quality metric computation"
    ),
    model: ModelName = Query(
        default=ModelName.spectral_subtraction,
        description="Denoising model to use"
    ),
):
    """
    Enhance audio and compute objective quality metrics against a clean reference.
    """

    t0 = time.time()

    # Load both files
    noisy_bytes = noisy_file.file.read()
    clean_bytes = clean_file.file.read()

    noisy = validate_and_load_audio(noisy_bytes, noisy_file.filename)
    clean = validate_and_load_audio(clean_bytes, clean_file.filename)

    # Check duration compatibility
    noisy_dur = len(noisy) / SR
    clean_dur = len(clean) / SR
    if abs(noisy_dur - clean_dur) > 1.0:
        raise HTTPException(
            status_code=422,
            detail=f"Noisy and clean audio durations differ by more than 1 second "
                   f"({noisy_dur:.2f}s vs {clean_dur:.2f}s). "
                   "Please provide matching audio files."
        )

    logger.info(f"Evaluating {noisy_file.filename} ({noisy_dur:.2f}s) with {model}")

    # Enhance
    try:
        enhanced = enhance_audio(noisy, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

    # Compute metrics: enhanced vs clean
    n = min(len(enhanced), len(clean))
    metrics = evaluate_all(clean[:n], enhanced[:n], SR)

    # Compute noisy baseline metrics for comparison
    baseline_metrics = evaluate_all(clean[:n], noisy[:n], SR)

    processing_time = time.time() - t0

    # Build delta comparison
    above_baseline = {}
    for key, val in metrics.items():
        base_val = baseline_metrics.get(key, float('nan'))
        delta = val - base_val if not (np.isnan(val) or np.isnan(base_val)) else float('nan')
        above_baseline[key] = {
            "noisy_baseline": round(base_val, 4),
            "enhanced": round(val, 4),
            "delta": round(delta, 4),
            "improved": bool(delta > 0) if not np.isnan(delta) else None,
        }

    return MetricsResponse(
        model_used=model.value,
        input_duration_seconds=round(noisy_dur, 3),
        processing_time_seconds=round(processing_time, 3),
        metrics={k: round(v, 4) for k, v in metrics.items() if not np.isnan(v)},
        above_random_baseline=above_baseline,
    )