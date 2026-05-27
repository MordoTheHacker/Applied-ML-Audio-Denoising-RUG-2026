import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api import app

client = TestClient(app)

SR = 16000

# ── Helpers ────────────────────────────────────────────────────────────────

def make_wav(duration=2.0, sr=SR, noise=True) -> bytes:
    """Generate a synthetic noisy WAV in memory."""
    n = int(duration * sr)
    if noise:
        audio = np.random.randn(n).astype(np.float32) * 0.1
    else:
        t = np.linspace(0, duration, n)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def make_wav_stereo(duration=2.0) -> bytes:
    """Generate a stereo WAV."""
    n = int(duration * SR)
    audio = np.random.randn(n, 2).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    buf.seek(0)
    return buf.read()

# ── Health check ───────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "running"
    assert "endpoints" in data

# ── Models listing ─────────────────────────────────────────────────────────

def test_list_models():
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()
    names = [m["name"] for m in models]
    assert "spectral_subtraction" in names
    assert "geometric_subtraction" in names
    assert "mlp" in names
    assert "unet" in names

def test_get_model_info_valid():
    r = client.get("/models/spectral_subtraction")
    assert r.status_code == 200
    assert "name" in r.json()

def test_get_model_info_invalid():
    r = client.get("/models/nonexistent_model")
    assert r.status_code == 422  # FastAPI rejects invalid enum

# ── /enhance — happy paths ─────────────────────────────────────────────────

def test_enhance_spectral_subtraction():
    wav = make_wav()
    r = client.post(
        "/enhance?model=spectral_subtraction",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert "X-Model-Used" in r.headers
    assert r.headers["X-Model-Used"] == "spectral_subtraction"
    # Check returned bytes are valid audio
    audio, sr = sf.read(io.BytesIO(r.content))
    assert sr == SR
    assert len(audio) > 0

def test_enhance_geometric_subtraction():
    wav = make_wav()
    r = client.post(
        "/enhance?model=geometric_subtraction",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200

def test_enhance_stereo_converted_to_mono():
    """Stereo input should be accepted and converted silently."""
    wav = make_wav_stereo()
    r = client.post(
        "/enhance?model=spectral_subtraction",
        files={"file": ("stereo.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200

def test_enhance_default_model():
    """No model param → should use spectral_subtraction default."""
    wav = make_wav()
    r = client.post(
        "/enhance",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200

# ── /enhance — error handling ──────────────────────────────────────────────

def test_enhance_unsupported_format():
    """Non-audio file should return 415."""
    r = client.post(
        "/enhance?model=spectral_subtraction",
        files={"file": ("test.txt", b"this is not audio", "text/plain")},
    )
    assert r.status_code == 415

def test_enhance_corrupt_audio():
    """Random bytes with .wav extension should return 422."""
    r = client.post(
        "/enhance?model=spectral_subtraction",
        files={"file": ("corrupt.wav", b"not a wav file at all", "audio/wav")},
    )
    assert r.status_code == 422

def test_enhance_file_too_large():
    """File over 50MB should return 413."""
    big = b"0" * (51 * 1024 * 1024)
    r = client.post(
        "/enhance?model=spectral_subtraction",
        files={"file": ("big.wav", big, "audio/wav")},
    )
    assert r.status_code == 413

def test_enhance_too_short():
    """Sub 0.1s audio should return 422."""
    n = int(0.05 * SR)
    audio = np.zeros(n, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    buf.seek(0)
    r = client.post(
        "/enhance?model=spectral_subtraction",
        files={"file": ("short.wav", buf.read(), "audio/wav")},
    )
    assert r.status_code == 422

def test_enhance_invalid_model_name():
    """Unknown model name should return 422 from FastAPI enum validation."""
    wav = make_wav()
    r = client.post(
        "/enhance?model=banana",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 422

# ── /evaluate ─────────────────────────────────────────────────────────────

def test_evaluate_returns_metrics():
    noisy = make_wav(noise=True)
    clean = make_wav(noise=False)
    r = client.post(
        "/evaluate?model=spectral_subtraction",
        files={
            "noisy_file": ("noisy.wav", noisy, "audio/wav"),
            "clean_file": ("clean.wav", clean, "audio/wav"),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data
    assert "above_random_baseline" in data
    assert "PESQ" in data["metrics"] or "STOI" in data["metrics"]

def test_evaluate_duration_mismatch():
    """Clean and noisy with >1s duration difference should return 422."""
    noisy = make_wav(duration=2.0)
    clean = make_wav(duration=5.0, noise=False)
    r = client.post(
        "/evaluate?model=spectral_subtraction",
        files={
            "noisy_file": ("noisy.wav", noisy, "audio/wav"),
            "clean_file": ("clean.wav", clean, "audio/wav"),
        },
    )
    assert r.status_code == 422