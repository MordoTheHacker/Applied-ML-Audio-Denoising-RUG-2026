import numpy as np
from typing import Dict
from pesq import pesq
from pystoi import stoi
import librosa
import soundfile as sf
from pathlib import Path
import json

"""
Evaluation metrics for audio denoising.

Implements all standard VoiceBank+DEMAND evaluation metrics:
- PESQ  : Perceptual Evaluation of Speech Quality     (-0.5 to 4.5)
- STOI  : Short-Time Objective Intelligibility         (0 to 1)
- CSIG  : Composite Signal Distortion MOS predictor   (1 to 5)
- CBAK  : Composite Background Noise MOS predictor    (1 to 5)
- COVL  : Composite Overall Quality MOS predictor     (1 to 5)
- SSNR  : Segmental Signal-to-Noise Ratio             (dB)
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio  (dB)
"""

# ─────────────────────────────────────────────
# 1. PESQ
# ─────────────────────────────────────────────

def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """
    Perceptual Evaluation of Speech Quality.
    Range: -0.5 (worst) to 4.5 (best).
    Uses 'wb' (wideband) mode for 16kHz audio.
    """
    try:
        return float(pesq(sr, clean, enhanced, 'wb'))
    except Exception as e:
        print(f"  [PESQ] Warning: {e}")
        return float('nan')

# ─────────────────────────────────────────────
# 2. STOI
# ─────────────────────────────────────────────

def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """
    Short-Time Objective Intelligibility.
    Range: 0 (unintelligible) to 1 (fully intelligible).
    """
    try:
        return float(stoi(clean, enhanced, sr, extended=False))
    except Exception as e:
        print(f"  [STOI] Warning: {e}")
        return float('nan')

# ─────────────────────────────────────────────
# 3. Composite Metrics (CSIG, CBAK, COVL)
#    Loizou (2007) composite MOS predictors
#    Implemented from scratch
# ─────────────────────────────────────────────

def compute_composite(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Composite MOS predictors from Loizou (2007).
    
    Returns:
        dict with keys: CSIG, CBAK, COVL
        Ranges: 1–5 for all three.
    
    Reference:
        Hu & Loizou (2008). Evaluation of Objective Quality
        Measures for Speech Enhancement. IEEE Trans. Audio.
    """

    def _lpc_order(sr: int) -> int:
        return 2 + sr // 1000

    def _lpcoeff(speech: np.ndarray, order: int):
        """Compute LPC coefficients via autocorrelation."""
        n = len(speech)
        r = np.array([np.dot(speech[:n - k], speech[k:]) for k in range(order + 1)])
        if r[0] == 0:
            return np.zeros(order), 0.0

        # Levinson-Durbin
        a = np.zeros(order)
        e = r[0]
        for i in range(order):
            if e == 0:
                break
            lam = -r[i + 1] - np.dot(a[:i], r[i:0:-1])
            lam /= e
            a[:i + 1] = a[:i + 1] + lam * a[i::-1]
            a[i] = lam
            e *= (1 - lam ** 2)
        return -a, e

    def _wss(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
        """
        Weighted Spectral Slope distance — vectorized using librosa.util.frame.
        """
        winlen = int(sr * 0.025)
        skiprate = int(sr * 0.010)
        num_crit = 25

        n = min(len(clean), len(enhanced))
        if n < winlen:
            return 0.0

        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(winlen) / winlen))
        lpc_order = _lpc_order(sr)

        # Vectorize framing — shape (winlen, num_frames)
        c_frames = librosa.util.frame(clean[:n], frame_length=winlen, hop_length=skiprate) * w[:, None]
        e_frames = librosa.util.frame(enhanced[:n], frame_length=winlen, hop_length=skiprate) * w[:, None]

        num_frames = c_frames.shape[1]
        clean_spec = np.zeros((num_frames, num_crit))
        enhanced_spec = np.zeros((num_frames, num_crit))

        for i in range(num_frames):
            c_frame = c_frames[:, i]
            e_frame = e_frames[:, i]

            # Skip silence frames
            if np.dot(c_frame, c_frame) < 1e-7:
                continue

            _, c_err = _lpcoeff(c_frame, lpc_order)
            _, e_err = _lpcoeff(e_frame, lpc_order)

            c_err = max(c_err, 1e-10)
            e_err = max(e_err, 1e-10)

            clean_spec[i, :] = np.log(c_err)
            enhanced_spec[i, :] = np.log(e_err)

        # Slope differences (vectorized across all frames)
        clean_slope = np.diff(clean_spec, axis=1)
        enhanced_slope = np.diff(enhanced_spec, axis=1)

        weights = np.maximum(clean_slope ** 2, 1e-10)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-10

        wss_dist = np.sum(weights * (clean_slope - enhanced_slope) ** 2, axis=1)
        return float(np.mean(wss_dist))

    def _llr(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
        """
        Log Likelihood Ratio — vectorized using librosa.util.frame.
        Skips silence frames (energy < 1e-7) to avoid division instability.
        """
        winlen = int(sr * 0.025)
        skiprate = int(sr * 0.010)
        lpc_order = _lpc_order(sr)

        n = min(len(clean), len(enhanced))
        if n < winlen:
            return 0.0

        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(winlen) / winlen))

        # Vectorize: shape (winlen, num_frames)
        c_frames = librosa.util.frame(clean[:n], frame_length=winlen, hop_length=skiprate) * w[:, None]
        e_frames = librosa.util.frame(enhanced[:n], frame_length=winlen, hop_length=skiprate) * w[:, None]

        num_frames = c_frames.shape[1]
        llr_vals = []

        for i in range(num_frames):
            c_frame = c_frames[:, i]
            e_frame = e_frames[:, i]

            # Skip silence frames to avoid LLR instability
            frame_energy = np.dot(c_frame, c_frame)
            if frame_energy < 1e-7:
                continue

            c_lpc, c_err = _lpcoeff(c_frame, lpc_order)
            c_err = max(c_err, 1e-10)

            # Vectorized autocorrelation using np.correlate
            r_c = np.array([np.dot(c_frame[:winlen - k], c_frame[k:]) for k in range(lpc_order + 1)])
            r_e = np.array([np.dot(e_frame[:winlen - k], e_frame[k:]) for k in range(lpc_order + 1)])

            num = max(np.dot(c_lpc, r_e[:lpc_order]), 1e-10)
            den = max(np.dot(c_lpc, r_c[:lpc_order]), 1e-10)

            llr_vals.append(np.log(num / den + 1e-10))

        if not llr_vals:
            return 0.0

        # Trim top 5% outliers
        llr_arr = np.sort(np.array(llr_vals))
        llr_arr = llr_arr[:int(0.95 * len(llr_arr))]
        return float(np.mean(np.maximum(llr_arr, 0)))

    # Compute WSS and LLR
    wss_dist = _wss(clean, enhanced, sr)
    llr_mean = _llr(clean, enhanced, sr)

    # PESQ (needed for composite scores — use NaN if unavailable)
    try:
        pesq_score = compute_pesq(clean, enhanced, sr)
    except Exception:
        print("PESQ compute ERROR in compute_composite")
        pesq_score = float('nan')

    # Composite formulae (Hu & Loizou 2008, Table III)
    if not np.isnan(pesq_score):
        csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_score - 0.009 * wss_dist
        cbak = 1.634 + 0.478 * pesq_score - 0.007 * wss_dist + 0.063 * llr_mean
        covl = 1.594 + 0.805 * pesq_score - 0.512 * llr_mean - 0.007 * wss_dist
    else:
        # Fallback without PESQ
        csig = 3.093 - 1.029 * llr_mean - 0.009 * wss_dist
        cbak = 1.634 - 0.007 * wss_dist + 0.063 * llr_mean
        covl = 1.594 - 0.512 * llr_mean - 0.007 * wss_dist

    # Clip to valid range [1, 5]
    csig = float(np.clip(csig, 1, 5))
    cbak = float(np.clip(cbak, 1, 5))
    covl = float(np.clip(covl, 1, 5))

    return {"CSIG": csig, "CBAK": cbak, "COVL": covl}

# ─────────────────────────────────────────────
# 4. SSNR — Segmental SNR
# ─────────────────────────────────────────────

def compute_ssnr(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000,
                 frame_len: float = 0.025, frame_shift: float = 0.010) -> float:
    """
    Segmental Signal-to-Noise Ratio (dB).
    More accurate than global SNR for speech signals.
    
    Clips each frame SNR to [-10, 35] dB to avoid outlier frames
    (silence, very loud bursts) distorting the mean.
    """
    winlen = int(sr * frame_len)
    skiprate = int(sr * frame_shift)
    n = min(len(clean), len(enhanced))

    num_frames = (n - winlen) // skiprate
    if num_frames <= 0:
        return float('nan')

    snr_frames = []
    for i in range(num_frames):
        start = i * skiprate
        c = clean[start:start + winlen]
        e = enhanced[start:start + winlen]

        signal_energy = np.dot(c, c)
        noise_energy = np.dot(c - e, c - e)

        if signal_energy < 1e-10:
            continue  # Skip silence frames

        snr = 10 * np.log10(signal_energy / (noise_energy + 1e-10))
        snr_frames.append(np.clip(snr, -10, 35))

    return float(np.mean(snr_frames)) if snr_frames else float('nan')

# ─────────────────────────────────────────────
# 5. SI-SDR — Scale-Invariant SDR
# ─────────────────────────────────────────────

def compute_si_sdr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (dB).
    
    Robust to volume/amplitude differences introduced by preprocessing
    (e.g., RMS normalization). Higher is better. Typical range: -∞ to ~30 dB.
    
    Reference: Le Roux et al. (2019). SDR – half-baked or well done?
    """
    # Zero-mean
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)

    # Scale factor: project enhanced onto clean
    alpha = np.dot(enhanced, clean) / (np.dot(clean, clean) + 1e-10)
    target = alpha * clean
    noise = enhanced - target

    si_sdr = 10 * np.log10(
        (np.dot(target, target) + 1e-10) /
        (np.dot(noise, noise) + 1e-10)
    )
    return float(si_sdr)

# ─────────────────────────────────────────────
# Master evaluation function
# ─────────────────────────────────────────────

def evaluate_all(clean: np.ndarray, enhanced: np.ndarray,
                 sr: int = 16000) -> Dict[str, float]:
    """
    Run all evaluation metrics on a clean/enhanced audio pair.

    Args:
        clean:    Ground-truth clean waveform (1D numpy array, float32/64)
        enhanced: Model output waveform (1D numpy array, same length as clean)
        sr:       Sample rate in Hz (default 16000)

    Returns:
        Dictionary with all metric scores.
    """
    # Ensure same length
    n = min(len(clean), len(enhanced))
    clean = clean[:n].astype(np.float64)
    enhanced = enhanced[:n].astype(np.float64)

    results = {}
    results['PESQ'] = compute_pesq(clean, enhanced, sr)
    results['STOI'] = compute_stoi(clean, enhanced, sr)
    composite = compute_composite(clean, enhanced, sr)
    results.update(composite)
    results['SSNR'] = compute_ssnr(clean, enhanced, sr)
    results['SI_SDR'] = compute_si_sdr(clean, enhanced)

    return results

def print_results(results: Dict[str, float], model_name: str = "Model") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<12} {'Score':>10}   {'Range':<15} {'Notes'}")
    print(f"  {'-'*55}")
    
    metrics_info = {
        'PESQ':   ('-0.5 → 4.5', 'Higher is better, ~2.0 for SS'),
        'STOI':   ('0 → 1',      'Higher is better, >0.8 is good'),
        'CSIG':   ('1 → 5',      'Signal distortion MOS'),
        'CBAK':   ('1 → 5',      'Background noise MOS'),
        'COVL':   ('1 → 5',      'Overall quality MOS'),
        'SSNR':   ('dB',         'Higher is better'),
        'SI_SDR': ('dB',         'Higher is better'),
    }

    for metric, score in results.items():
        range_str, note = metrics_info.get(metric, ('', ''))
        score_str = f"{score:.4f}" if not np.isnan(score) else "N/A"
        print(f"  {metric:<12} {score_str:>10}   {range_str:<15} {note}")

    print(f"{'='*50}\n")

# ─────────────────────────────────────────────
# Noisy to Clean Absolute Baseline (no denoising no generation)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    clean_dir = Path("data/raw/wavs/test/clean")
    noisy_dir = Path("data/raw/wavs/test/noisy")

    clean_files = sorted(clean_dir.glob("*.wav"))
    noisy_files = sorted(noisy_dir.glob("*.wav"))

    all_results = {
        'PESQ': [], 'STOI': [], 'CSIG': [],
        'CBAK': [], 'COVL': [], 'SSNR': [], 'SI_SDR': []
    }

    for i, (cf, nf) in enumerate(zip(clean_files, noisy_files)):
        clean, sr = sf.read(cf)
        noisy, _  = sf.read(nf)
        clean    = clean.astype(np.float64)
        noisy    = noisy.astype(np.float64)

        print(f"[{i+1}/{len(clean_files)}] {cf.name}", end='\r')
        results = evaluate_all(clean, noisy, sr)

        for k, v in results.items():
            if not np.isnan(v):
                all_results[k].append(v)

    # Average across all files
    print("\n")
    avg = {k: float(np.mean(v)) for k, v in all_results.items()}
    print_results(avg, model_name="Noisy Baseline — Average over 824 test files")

    # Save results to file
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "model": "Noisy Baseline (no denoising)",
        "description": "Average metrics over all 824 test files. Use as floor for comparison.",
        "n_files": len(clean_files),
        "metrics": avg
    }

    output_path = results_dir / "noisy_baseline.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved baseline to {output_path}")