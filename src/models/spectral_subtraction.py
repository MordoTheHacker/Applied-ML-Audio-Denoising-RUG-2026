import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import librosa
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # Add src to path
from evaluate import evaluate_all, print_results

"""
Spectral Subtraction for Speech Enhancement.

Implements the classical spectral subtraction algorithm as described in:
    Boll (1979). Suppression of acoustic noise in speech using spectral subtraction.
    IEEE Transactions on Acoustics, Speech, and Signal Processing.

Pipeline (following proposal equations 1-11):
    1. Frame the noisy signal with overlapping windows
    2. DFT each frame, store phase
    3. Estimate noise power from initial silent frames
    4. Update noise estimate recursively
    5. Subtract noise power, apply spectral floor
    6. Square root back to magnitude
    7. Recombine with original phase
    8. IFFT back to time domain
    9. Overlap-add reconstruction
"""

class SpectralSubtraction:
    """
    Classical Spectral Subtraction speech enhancer.

    Args:
        sr:             Sample rate (Hz)
        frame_len:      Frame length in seconds (default 25ms)
        frame_shift:    Frame shift in seconds (default 10ms)
        n_fft:          FFT size (default: next power of 2 above frame_len * sr)
        noise_frames:   Number of initial frames to use for noise estimation
        lambda_n:       Noise estimate smoothing factor (0.7 - 0.9)
        alpha:          Over-subtraction factor (1.0 = standard, >1.0 = aggressive)
        beta:           Spectral floor factor (prevents negative values)
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_len: float = 0.025,
        frame_shift: float = 0.010,
        n_fft: Optional[int] = None,
        noise_frames: int = 20,
        lambda_n: float = 0.95,
        alpha: float = 1.5,
        beta: float = 0.002,
    ):
        self.sr = sr
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.noise_frames = noise_frames
        self.lambda_n = lambda_n
        self.alpha = alpha
        self.beta = beta

        # Derived parameters
        self.win_len = int(sr * frame_len)
        self.hop_len = int(sr * frame_shift)

        # FFT size: next power of 2 above win_len for efficiency
        if n_fft is None:
            self.n_fft = int(2 ** np.ceil(np.log2(self.win_len)))
        else:
            self.n_fft = n_fft

        # Hann window
        self.window = np.hanning(self.win_len)

    def _frame_signal(self, y: np.ndarray) -> np.ndarray:
        """
        Split signal into overlapping frames. (Proposal eq. 3)

        Returns:
            frames: (num_frames, win_len)
        """
        n_samples = len(y)
        num_frames = 1 + (n_samples - self.win_len) // self.hop_len

        frames = np.zeros((num_frames, self.win_len))
        for i in range(num_frames):
            start = i * self.hop_len
            frames[i] = y[start:start + self.win_len] * self.window

        return frames

    def _estimate_noise(self, frames: np.ndarray) -> np.ndarray:
        """
        Estimate initial noise power from first N silent frames. (Proposal eq. 5)

        Returns:
            noise_power: (n_fft // 2 + 1,) — noise power spectrum
        """
        n = min(self.noise_frames, len(frames))
        spectra = np.abs(np.fft.rfft(frames[:n], n=self.n_fft)) ** 2
        return np.mean(spectra, axis=0)

    def _update_noise(self, noise_power: np.ndarray,
                      frame_power: np.ndarray) -> np.ndarray:
        """
        Recursive noise power update. (Proposal eq. 6)

        λ_n controls the smoothing:
        - High λ_n (0.98): slow adaptation, stable but misses fast noise changes
        - Low  λ_n (0.70): fast adaptation, tracks noise better but may distort speech
        """
        return self.lambda_n * noise_power + (1 - self.lambda_n) * frame_power

    def enhance(self, y: np.ndarray) -> np.ndarray:
        """
        Enhance a single noisy waveform using spectral subtraction.

        Args:
            y: Noisy waveform (1D numpy array, float64)

        Returns:
            enhanced: Denoised waveform (same length as input)
        """
        y = y.astype(np.float64)
        n_samples = len(y)

        # Step 1: Frame the signal
        frames = self._frame_signal(y)
        num_frames = len(frames)

        # Step 2: DFT each frame, store magnitude and phase (Proposal eq. 4)
        spectra = np.fft.rfft(frames, n=self.n_fft)       # (num_frames, n_fft//2+1)
        magnitudes = np.abs(spectra)                        # |Y_k(ω)|
        phases = np.angle(spectra)                          # ∠Y_k(ω)
        power = magnitudes ** 2                             # |Y_k(ω)|²

        # Step 3: Initialize noise estimate from first N frames (Proposal eq. 5)
        noise_power = self._estimate_noise(frames)

        # Steps 4-8: Process each frame
        enhanced_spectra = np.zeros_like(spectra)

        for k in range(num_frames):
            # Step 4: Update noise estimate recursively (Proposal eq. 6)
            # Only update during non-speech frames (simple VAD: low energy frames)
            frame_energy = np.sum(power[k])
            noise_energy = np.sum(noise_power)

            # Simple VAD: if frame energy < 2x noise energy, treat as noise
            if frame_energy < 2.0 * noise_energy:
                noise_power = self._update_noise(noise_power, power[k])

            # Step 5: Subtract noise power, apply spectral floor (Proposal eq. 7)
            # |X̂_k(ω)|² = max(|Y_k(ω)|² - α·P̂_noise(ω), β·P̂_noise(ω))
            clean_power = np.maximum(
                power[k] - self.alpha * noise_power,
                self.beta * noise_power
            )

            # Step 6: Square root back to magnitude (Proposal eq. 8)
            clean_magnitude = np.sqrt(clean_power)

            # Step 7: Recombine with original phase (Proposal eq. 9)
            enhanced_spectra[k] = clean_magnitude * np.exp(1j * phases[k])

        # Step 8: IFFT each frame back to time domain (Proposal eq. 10)
        enhanced_frames = np.fft.irfft(enhanced_spectra, n=self.n_fft)
        enhanced_frames = enhanced_frames[:, :self.win_len]

        # Step 9: Overlap-add reconstruction (Proposal eq. 11)
        enhanced = np.zeros(n_samples)
        window_sum = np.zeros(n_samples)

        for k in range(num_frames):
            start = k * self.hop_len
            end = min(start + self.win_len, n_samples)
            length = end - start

            enhanced[start:end] += enhanced_frames[k, :length] * self.window[:length]
            window_sum[start:end] += self.window[:length] ** 2

        # Normalize by window overlap to avoid amplitude artifacts
        window_sum = np.maximum(window_sum, 1e-8)
        enhanced /= window_sum

        return enhanced

    def enhance_file(self, input_path: Path, output_path: Optional[Path] = None) -> np.ndarray:
        """
        Enhance a single audio file.

        Args:
            input_path:  Path to noisy .wav file
            output_path: If provided, save enhanced audio here

        Returns:
            enhanced: Denoised waveform
        """
        y, sr = sf.read(input_path)

        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Resample if needed
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        enhanced = self.enhance(y)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, enhanced, self.sr)

        return enhanced

    def evaluate_dataset(
        self,
        clean_dir: Path,
        noisy_dir: Path,
        output_dir: Optional[Path] = None,
        max_files: Optional[int] = None,
    ) -> dict:
        """
        Run spectral subtraction on the full test set and evaluate.

        Args:
            clean_dir:   Directory of clean reference .wav files
            noisy_dir:   Directory of noisy input .wav files
            output_dir:  If provided, save enhanced audio here
            max_files:   If set, only evaluate first N files

        Returns:
            dict of averaged evaluation metrics
        """

        clean_files = sorted(Path(clean_dir).glob("*.wav"))
        noisy_files = sorted(Path(noisy_dir).glob("*.wav"))

        if max_files:
            clean_files = clean_files[:max_files]
            noisy_files = noisy_files[:max_files]

        print(f"Evaluating Spectral Subtraction on {len(clean_files)} files...")

        all_results = {
            'PESQ': [], 'STOI': [], 'CSIG': [],
            'CBAK': [], 'COVL': [], 'SSNR': [], 'SI_SDR': []
        }

        for i, (cf, nf) in enumerate(tqdm(zip(clean_files, noisy_files),
                                           total=len(clean_files),
                                           desc="Evaluating")):
            # Load
            clean, sr = sf.read(cf)
            noisy, _  = sf.read(nf)

            clean = clean.astype(np.float64)
            noisy = noisy.astype(np.float64)

            # Ensure mono
            if len(clean.shape) > 1:
                clean = np.mean(clean, axis=1)
            if len(noisy.shape) > 1:
                noisy = np.mean(noisy, axis=1)

            # Enhance
            enhanced = self.enhance(noisy)

            # Save if output_dir provided
            if output_dir is not None:
                out_path = Path(output_dir) / nf.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_path, enhanced, self.sr)

            # Evaluate
            results = evaluate_all(clean, enhanced, sr)
            for k, v in results.items():
                if not np.isnan(v):
                    all_results[k].append(v)

        # Average across all files
        avg = {k: float(np.mean(v)) for k, v in all_results.items() if v}
        print_results(avg, model_name="Spectral Subtraction")

        return avg

def main():
    """
    Run spectral subtraction on the full test set, evaluate, and save results.
    """

    # Paths
    clean_dir  = Path("data/raw/wavs/test/clean")
    noisy_dir  = Path("data/raw/wavs/test/noisy")
    output_dir = Path("outputs/audio/spectral_subtraction")
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Add src to path so evaluate.py is importable
    sys.path.insert(0, str(Path(__file__).parent))

    # Initialize model with proposal parameters
    model = SpectralSubtraction(
        sr=16000,
        frame_len=0.025,      # 25ms frames
        frame_shift=0.010,    # 10ms hop
        noise_frames=20,      # Use first 20 frames for noise estimation
        lambda_n=0.95,        # Slow noise adaptation (stable) Gridsearch justification
        alpha=1.5,            # Gridsearch justification
        beta=0.002,           # Spectral floor Gridsearch justification
    )

    # Evaluate on test set
    avg = model.evaluate_dataset(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        output_dir=output_dir,
    )

    # Save results
    output = {
        "model": "Spectral Subtraction",
        "hyperparameters": {
            "frame_len": model.frame_len,
            "frame_shift": model.frame_shift,
            "noise_frames": model.noise_frames,
            "lambda_n": model.lambda_n,
            "alpha": model.alpha,
            "beta": model.beta,
        },
        "n_files": 824,
        "metrics": avg,
    }

    results_path = results_dir / "spectral_subtraction.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Compare against noisy baseline
    baseline_path = results_dir / "noisy_baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)["metrics"]

        print("\nImprovement over noisy baseline:")
        print(f"  {'Metric':<10} {'Baseline':>10} {'SS':>10} {'Delta':>10}")
        print(f"  {'-'*42}")
        for metric, score in avg.items():
            base = baseline.get(metric, float('nan'))
            delta = score - base
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {metric:<10} {base:>10.4f} {score:>10.4f} {arrow} {abs(delta):.4f}")


if __name__ == "__main__":
    main()