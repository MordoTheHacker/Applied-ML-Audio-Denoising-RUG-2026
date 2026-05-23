import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
import sys
import time
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate import evaluate_all, print_results

"""
MLP Baseline for Speech Enhancement.

Implements a frame-level MLP that predicts an Ideal Ratio Mask (IRM)
applied to the noisy STFT magnitude to recover clean speech.

Key design choices (following research recommendations):
    - Input:  Concatenated temporal context window (11 frames x 257 bins)
    - Output: IRM mask (257 bins, sigmoid activation → [0, 1])
    - Depth:  4 hidden layers, 1024 neurons each
    - Norm:   BatchNorm after every hidden layer
    - Loss:   MSE between predicted IRM and ideal IRM
    - Optim:  Adam with learning rate 1e-3 or 1e-4

IRM Definition:
    IRM(t, f) = sqrt( |S(t,f)|^2 / (|S(t,f)|^2 + |N(t,f)|^2) )

    Value of 1.0 = keep this frequency bin (speech dominant)
    Value of 0.0 = remove this bin (noise dominant)

Reference:
    Wang et al. (2014). Towards scaling up classification-based
    speech separation. IEEE Trans. Audio, Speech, Language Process.
"""

# ─────────────────────────────────────────────
# IRM Dataset
# ─────────────────────────────────────────────

class IRMDataset(Dataset):
    """
    Cluster-optimized dataset that leverages true memory mapping.
    Calculates log magnitudes and IRM targets dynamically inside __getitem__
    to maintain a rock-solid, low-RAM profile under 1GB.
    """
    def __init__(
        self,
        npz_path: Path,
        context_frames: int = 5,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        self.context = context_frames
        self.window = 2 * context_frames + 1

        print(f"Loading {npz_path} into RAM...")
        data = np.load(npz_path)
        self.noisy_log_mag = data['noisy_magnitude'].astype(np.float32)  # (N, T, F)
        clean_log          = data['clean_magnitude'].astype(np.float32)
        data.close()

        n_chunks, n_frames, n_bins = self.noisy_log_mag.shape
        self.n_bins = n_bins
        print(f"  Loaded: {n_chunks} chunks, {n_frames} frames, {n_bins} bins")

        print("  Computing IRM targets...")
        self.irm = np.zeros((n_chunks, n_frames, n_bins), dtype=np.float32)
        chunk_size = 500

        for start in range(0, n_chunks, chunk_size):
            end = min(start + chunk_size, n_chunks)
            c = np.exp(clean_log[start:end])
            n = np.exp(self.noisy_log_mag[start:end])
            cp = c ** 2
            np_ = np.maximum(n ** 2 - cp, 1e-10)
            self.irm[start:end] = np.clip(
                np.sqrt(cp / (cp + np_ + 1e-10)), 0.0, 1.0
            )
            del c, n, cp, np_

        del clean_log
        import gc
        gc.collect()

        # Build indices
        all_indices = [
            (ci, fi)
            for ci in range(n_chunks)
            for fi in range(context_frames, n_frames - context_frames)
        ]

        # Subsample to 1M
        max_frames = 1_000_000
        if len(all_indices) > max_frames:
            print(f"  Subsampling {len(all_indices):,} → {max_frames:,} frames")
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(all_indices), max_frames, replace=False)
            self.indices = [all_indices[i] for i in chosen]
        else:
            self.indices = all_indices

        print(f"  Final frames: {len(self.indices):,}")

        # Normalization stats
        if mean is None or std is None:
            print("  Computing normalization statistics...")
            sample_size = min(10000, len(self.indices))
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(self.indices), sample_size, replace=False)
            samples = np.array([
                self.noisy_log_mag[
                    self.indices[i][0],
                    self.indices[i][1] - context_frames : self.indices[i][1] + context_frames + 1,
                    :
                ].flatten()
                for i in sample_idx
            ], dtype=np.float32)
            self.mean = samples.mean(axis=0)
            self.std  = samples.std(axis=0) + 1e-8
        else:
            self.mean = mean.astype(np.float32)
            self.std  = std.astype(np.float32)

        print(f"  Dataset ready. RAM usage: ~{(self.noisy_log_mag.nbytes + self.irm.nbytes) / 1e9:.1f} GB")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        chunk_idx, frame_idx = self.indices[idx]

        # All reads from RAM now — fast
        window = self.noisy_log_mag[
            chunk_idx,
            frame_idx - self.context : frame_idx + self.context + 1,
            :
        ].flatten()

        normalized = (window - self.mean) / self.std
        irm_target = self.irm[chunk_idx, frame_idx, :]

        return torch.from_numpy(normalized.copy()), torch.from_numpy(irm_target.copy())

# ─────────────────────────────────────────────
# MLP Model
# ─────────────────────────────────────────────

class SpeechMLP(nn.Module):
    """
    4-layer MLP for IRM-based speech enhancement.

    Architecture:
        Input → [Linear → BatchNorm → LeakyReLU] × 4 → Linear → Sigmoid

    Args:
        input_dim:   window_size × freq_bins (e.g., 11 × 257 = 2827)
        output_dim:  freq_bins (257)
        hidden_dim:  neurons per hidden layer (1024)
        n_layers:    number of hidden layers (4)
        dropout:     dropout probability (0.2)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 257,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []

        # Input → first hidden layer
        layers += [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
        ]

        # Hidden layers
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(dropout),
            ]

        # Output layer: sigmoid → [0, 1] for IRM
        layers += [
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*layers)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_mlp(
    train_ds: Dataset,
    val_ds: Dataset,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    output_dir: Path,
    n_bins: int,
    context_frames: int = 5,
    hidden_dim: int = 1024,
    n_layers: int = 4,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 2048,
    max_epochs: int = 50,
    patience: int = 5,
) -> SpeechMLP:
    """
    Train the MLP with early stopping.

    Args:
        train_npz:      Path to train_spectrograms.npz
        val_npz:        Path to val_spectrograms.npz
        output_dir:     Where to save checkpoints and logs
        context_frames: Temporal context on each side (window = 2*context+1)
        hidden_dim:     Neurons per hidden layer
        n_layers:       Number of hidden layers
        dropout:        Dropout probability
        lr:             Learning rate
        batch_size:     Training batch size
        max_epochs:     Maximum training epochs
        patience:       Early stopping patience (epochs without improvement)

    Returns:
        Trained SpeechMLP model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: Running on CPU — check CUDA installation")
        sys.exit(1)  # Fail fast rather than waste 2 hours on CPU

    # Save normalization stats for inference
    np.save(output_dir / 'norm_mean.npy', train_mean)
    np.save(output_dir / 'norm_std.npy', train_std)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
        drop_last=True, # Protects BatchNorm1d from single-sample remainder batches
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )

    # Model
    window_size = 2 * context_frames + 1
    freq_bins = n_bins
    input_dim = window_size * freq_bins

    model = SpeechMLP(
        input_dim=input_dim,
        output_dim=freq_bins,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    print(f"\nModel Parameters: {model.count_parameters():,}")
    print(f"Input Dim:  {input_dim} ({window_size} frames × {freq_bins} bins)")
    print(f"Output Dim: {freq_bins} bins (IRM mask)")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )   
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...")
    print("="*65)

    for epoch in range(max_epochs):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train", mininterval=10.0, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} Val  ", mininterval=10.0, leave=False):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{max_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(best was epoch {best_epoch})")
                break

    # Save training log
    log = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "hyperparameters": {
            "context_frames": context_frames,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
        }
    }
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete. Best epoch: {best_epoch}, Val Loss: {best_val_loss:.6f}")

    # Load best weights
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', weights_only=True))
    return model

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def enhance_file(
    noisy_path: Path,
    model: SpeechMLP,
    mean: np.ndarray,
    std: np.ndarray,
    context_frames: int = 5,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Enhance a single audio file using the trained MLP.

    Steps:
        1. Load and normalize audio
        2. Compute STFT
        3. For each frame: predict IRM mask
        4. Apply mask to noisy magnitude
        5. Reconstruct waveform with IFFT + overlap-add
    """

    device = next(model.parameters()).device
    model.eval()

    # Load audio
    y, file_sr = sf.read(noisy_path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    y = y.astype(np.float64)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    noisy_mag_linear = np.abs(D).astype(np.float32)        # for mask application
    noisy_log_mag = np.log(noisy_mag_linear + 1e-9)         # for MLP input
    noisy_phase = np.angle(D)                        # (freq_bins, time_frames)

    freq_bins, n_frames = noisy_mag_linear.shape
    window_size = 2 * context_frames + 1

    # Pad the LOG magnitude once for inference
    pad = np.zeros((freq_bins, context_frames), dtype=np.float32)
    noisy_log_padded = np.concatenate([pad, noisy_log_mag, pad], axis=1)

    # Predict IRM frame by frame
    masks = np.zeros_like(noisy_mag_linear)

    batch_size = 512
    frames_batch = []
    frame_indices = []

    for t in range(n_frames):
        # Slice from the pre-padded log matrix
        window = noisy_log_padded[:, t:t + window_size].T.flatten()
        window_norm = (window - mean) / std
        frames_batch.append(window_norm)
        frame_indices.append(t)

        if len(frames_batch) == batch_size or t == n_frames - 1:
            x = torch.tensor(np.array(frames_batch), dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_mask = model(x).cpu().numpy()
            for i, fi in enumerate(frame_indices):
                masks[:, fi] = pred_mask[i]
            frames_batch = []
            frame_indices = []

    # Apply mask to noisy magnitude
    enhanced_mag = noisy_mag_linear * masks

    # Reconstruct waveform
    D_enhanced = enhanced_mag * np.exp(1j * noisy_phase)
    enhanced = librosa.istft(D_enhanced, hop_length=hop_length, length=len(y))

    return enhanced.astype(np.float64)


def evaluate_dataset(
    model: SpeechMLP,
    mean: np.ndarray,
    std: np.ndarray,
    clean_dir: Path,
    noisy_dir: Path,
    output_dir: Path = None,
    context_frames: int = 5,
    sr: int = 16000,
) -> dict:
    """Evaluate MLP on full test set and return averaged metrics."""

    clean_files = sorted(Path(clean_dir).glob("*.wav"))
    noisy_files = sorted(Path(noisy_dir).glob("*.wav"))

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating MLP on {len(clean_files)} test files...")

    all_results = {k: [] for k in ['PESQ', 'STOI', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'SI_SDR']}

    for cf, nf in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
        clean, file_sr = sf.read(cf)
        if len(clean.shape) > 1:
            clean = np.mean(clean, axis=1)
        if file_sr != sr:
            clean = librosa.resample(clean, orig_sr=file_sr, target_sr=sr)
            file_sr = sr
        clean = clean.astype(np.float64)

        enhanced = enhance_file(nf, model, mean, std, context_frames=context_frames, sr=sr)

        if output_dir:
            sf.write(Path(output_dir) / nf.name, enhanced, sr)

        results = evaluate_all(clean, enhanced, file_sr)
        for k, v in results.items():
            if not np.isnan(v):
                all_results[k].append(v)

    avg = {k: float(np.mean(v)) for k, v in all_results.items() if v}
    return avg

def main():
    TRAIN_NPZ   = Path("data/processed/train_spectrograms.npz")
    TEST_NPZ    = Path("data/processed/test_spectrograms.npz")
    CLEAN_DIR   = Path("data/raw/wavs/test/clean")
    NOISY_DIR   = Path("data/raw/wavs/test/noisy")
    OUTPUT_DIR  = Path("outputs/models/mlp")
    RESULTS_DIR = Path("outputs/results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the complete training dataset profile
    print("\nLoading master dataset...")
    full_train_ds = IRMDataset(TRAIN_NPZ, context_frames=5)

    # 2. Split into clean 80/20 train and validation sets to fix the test leak
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    print(f"Splitting dataset: {train_size:,} train samples, {val_size:,} validation samples")
    
    # Secure reproducibility across runs using a generator seed
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [train_size, val_size], generator=generator
    )

    # Note: Because random_split returns subset wrappers, we pull stats from the base object
    train_mean = full_train_ds.mean
    train_std  = full_train_ds.std

    # Train
    model = train_mlp(
        train_ds=train_ds,
        val_ds=val_ds,
        train_mean=train_mean,
        train_std=train_std,
        output_dir=OUTPUT_DIR,
        n_bins=full_train_ds.n_bins,
        context_frames=5,
        hidden_dim=1024,
        n_layers=4,
        dropout=0.2,
        lr=1e-3,
        batch_size=2048,
        max_epochs=50,
        patience=5,
    )

    # Load normalization stats
    mean = np.load(OUTPUT_DIR / 'norm_mean.npy')
    std  = np.load(OUTPUT_DIR / 'norm_std.npy')

    # Evaluate
    avg= evaluate_dataset(
        model=model,
        mean=mean,
        std=std,
        clean_dir=CLEAN_DIR,
        noisy_dir=NOISY_DIR,
        context_frames=5,
    )

    print_results(avg, model_name="MLP (IRM)")

    # Save results
    output = {
        "model": "MLP (IRM masking)",
        "hyperparameters": {
            "context_frames": 5,
            "hidden_dim": 1024,
            "n_layers": 4,
            "dropout": 0.2,
            "lr": 1e-3,
            "batch_size": 512,
        },
        "n_files": 824,
        "metrics": avg,
    }
    results_path = RESULTS_DIR / "mlp.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Compare against baselines
    baseline_path = RESULTS_DIR / "noisy_baseline.json"
    ss_path = RESULTS_DIR / "spectral_subtraction.json"

    if baseline_path.exists() and ss_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)["metrics"]
        with open(ss_path) as f:
            ss = json.load(f)["metrics"]

        print("\nComparison Table:")
        print(f"  {'Metric':<10} {'Baseline':>10} {'SS':>10} {'MLP':>10}")
        print(f"  {'-'*42}")
        for metric, score in avg.items():
            b = baseline.get(metric, float('nan'))
            s = ss.get(metric, float('nan'))
            print(f"  {metric:<10} {b:>10.4f} {s:>10.4f} {score:>10.4f}")


if __name__ == "__main__":
    main()