import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
U-Net for Speech Enhancement.

Implements a U-Net architecture operating on 2D STFT log-magnitude
spectrograms, predicting an IRM mask to recover clean speech.

Key design choices (following research proposal):
    - Input:   (Batch, 1, 256, 256) — single-channel spectrogram image
    - Output:  (Batch, 1, 256, 256) — IRM mask, Sigmoid → [0, 1]
    - Encoder: 4 conv blocks with stride-2 downsampling
    - Decoder: 4 upconv blocks with skip connections (torch.cat)
    - Bottleneck: heavy dropout (0.5) to prevent noise memorization
    - Loss:    MSE + 0.1 * L1 for sharp harmonic reconstruction
    - Optimizer:   AdamW with weight decay for regularization

Architecture:
    Input (1,256,256) spectrogram
      ↓ Encoder
    E1: Conv → 32  (256,256) ──────────────────────────────┐ skip1
    E2: Conv → 64  (128,128) ─────────────────────────┐    │ skip2
    E3: Conv → 128 (64,64)   ────────────────────┐    │    │ skip3
    E4: Conv → 256 (32,32)   ───────────────┐    │    │    │ skip4
      ↓ Bottleneck
    B:  Conv → 512 (16,16) + Dropout(0.5)   │    │    │    │
      ↓ Decoder
    D4: UpConv → 256 + cat(skip4) → 256  ───┘    │    │    │
    D3: UpConv → 128 + cat(skip3) → 128  ─────────┘    │    │
    D2: UpConv → 64  + cat(skip2) → 64   ──────────────┘    │
    D1: UpConv → 32  + cat(skip1) → 32   ───────────────────┘
      ↓ Output
    Conv 1x1 → 1 channel → Sigmoid → IRM mask

Reference:
    Ronneberger et al. (2015). U-Net: Convolutional Networks for
    Biomedical Image Segmentation. MICCAI.
"""

# ─────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Dataset for U-Net training.

    Loads preprocessed NPZ spectrograms and returns:
        - Input:  noisy log-magnitude spectrogram (1, 256, 256)
        - Target: IRM mask computed from clean + noisy magnitudes (1, 256, 256)

    The 257th frequency bin is dropped to make the spatial dimensions
    powers of 2, which is required for clean U-Net skip connections.

    Args:
        npz_path: Path to train_spectrograms.npz or test_spectrograms.npz
        mean:     Global mean for input standardization (from train set)
        std:      Global std for input standardization (from train set)
    """

    def __init__(
        self,
        npz_path: Path,
        mean: float = None,
        std: float = None,
    ):
        print(f"Loading {npz_path}...")
        data = np.load(npz_path)

        # Shape: (N, time_frames, freq_bins) = (N, 256, 257)
        noisy_log = data['noisy_magnitude'].astype(np.float32)
        clean_log = data['clean_magnitude'].astype(np.float32)
        data.close()

        # Drop the 257th bin to (N, 256, 256) — makes dims powers of 2
        noisy_log = noisy_log[:, :, :256]
        clean_log = clean_log[:, :, :256]

        n_chunks, n_frames, n_bins = noisy_log.shape
        print(f"  Chunks: {n_chunks}, Shape per chunk: ({n_frames}, {n_bins})")

        # Convert log to linear for IRM computation
        clean_mag = np.exp(clean_log)
        noisy_mag = np.exp(noisy_log)

        # Square-root IRM: sqrt(|S|^2 / (|S|^2 + |N|^2))
        clean_power = clean_mag ** 2
        noise_power = np.maximum(noisy_mag ** 2 - clean_power, 1e-10)
        irm = np.sqrt(clean_power / (clean_power + noise_power + 1e-10))
        self.irm = np.clip(irm, 0.0, 1.0).astype(np.float32)

        # Free linear arrays
        del clean_mag, noisy_mag, clean_log

        # Store log-magnitude for input
        self.noisy_log = noisy_log  # (N, 256, 256)

        # Compute normalization stats
        if mean is None or std is None:
            print("  Computing normalization statistics...")
            self.mean = float(np.mean(self.noisy_log))
            self.std  = float(np.std(self.noisy_log)) + 1e-8
        else:
            self.mean = float(mean)
            self.std  = float(std)

        print(f"  Total chunks: {n_chunks:,}")
        print(f"  Input mean: {self.mean:.4f}, std: {self.std:.4f}")

    def __len__(self):
        return len(self.noisy_log)

    def __getitem__(self, idx):
        # Normalize input
        x = (self.noisy_log[idx] - self.mean) / self.std  # (256, 256)

        # Add channel dimension: (1, 256, 256)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.irm[idx], dtype=torch.float32).unsqueeze(0)

        return x, y

# ─────────────────────────────────────────────
# U-Net Blocks
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Used in both encoder and decoder paths.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncoderBlock(nn.Module):
    """
    Encoder block: ConvBlock + MaxPool downsampling.
    Returns skip connection feature map and downsampled output.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)    # saved for skip connection
        down = self.pool(skip) # passed to next encoder block
        return skip, down

class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + cat(skip) + ConvBlock.
    Skip connection doubles channel count before ConvBlock reduces it.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle any size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([skip, x], dim=1)  # Concatenate along channel dimension
        return self.conv(x)

# ─────────────────────────────────────────────
# U-Net Model
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for spectrogram-domain speech enhancement.

    Input:  (B, 1, 256, 256) normalized log-magnitude spectrogram
    Output: (B, 1, 256, 256) IRM mask in [0, 1]

    Args:
        base_filters:       Filters in first encoder block (doubles each level)
        dropout_enc:        Dropout in encoder blocks
        dropout_bottleneck: Dropout in bottleneck (0.5 to prevent memorization)
        dropout_dec:        Dropout in decoder blocks
    """

    def __init__(
        self,
        base_filters: int = 32,
        dropout_enc: float = 0.1,
        dropout_bottleneck: float = 0.5,
        dropout_dec: float = 0.1,
    ):
        super().__init__()

        f = base_filters

        # Encoder: 4 blocks, channels double each time
        self.enc1 = EncoderBlock(1,    f,    dropout_enc)          # (B,32, 128,128)
        self.enc2 = EncoderBlock(f,    f*2,  dropout_enc)          # (B,64, 64, 64)
        self.enc3 = EncoderBlock(f*2,  f*4,  dropout_enc)          # (B,128,32, 32)
        self.enc4 = EncoderBlock(f*4,  f*8,  dropout_enc)          # (B,256,16, 16)

        # Bottleneck
        self.bottleneck = ConvBlock(f*8, f*16, dropout_bottleneck)  # (B,512,16, 16)

        # Decoder: in_channels = bottleneck + skip channels
        self.dec4 = DecoderBlock(f*16 + f*8, f*8,  dropout_dec)   # (B,256,32, 32)
        self.dec3 = DecoderBlock(f*8  + f*4, f*4,  dropout_dec)   # (B,128,64, 64)
        self.dec2 = DecoderBlock(f*4  + f*2, f*2,  dropout_dec)   # (B,64, 128,128)
        self.dec1 = DecoderBlock(f*2  + f,   f,    dropout_dec)   # (B,32, 256,256)

        # Output: 1x1 conv -> single channel IRM mask
        self.output = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.output(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_unet(
    train_npz: Path,
    val_npz: Path,
    output_dir: Path,
    base_filters: int = 32,
    dropout_enc: float = 0.1,
    dropout_bottleneck: float = 0.5,
    dropout_dec: float = 0.1,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 16,
    max_epochs: int = 50,
    patience: int = 5,
) -> UNet:
    """
    Train the U-Net with AdamW, MSE+L1 loss, and early stopping.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Datasets
    print("\nLoading training data...")
    train_ds = SpectrogramDataset(train_npz)

    print("\nLoading validation data...")
    val_ds = SpectrogramDataset(val_npz, mean=train_ds.mean, std=train_ds.std)

    # Save normalization stats
    np.save(output_dir / 'norm_mean.npy', np.array([train_ds.mean]))
    np.save(output_dir / 'norm_std.npy',  np.array([train_ds.std]))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'),
    )

    # Model
    model = UNet(
        base_filters=base_filters,
        dropout_enc=dropout_enc,
        dropout_bottleneck=dropout_bottleneck,
        dropout_dec=dropout_dec,
    ).to(device)

    print(f"\nModel Parameters: {model.count_parameters():,}")
    print(f"Input Shape:  (batch, 1, 256, 256)")
    print(f"Output Shape: (batch, 1, 256, 256)")

    # Loss: MSE + 0.1 * L1
    mse_loss = nn.MSELoss()
    l1_loss  = nn.L1Loss()

    def criterion(pred, target):
        return mse_loss(pred, target) + 0.1 * l1_loss(pred, target)

    # AdamW with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_losses = []
    val_losses   = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...")
    print("="*70)

    for epoch in range(max_epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} Val  ", leave=False):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{max_epochs} | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"Best model saved (val={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: epoch {best_epoch})")
                break

    log = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "hyperparameters": {
            "base_filters": base_filters,
            "dropout_enc": dropout_enc,
            "dropout_bottleneck": dropout_bottleneck,
            "dropout_dec": dropout_dec,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        }
    }
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete. Best epoch: {best_epoch}, Val Loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', weights_only=True))
    return model

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def enhance_file(
    noisy_path: Path,
    model: UNet,
    mean: float,
    std: float,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Enhance a single audio file using the trained U-Net.

    Steps:
        1. Load audio and compute STFT
        2. Drop 257th bin
        3. Tile into 256x256 windows and run through U-Net
        4. Apply predicted IRM mask to noisy magnitude
        5. Reconstruct waveform with phase from noisy signal
    """

    device = next(model.parameters()).device
    model.eval()

    y, file_sr = sf.read(noisy_path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    y = y.astype(np.float64)

    # STFT: (257, T)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    noisy_mag_full = np.abs(D).astype(np.float32)
    noisy_phase    = np.angle(D)
    noisy_log_full = np.log(noisy_mag_full + 1e-9)

    n_freq, n_frames = noisy_log_full.shape

    # Pad time to multiple of 16
    target_frames = ((n_frames - 1) // 16 + 1) * 16
    pad_frames = target_frames - n_frames

    # Crop freq to 256, pad time
    noisy_log_256 = noisy_log_full[:256, :]
    noisy_log_pad = np.pad(
        noisy_log_256, ((0, 0), (0, pad_frames)), mode='constant'
    )

    # Process in 256x256 tiles
    mask_256 = np.zeros((256, target_frames), dtype=np.float32)
    tile_size = 256

    for start in range(0, target_frames, tile_size):
        end = min(start + tile_size, target_frames)
        tile = noisy_log_pad[:, start:end]

        # Pad tile to 256x256
        tile_pad = np.zeros((256, tile_size), dtype=np.float32)
        tile_pad[:, :tile.shape[1]] = tile

        # Normalize and run through U-Net
        tile_norm = (tile_pad - mean) / std
        x = torch.tensor(tile_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()   # (256, 256)

        valid = end - start
        mask_256[:, start:start + valid] = pred[:, :valid]

    # Trim and restore to 257 bins (keep 257th bin unmasked)
    mask_256  = mask_256[:, :n_frames]
    mask_full = np.ones((257, n_frames), dtype=np.float32)
    mask_full[:256, :] = mask_256

    # Apply mask and reconstruct
    enhanced_mag = noisy_mag_full * mask_full
    D_enhanced   = enhanced_mag * np.exp(1j * noisy_phase)
    enhanced     = librosa.istft(D_enhanced, hop_length=hop_length, length=len(y))

    return enhanced.astype(np.float64)

def evaluate_dataset(
    model: UNet,
    mean: float,
    std: float,
    clean_dir: Path,
    noisy_dir: Path,
    output_dir: Path = None,
    sr: int = 16000,
) -> dict:
    """Evaluate U-Net on full test set and return averaged metrics."""

    clean_files = sorted(Path(clean_dir).glob("*.wav"))
    noisy_files = sorted(Path(noisy_dir).glob("*.wav"))

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating U-Net on {len(clean_files)} test files...")

    all_results = {k: [] for k in ['PESQ', 'STOI', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'SI_SDR']}

    for cf, nf in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
        clean, file_sr = sf.read(cf)
        clean = clean.astype(np.float64)
        if len(clean.shape) > 1:
            clean = np.mean(clean, axis=1)

        enhanced = enhance_file(nf, model, mean, std, sr=sr)

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
    OUTPUT_DIR  = Path("outputs/models/unet")
    RESULTS_DIR = Path("outputs/results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    model = train_unet(
        train_npz=TRAIN_NPZ,
        val_npz=TEST_NPZ,
        output_dir=OUTPUT_DIR,
        base_filters=32,
        dropout_enc=0.1,
        dropout_bottleneck=0.5,
        dropout_dec=0.1,
        lr=1e-4,
        weight_decay=1e-2,
        batch_size=16,
        max_epochs=50,
        patience=5,
    )

    # Load normalization stats
    mean = float(np.load(OUTPUT_DIR / 'norm_mean.npy'))
    std  = float(np.load(OUTPUT_DIR / 'norm_std.npy'))

    # Evaluate
    avg = evaluate_dataset(
        model=model,
        mean=mean,
        std=std,
        clean_dir=CLEAN_DIR,
        noisy_dir=NOISY_DIR,
        output_dir=None,  # Set to Path("outputs/audio/unet") to save audio
    )

    print_results(avg, model_name="U-Net (IRM)")

    # Save results
    output = {
        "model": "U-Net (IRM masking)",
        "hyperparameters": {
            "base_filters": 32,
            "dropout_enc": 0.1,
            "dropout_bottleneck": 0.5,
            "dropout_dec": 0.1,
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "batch_size": 16,
        },
        "n_files": 824,
        "metrics": avg,
    }
    results_path = RESULTS_DIR / "unet.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Full comparison table
    models_to_compare = {
        "Baseline": "noisy_baseline.json",
        "Spec Sub":  "spectral_subtraction.json",
        "MLP":       "mlp.json",
        "U-Net":     "unet.json",
    }

    print("\nFull Comparison Table:")
    print(f"  {'Metric':<10}", end="")
    for name in models_to_compare:
        print(f" {name:>10}", end="")
    print()
    print(f"  {'-'*52}")

    metrics = ['PESQ', 'STOI', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'SI_SDR']
    scores = {}
    for name, fname in models_to_compare.items():
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                scores[name] = json.load(f)["metrics"]
        else:
            scores[name] = {}

    for metric in metrics:
        print(f"  {metric:<10}", end="")
        for name in models_to_compare:
            val = scores.get(name, {}).get(metric, float('nan'))
            print(f" {val:>10.4f}", end="")
        print()

if __name__ == "__main__":
    main()