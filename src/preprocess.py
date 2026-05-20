import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
import gc

"""
Audio preprocessing pipeline for VoiceBank+DEMAND dataset.
Raw waveforms are segmented into fixed-length chunks and transformed into STFT spectrograms.
The magnitude component is extracted and log-scaled to compress the dynamic range, 
while phase information is stored separately for later signal reconstruction.
"""

class AudioPreprocessor:
    """Configurable audio preprocessing pipeline with chunking support."""
    
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        fixed_shape: tuple = (256, 257),
        chunk_duration: float = 2.0,
        chunk_overlap: float = 0.0,
    ):
        """
        Initialize preprocessor with STFT parameters and chunking.
        
        Args:
            sr: Sample rate (Hz)
            n_fft: FFT size (window length in samples)
            hop_length: Number of samples between successive frames
            window: Window function (e.g., 'hann', 'hamming')
            fixed_shape: (n_frames, n_freq_bins) to pad/truncate to
            chunk_duration: Duration of each chunk in seconds (e.g., 1.0 for 1-second chunks)
            chunk_overlap: Overlap ratio between 0 and 1 (e.g., 0.5 for 50% overlap)
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.fixed_shape = fixed_shape
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        
        # Derived parameters
        self.n_fft_bins = n_fft // 2 + 1  # Number of frequency bins (513 for n_fft=1024, 257 for 512)
        self.chunk_samples = int(sr * chunk_duration)  # Number of samples per chunk
        self.chunk_stride = int(self.chunk_samples * (1 - chunk_overlap))  # Stride for sliding window
        
        # Validate
        if fixed_shape[1] != self.n_fft_bins:
            raise ValueError(
                f"fixed_shape[1]={fixed_shape[1]} but n_fft={n_fft} gives {self.n_fft_bins} bins. "
                f"Either change n_fft or fixed_shape[1]."
            )
        
        if not (0 <= chunk_overlap < 1):
            raise ValueError(f"chunk_overlap must be in [0, 1), got {chunk_overlap}")
    
    def remove_dc_offset(self, y: np.ndarray) -> np.ndarray:
        """Remove DC component (center signal around 0)."""
        return y - np.mean(y)
    
    def rms_normalize(self, y: np.ndarray) -> np.ndarray:
        """RMS normalization: scale so that RMS = 1."""
        rms = np.sqrt(np.mean(y ** 2))
        if rms == 0:
            return y
        return y / rms
    
    def chunk_waveform(self, y: np.ndarray) -> list:
        """
        Split waveform into fixed-duration overlapping chunks.
        
        Args:
            y: Input waveform
        
        Returns:
            List of chunks (each of length chunk_samples, padded if necessary)
        """
        chunks = []
        n_samples = len(y)
        
        # Generate chunk boundaries
        start = 0
        while start < n_samples:
            end = min(start + self.chunk_samples, n_samples)
            chunk = y[start:end]
            
            # Pad the last chunk if it's shorter than chunk_samples
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')
            
            chunks.append(chunk)
            start += self.chunk_stride
        
        return chunks
    
    def apply_stft(self, y: np.ndarray) -> tuple:
        """
        Apply STFT and extract magnitude and phase.
        
        Returns:
            mag: Log-scaled magnitude spectrogram (freq_bins, time_frames)
            phase: Phase in radians (freq_bins, time_frames)
        """
        # STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)
        
        # Magnitude and phase
        mag = np.abs(D)
        phase = np.angle(D)
        
        # Log scaling: add small epsilon to avoid log(0)
        log_mag = np.log(mag + 1e-9)
        
        return log_mag, phase
    
    def pad_or_truncate(self, spec: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Pad or truncate spectrogram to fixed shape.
        
        Args:
            spec: Spectrogram (n_freq_bins, n_frames)
            target_shape: (n_frames_target, n_freq_bins_target)
        
        Returns:
            Reshaped spectrogram
        """
        n_frames_target, n_freq_target = target_shape
        n_freq, n_frames = spec.shape
        
        # Frequency dimension should match already
        assert n_freq == n_freq_target, f"Frequency bins mismatch: {n_freq} vs {n_freq_target}"
        
        if n_frames > n_frames_target:
            # Truncate: take first n_frames_target
            return spec[:, :n_frames_target]
        elif n_frames < n_frames_target:
            # Pad with zeros (silence/noise floor)
            pad_width = n_frames_target - n_frames
            return np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            return spec
    
    def preprocess_single(self, audio_path: Path) -> list:
        """
        Preprocess a single audio file into multiple chunks.
        
        Returns:
            List of tuples: [(log_mag, phase), ...]
            where each has shape fixed_shape
        """
        # Load
        y, loaded_sr = sf.read(audio_path)
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Resample if needed
        if loaded_sr != self.sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=self.sr)
        
        # Clean: remove DC offset
        y = self.remove_dc_offset(y)
        
        # Normalize: RMS normalization
        y = self.rms_normalize(y)
        
        # Chunk the waveform
        chunks = self.chunk_waveform(y)
        
        # Process each chunk through STFT
        processed_chunks = []
        for chunk in chunks:
            # STFT
            log_mag, phase = self.apply_stft(chunk)
            
            # Shape alignment
            log_mag = self.pad_or_truncate(log_mag, self.fixed_shape)
            phase = self.pad_or_truncate(phase, self.fixed_shape)
            
            processed_chunks.append((log_mag, phase))
        
        return processed_chunks
    
    def preprocess_dataset(self, input_dir: Path, output_dir: Path, split: str = "train"):
        """
        Preprocess all audio files and save in batches.
        Processes files sequentially, saving batches to disk to minimize memory usage.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        clean_dir = input_dir / "clean"
        noisy_dir = input_dir / "noisy"
        
        clean_files = sorted(clean_dir.glob("*.wav"))
        noisy_files = sorted(noisy_dir.glob("*.wav"))
        
        print(f"Processing {split} set: {len(clean_files)} file pairs...")
        
        # Batch size: process this many chunks before saving
        batch_size = 5000
        batch_num = 0
        
        # Keep running lists for current batch
        batch_clean_mag = []
        batch_noisy_mag = []
        batch_clean_phase = []
        batch_noisy_phase = []
        
        total_chunks = 0
        
        for clean_path, noisy_path in tqdm(zip(clean_files, noisy_files), total=len(clean_files), desc="Processing"):
            clean_chunks = self.preprocess_single(clean_path)
            noisy_chunks = self.preprocess_single(noisy_path)
            
            for clean_chunk, noisy_chunk in zip(clean_chunks, noisy_chunks):
                clean_log_mag, clean_phase = clean_chunk
                noisy_log_mag, noisy_phase = noisy_chunk
                
                # Add to batch
                batch_clean_mag.append(clean_log_mag.T)
                batch_noisy_mag.append(noisy_log_mag.T)
                batch_clean_phase.append(clean_phase.T)
                batch_noisy_phase.append(noisy_phase.T)
                
                total_chunks += 1
                
                # Save batch when it reaches batch_size
                if total_chunks % batch_size == 0:
                    self._save_batch(
                        output_dir, split, batch_num,
                        batch_clean_mag,
                        batch_noisy_mag,
                        batch_clean_phase,
                        batch_noisy_phase
                    )
                    batch_num += 1
                    # Clear batch
                    batch_clean_mag = []
                    batch_noisy_mag = []
                    batch_clean_phase = []
                    batch_noisy_phase = []
        
        # Save final partial batch
        if batch_clean_mag:
            self._save_batch(
                output_dir, split, batch_num,
                batch_clean_mag,
                batch_noisy_mag,
                batch_clean_phase,
                batch_noisy_phase
            )
            batch_num += 1
        
        # Merge all batches into final file
        self._merge_batches(output_dir, split, batch_num)
        
        # Save config
        config = {
            "sr": self.sr,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "window": self.window,
            "fixed_shape": list(self.fixed_shape),  # Convert tuple to list for JSON
            "chunk_duration": self.chunk_duration,
            "chunk_overlap": self.chunk_overlap,
        }

        # Load existing config to preserve other splits' chunk counts
        config_path = output_dir / "preprocessing_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                existing = json.load(f)
            # Keep train/test chunks from previous runs
            if "train_chunks" in existing:
                config["train_chunks"] = existing["train_chunks"]
            if "test_chunks" in existing:
                config["test_chunks"] = existing["test_chunks"]

        # Load the saved npz file and count actual chunks
        npz_path = output_dir / f"{split}_spectrograms.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            n_chunks = len(data['clean_magnitude'])
            config[f"{split}_chunks"] = int(n_chunks)
            data.close()

        # Recalculate total
        config["total_chunks"] = config.get("train_chunks", 0) + config.get("test_chunks", 0)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Preprocessing {split} complete! Total chunks: {total_chunks}")
    
    def _save_batch(self, output_dir, split, batch_num, clean_mag, noisy_mag, clean_phase, noisy_phase):
        """Save a batch of chunks to a temporary npz file."""
        batch_path = output_dir / f"{split}_batch_{batch_num:04d}.npz"
        
        np.savez(
            batch_path,
            clean_magnitude=np.array(clean_mag, dtype=np.float32),
            noisy_magnitude=np.array(noisy_mag, dtype=np.float32),
            clean_phase=np.array(clean_phase, dtype=np.float32),
            noisy_phase=np.array(noisy_phase, dtype=np.float32),
        )
        print(f"Saved batch {batch_num} ({len(clean_mag)} chunks)")
    
    def _merge_batches(self, output_dir, split, num_batches):
        """Merge batch files incrementally using disk stream buffers to save memory."""
        print(f"Merging {num_batches} batches into final file container...")
        
        if num_batches == 0:
            print("No batches to merge.")
            return
        
        first_path = output_dir / f"{split}_batch_0000.npz"
        with np.load(first_path) as data:
            merged_clean_mag   = data['clean_magnitude'].copy()
            merged_noisy_mag   = data['noisy_magnitude'].copy()
            merged_clean_phase = data['clean_phase'].copy()
            merged_noisy_phase = data['noisy_phase'].copy()
        first_path.unlink()
        
        for batch_num in range(1, num_batches):
            batch_path = output_dir / f"{split}_batch_{batch_num:04d}.npz"
            if not batch_path.exists():
                continue

            with np.load(batch_path) as data:
                new_clean_mag   = data['clean_magnitude'].copy()
                new_noisy_mag   = data['noisy_magnitude'].copy()
                new_clean_phase = data['clean_phase'].copy()
                new_noisy_phase = data['noisy_phase'].copy()

            # Concatenate one array at a time, delete old immediately
            old = merged_clean_mag
            merged_clean_mag = np.concatenate([old, new_clean_mag], axis=0)
            del old, new_clean_mag

            old = merged_noisy_mag
            merged_noisy_mag = np.concatenate([old, new_noisy_mag], axis=0)
            del old, new_noisy_mag

            old = merged_clean_phase
            merged_clean_phase = np.concatenate([old, new_clean_phase], axis=0)
            del old, new_clean_phase

            old = merged_noisy_phase
            merged_noisy_phase = np.concatenate([old, new_noisy_phase], axis=0)
            del old, new_noisy_phase

            batch_path.unlink()
            gc.collect()
            print(f"  Merged batch {batch_num} → running total: {len(merged_clean_mag)} chunks")

        print("Saving final compiled archive file...")
        np.savez_compressed(
            output_dir / f"{split}_spectrograms.npz",
            clean_magnitude=merged_clean_mag,
            noisy_magnitude=merged_noisy_mag,
            clean_phase=merged_clean_phase,
            noisy_phase=merged_noisy_phase,
        )
        
        # Free memory allocations cleanly
        del merged_clean_mag, merged_noisy_mag, merged_clean_phase, merged_noisy_phase
        gc.collect()
        print(f"Merged file saved successfully: {split}_spectrograms.npz")
 
def main():
    """Run preprocessing pipeline on full dataset."""
    
    raw_dir = Path("data/raw/wavs")
    processed_dir = Path("data/processed")
    
    preprocessor = AudioPreprocessor(
        sr=16000,
        n_fft=512,
        hop_length=128,
        window="hann",
        fixed_shape=(256, 257),
        chunk_duration=2.0,
        chunk_overlap=0.0,       # No overlap to reduce chunks
    )
 
    # Process train and test, but skip if already exists
    for split in ["train", "test"]:
        split_dir = raw_dir / split
        output_file = processed_dir / f"{split}_spectrograms.npz"
        
        # Check if already processed
        if output_file.exists():
            print(f"✓ {split}_spectrograms.npz already exists, skipping {split} split.")
            continue
        
        if split_dir.exists():
            print(f"Processing {split} split...")
            preprocessor.preprocess_dataset(split_dir, processed_dir, split=split)
        else:
            print(f"Warning: {split_dir} not found, skipping {split} split.")
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()