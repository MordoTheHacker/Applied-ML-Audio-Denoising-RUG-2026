import io
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio

RAW_DIR = Path("data/raw")
WAV_DIR = RAW_DIR / "wavs"

def download_dataset():
    """Download VoiceBank+DEMAND from HuggingFace."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(
        "JacobLinCool/VoiceBank-DEMAND-16k",
        cache_dir=str(RAW_DIR)
    )
    # Disable automatic decoding to avoid torchcodec dependency
    dataset = dataset.cast_column("clean", Audio(decode=False))
    dataset = dataset.cast_column("noisy", Audio(decode=False))
    print("Download complete.")
    return dataset

def extract_wavs(dataset):
    """Extract dataset into clean folder structure of .wav files."""

    if WAV_DIR.exists() and any(WAV_DIR.rglob("*.wav")):
        print("WAV files already extracted, skipping.")
        return

    for split in ["train", "test"]:
        for kind in ["clean", "noisy"]:
            (WAV_DIR / split / kind).mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        print(f"Extracting {split} set...")
        for i, sample in enumerate(dataset[split]):
            for kind in ["clean", "noisy"]:
                audio_bytes = sample[kind]["bytes"]
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                sf.write(WAV_DIR / split / kind / f"{i:05d}.wav", audio_array, sr)

            if i % 100 == 0:
                print(f"  {split}: {i} files done")

    print(f"Extraction complete. WAV files at: {WAV_DIR}")

def prepare_dataset():
    """Main entry point: download and extract dataset."""
    dataset = download_dataset()
    extract_wavs(dataset)
    print("Dataset ready.")

if __name__ == "__main__":
    prepare_dataset()