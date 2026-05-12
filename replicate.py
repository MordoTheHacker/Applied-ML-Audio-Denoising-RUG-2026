import subprocess
import sys
import os
import argparse
from pathlib import Path

"""
Full Replication Script for Audio Denoising MLP Training.

Run this script to:
    1. Install all requirements
    2. Download VoiceBank+DEMAND dataset
    3. Preprocess the dataset
    4. Train the MLP and save best weights
    5. Evaluate and save results
    6. Push results to GitHub

Usage:
    python replicate.py --github_token YOUR_TOKEN

Requirements:
    - Python 3.9+
    - CUDA-capable GPU
    - ~30GB free disk space
    - Git configured
"""

def run(cmd: str, desc: str = "", check: bool = True):
    print(f"\n{'='*65}")
    if desc:
        print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result

def section(title: str):
    print(f"\n{'#'*65}")
    print(f"#  {title}")
    print(f"{'#'*65}\n")

# ─────────────────────────────────────────────
# Step 1 — Install Requirements
# ─────────────────────────────────────────────

def install_requirements():
    section("STEP 1: Installing Requirements")

    # Detect CUDA version
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: nvidia-smi not found. Falling back to CPU PyTorch.")
        cuda_version = None
    else:
        # Parse CUDA version from nvidia-smi output
        import re
        match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
        cuda_version = match.group(1) if match else None
        print(f"Detected CUDA version: {cuda_version}")

    # Install PyTorch with correct CUDA version
    if cuda_version:
        major = int(cuda_version.split(".")[0])
        minor = int(cuda_version.split(".")[1])

        if major == 12 and minor >= 4:
            torch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        elif major == 12 and minor >= 1:
            torch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        elif major == 11 and minor >= 8:
            torch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        else:
            print(f"WARNING: CUDA {cuda_version} not explicitly supported. Using cu118.")
            torch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_cmd = "pip install torch torchvision"

    run(torch_cmd, "Installing PyTorch with CUDA support")

    # Install all other requirements
    run("pip install -r requirements.txt", "Installing project requirements")

    # Verify GPU is available
    verify = subprocess.run(
        'python -c "import torch; print(f\'GPU available: {torch.cuda.is_available()}\'); print(f\'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'CPU\'}\')"',
        shell=True, capture_output=True, text=True
    )
    print(verify.stdout)
    if "GPU available: False" in verify.stdout:
        print("WARNING: GPU not detected. Training will be very slow on CPU.")

# ─────────────────────────────────────────────
# Step 2 — Download Dataset
# ─────────────────────────────────────────────

def download_dataset():
    section("STEP 2: Downloading VoiceBank+DEMAND Dataset")

    wav_dir = Path("data/raw/wavs")
    if wav_dir.exists() and any(wav_dir.rglob("*.wav")):
        print("Dataset already downloaded. Skipping.")
        return

    run("python src/data.py", "Downloading and extracting VoiceBank+DEMAND")

    # Verify download
    train_files = list(Path("data/raw/wavs/train/clean").glob("*.wav"))
    test_files  = list(Path("data/raw/wavs/test/clean").glob("*.wav"))
    print(f"\nVerification:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Test files:  {len(test_files)}")

    if len(train_files) < 1000:
        raise RuntimeError("Dataset download seems incomplete. Check data/raw/wavs/")

# ─────────────────────────────────────────────
# Step 3 — Preprocess Dataset
# ─────────────────────────────────────────────

def preprocess_dataset():
    section("STEP 3: Preprocessing Dataset")

    train_npz = Path("data/processed/train_spectrograms.npz")
    test_npz  = Path("data/processed/test_spectrograms.npz")

    if train_npz.exists() and test_npz.exists():
        print("Preprocessed data already exists. Skipping.")
        import numpy as np
        d = np.load(train_npz)
        print(f"  Train chunks: {len(d['clean_magnitude']):,}")
        d.close()
        d = np.load(test_npz)
        print(f"  Test chunks:  {len(d['clean_magnitude']):,}")
        d.close()
        return

    run("python src/preprocess.py", "Running preprocessing pipeline")

    # Verify
    if not train_npz.exists():
        raise RuntimeError("Preprocessing failed — train_spectrograms.npz not found.")
    if not test_npz.exists():
        raise RuntimeError("Preprocessing failed — test_spectrograms.npz not found.")

    import numpy as np
    d = np.load(train_npz)
    print(f"\nVerification:")
    print(f"  Train chunks: {len(d['clean_magnitude']):,}")
    print(f"  Shape: {d['clean_magnitude'].shape}")
    d.close()

# ─────────────────────────────────────────────
# Step 4 — Train MLP
# ─────────────────────────────────────────────

def train_mlp():
    section("STEP 4: Training MLP")

    best_model = Path("outputs/models/mlp/best_model.pt")
    if best_model.exists():
        print("Trained model already exists at outputs/models/mlp/best_model.pt")
        print("Delete it to retrain. Skipping.")
        return

    run("python src/models/mlp.py", "Training MLP (IRM masking)")

    # Verify
    if not best_model.exists():
        raise RuntimeError("Training failed — best_model.pt not found.")

    import json
    log_path = Path("outputs/models/mlp/training_log.json")
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        print(f"\nTraining Summary:")
        print(f"  Best epoch:     {log['best_epoch']}")
        print(f"  Best val loss:  {log['best_val_loss']:.6f}")
        print(f"  Total epochs:   {len(log['train_losses'])}")

# ─────────────────────────────────────────────
# Step 5 — Push Results to GitHub
# ─────────────────────────────────────────────

def push_results(github_token: str, github_username: str, repo_name: str):
    section("STEP 5: Pushing Results to GitHub")

    # Configure git remote with token
    remote_url = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
    run(f'git remote set-url origin "{remote_url}"', "Configuring git remote")

    # Stage only results and model weights (not data)
    files_to_push = [
        "outputs/models/mlp/best_model.pt",
        "outputs/models/mlp/norm_mean.npy",
        "outputs/models/mlp/norm_std.npy",
        "outputs/models/mlp/training_log.json",
        "outputs/results/mlp.json",
    ]

    for f in files_to_push:
        if Path(f).exists():
            run(f'git add "{f}"', f"Staging {f}", check=False)
        else:
            print(f"  WARNING: {f} not found, skipping.")

    # Commit
    run(
        'git commit -m "add MLP training results and best model weights"',
        "Committing results",
        check=False  # Don't fail if nothing to commit
    )

    # Push
    run("git push origin main", "Pushing to GitHub")

    print("\nResults pushed successfully!")
    print(f"View at: https://github.com/{github_username}/{repo_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Full replication script for Audio Denoising MLP training."
    )
    parser.add_argument(
        "--github_token",
        type=str,
        required=True,
        help="GitHub Personal Access Token for pushing results"
    )
    parser.add_argument(
        "--github_username",
        type=str,
        default="MordoTheHacker",
        help="GitHub username (default: MordoTheHacker)"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="Applied-ML-Audio-Denoising-RUG-2026",
        help="GitHub repository name"
    )
    parser.add_argument(
        "--skip_push",
        action="store_true",
        help="Skip pushing results to GitHub"
    )
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  AUDIO DENOISING MLP — FULL REPLICATION PIPELINE")
    print("="*65)
    print(f"  GitHub user: {args.github_username}")
    print(f"  Repo:        {args.repo_name}")
    print(f"  Skip push:   {args.skip_push}")
    print("="*65)

    # Verify that in repo root
    if not Path("src").exists() or not Path("requirements.txt").exists():
        raise RuntimeError(
            "Run this script from the repo root directory.\n"
            "Expected: cd Applied-ML-Audio-Denoising-RUG-2026 && python replicate.py"
        )

    try:
        install_requirements()
        download_dataset()
        preprocess_dataset()
        train_mlp()

        if not args.skip_push:
            push_results(
                github_token=args.github_token,
                github_username=args.github_username,
                repo_name=args.repo_name,
            )
        else:
            print("\nSkipping GitHub push (--skip_push flag set).")

        print("\n" + "="*65)
        print("  REPLICATION COMPLETE")
        print("="*65)
        print("  Results saved to:")
        print("    outputs/models/mlp/best_model.pt")
        print("    outputs/models/mlp/training_log.json")
        print("    outputs/results/mlp.json")
        print("="*65)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()