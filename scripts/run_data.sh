#!/bin/bash
#SBATCH --job-name=audio_data
#SBATCH --output=logs/data_%j.log
#SBATCH --error=logs/data_%j.err
#SBATCH --partition=regular
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ─────────────────────────────────────────────
# run_data.sh
# ─────────────────────────────────────────────

echo "============================================="
echo "  Audio Denoising — Data Pipeline"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  Start:  $(date)"
echo "============================================="

# ── Environment Setup ──────────────────────────

echo ""
echo "Setting up Python environment..."

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch/s4697103/AppliedML/Applied-ML-Audio-Denoising-RUG-2026 || exit 1
echo "Working directory: $(pwd)"

mkdir -p logs

source env/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ── Step 1: Download Dataset ───────────────────

echo ""
echo "============================================="
echo "  STEP 1: Downloading Dataset"
echo "============================================="

TRAIN_DIR="data/raw/wavs/train/clean"
TEST_DIR="data/raw/wavs/test/clean"

if [ -d "$TRAIN_DIR" ] && [ "$(ls -A $TRAIN_DIR 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "Dataset already downloaded. Skipping."
    echo "  Train files: $(ls $TRAIN_DIR | wc -l)"
    echo "  Test files:  $(ls $TEST_DIR | wc -l)"
else
    echo "Downloading VoiceBank+DEMAND from HuggingFace..."
    python src/data.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset download failed."
        exit 1
    fi

    echo "Download complete."
    echo "  Train files: $(ls $TRAIN_DIR | wc -l)"
    echo "  Test files:  $(ls $TEST_DIR | wc -l)"
fi

# ── Step 2: Preprocess ────────────────────────

echo ""
echo "============================================="
echo "  STEP 2: Preprocessing Dataset"
echo "============================================="

TRAIN_NPZ="data/processed/train_spectrograms.npz"
TEST_NPZ="data/processed/test_spectrograms.npz"

if [ -f "$TRAIN_NPZ" ] && [ -f "$TEST_NPZ" ]; then
    echo "Preprocessed data already exists. Skipping."
    python -c "
import numpy as np
d = np.load('$TRAIN_NPZ')
print(f'  Train chunks: {len(d[\"clean_magnitude\"]):,}')
print(f'  Shape: {d[\"clean_magnitude\"].shape}')
d.close()
d = np.load('$TEST_NPZ')
print(f'  Test chunks:  {len(d[\"clean_magnitude\"]):,}')
d.close()
"
else
    echo "Running preprocessing pipeline..."
    python src/preprocess.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed."
        exit 1
    fi

    if [ ! -f "$TRAIN_NPZ" ]; then
        echo "ERROR: train_spectrograms.npz not found after preprocessing."
        exit 1
    fi

    echo "Preprocessing complete."
fi

# ── Verification ──────────────────────────────

echo ""
echo "============================================="
echo "  Verification"
echo "============================================="

python -c "
import numpy as np
from pathlib import Path

train = np.load('data/processed/train_spectrograms.npz')
test  = np.load('data/processed/test_spectrograms.npz')

print(f'  Train chunks: {len(train[\"clean_magnitude\"]):,}')
print(f'  Train shape:  {train[\"clean_magnitude\"].shape}')
print(f'  Test chunks:  {len(test[\"clean_magnitude\"]):,}')
print(f'  Test shape:   {test[\"clean_magnitude\"].shape}')

train.close()
test.close()
print('  All checks passed.')
"

echo ""
echo "============================================="
echo "  Data pipeline complete."
echo "  End: $(date)"
echo "  You can now run:"
echo "    sbatch scripts/run_mlp.sh"
echo "    sbatch scripts/run_unet.sh"
echo "============================================="