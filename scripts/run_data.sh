#!/bin/bash
#SBATCH --job-name=audio_preprocess
#SBATCH --output=logs/preprocess_%j.log
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --partition=regular
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ─────────────────────────────────────────────
# run_data.sh
#
# SLURM job: Run preprocessing pipeline only.
# Dataset must already be downloaded to data/raw/wavs/
# ─────────────────────────────────────────────

echo "============================================="
echo "  Audio Denoising — Preprocessing"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  Start:  $(date)"
echo "============================================="

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd /scratch/s4697103/AppliedML/Applied-ML-Audio-Denoising-RUG-2026

source env/bin/activate

# Force Python libraries to respect SLURM's allocated CPU footprint
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python: $(which python)"

mkdir -p logs

# ── Verify dataset exists ─────────────────────

echo ""
echo "Checking dataset..."

TRAIN_DIR="data/raw/wavs/train/clean"
TEST_DIR="data/raw/wavs/test/clean"

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: $TRAIN_DIR not found."
    echo "Download the dataset first from the login node:"
    echo "  python src/data.py"
    exit 1
fi

TRAIN_COUNT=$(find "$TRAIN_DIR" -maxdepth 1 -type f | wc -l)
TEST_COUNT=$(find "$TEST_DIR" -maxdepth 1 -type f | wc -l)
echo "  Train files: $TRAIN_COUNT"
echo "  Test files:  $TEST_COUNT"

if [ "$TRAIN_COUNT" -lt 1000 ]; then
    echo "ERROR: Too few training files. Dataset may be incomplete."
    exit 1
fi

# ── Check if already preprocessed ────────────

TRAIN_NPZ="data/processed/train_spectrograms.npz"
TEST_NPZ="data/processed/test_spectrograms.npz"

if [ -f "$TRAIN_NPZ" ] && [ -f "$TEST_NPZ" ]; then
    echo ""
    echo "Preprocessed data already exists. Skipping."
    python -c "
import numpy as np
d = np.load('$TRAIN_NPZ')
print(f'  Train chunks: {len(d[\"clean_magnitude\"]):,}')
print(f'  Shape: {d[\"clean_magnitude\"].shape}')
d.close()
d = np.load('$TEST_NPZ')
print(f'  Test chunks: {len(d[\"clean_magnitude\"]):,}')
d.close()
"
    exit 0
fi

# ── Run preprocessing ─────────────────────────

echo ""
echo "============================================="
echo "  Running Preprocessing Pipeline"
echo "============================================="

python src/preprocess.py

if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed."
    exit 1
fi

# ── Verify output ─────────────────────────────

if [ ! -f "$TRAIN_NPZ" ]; then
    echo "ERROR: train_spectrograms.npz not found after preprocessing."
    exit 1
fi

echo ""
echo "============================================="
echo "  Verification"
echo "============================================="

python -c "
import numpy as np
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
echo "  Preprocessing complete."
echo "  End: $(date)"
echo "  You can now run:"
echo "    sbatch scripts/run_mlp.sh"
echo "    sbatch scripts/run_unet.sh"
echo "============================================="