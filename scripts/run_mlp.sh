#!/bin/bash
#SBATCH --job-name=audio_mlp
#SBATCH --output=logs/mlp_%j.log
#SBATCH --error=logs/mlp_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# ─────────────────────────────────────────────
# run_mlp.sh `sbatch --export=ALL,GITHUB_TOKEN="ghp_tokenHere" scripts/run_mlp.sh`
# ─────────────────────────────────────────────

echo "============================================="
echo "  Audio Denoising — MLP Training"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  Start:  $(date)"
echo "============================================="

# ── Environment Setup ──────────────────────────

echo ""
echo "Setting up environment..."

cd /scratch/s4697103/AppliedML/Applied-ML-Audio-Denoising-RUG-2026 || exit 1

# 1. Purge completely and load the official RUG Python-DataScience toolchain
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# 2. Recreate your virtual environment, using cluster packages as the base
rm -rf env
python -m venv --system-site-packages env
source env/bin/activate

# 3. Clean up installation order
# Force NumPy down to 1.x immediately so packages compiling from source match
pip install "numpy<2" 

# Install your dependencies, ignoring pre-installed torch/numpy versions
pip install -r requirements.txt --ignore-installed librosa soundfile tqdm pesq

# Verify sanity check outputs
echo "Python: $(which python)"
python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

# ── Check Prerequisites ────────────────────────

echo ""
echo "Checking prerequisites..."

if [ ! -f "data/processed/train_spectrograms.npz" ] || [ ! -f "data/processed/test_spectrograms.npz" ]; then
    echo "ERROR: Preprocessed data spectrograms missing."
    echo "Run data pipeline first: sbatch scripts/run_data.sh"
    exit 1
fi

echo "  train_spectrograms.npz: found"
echo "  test_spectrograms.npz:  found"

# ── Check if already trained ──────────────────

if [ -f "outputs/models/mlp/best_model.pt" ]; then
    echo ""
    echo "WARNING: outputs/models/mlp/best_model.pt already exists."
    echo "Delete it to retrain. Skipping script actions."
else

    # ── Train MLP ─────────────────────────────────

    echo ""
    echo "============================================="
    echo "  Training MLP"
    echo "============================================="

    python src/models/mlp.py

    if [ $? -ne 0 ]; then
        echo "ERROR: MLP training failed."
        exit 1
    fi

    if [ ! -f "outputs/models/mlp/best_model.pt" ]; then
        echo "ERROR: best_model.pt not found after training."
        exit 1
    fi

    echo ""
    echo "Training complete."
    python -c "
import json
with open('outputs/models/mlp/training_log.json') as f:
    log = json.load(f)
print(f'  Best epoch:    {log[\"best_epoch\"]}')
print(f'  Best val loss: {log[\"best_val_loss\"]:.6f}')
print(f'  Total epochs:  {len(log[\"train_losses\"])}')
"

    # ── Push to GitHub (optional) ─────────────────
    # Triggers only if training succeeded AND GITHUB_TOKEN variable was passed

    if [ -n "$GITHUB_TOKEN" ]; then
        echo ""
        echo "============================================="
        echo "  Pushing MLP Results to GitHub"
        echo "============================================="

        git remote set-url origin "https://MordoTheHacker:${GITHUB_TOKEN}@github.com/MordoTheHacker/Applied-ML-Audio-Denoising-RUG-2026.git"

        git add outputs/models/mlp/best_model.pt
        git add outputs/models/mlp/norm_mean.npy
        git add outputs/models/mlp/norm_std.npy
        git add outputs/models/mlp/training_log.json
        git add outputs/results/mlp.json 2>/dev/null || true

        git commit -m "add MLP training results [SLURM job $SLURM_JOB_ID]" || echo "Nothing new to commit."
        git push origin main

        echo "Results pushed to GitHub."
    else
        echo ""
        echo "Skipping GitHub push (no GITHUB_TOKEN environment variable provided)."
    fi

fi

echo ""
echo "============================================="
echo "  MLP job processing sequence complete."
echo "  End: $(date)"
echo "============================================="