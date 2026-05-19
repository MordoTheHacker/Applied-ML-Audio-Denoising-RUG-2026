#!/bin/bash
#SBATCH --job-name=audio_mlp
#SBATCH --output=logs/mlp_%j.log
#SBATCH --error=logs/mlp_%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
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

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

cd /scratch/s4697103/AppliedML/Applied-ML-Audio-Denoising-RUG-2026 || exit 1
source env/bin/activate

# Create directories inside the repository workspace
mkdir -p logs
mkdir -p outputs/models/mlp
mkdir -p outputs/results

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Upgrade/verify PyTorch inside the env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install -r requirements.txt --quiet

echo ""
python -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

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