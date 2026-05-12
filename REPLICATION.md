# Replication Guide: Audio Denoising MLP Training

This guide allows anyone with a CUDA-capable GPU to fully replicate our MLP training pipeline in a single command. The script handles everything automatically: environment setup, dataset download, preprocessing, training, and pushing results back to the repository.

---

## Requirements

Before running anything, make sure your machine has:

- Python 3.9 or higher
- A CUDA-capable NVIDIA GPU (RTX 2060 or better recommended)
- NVIDIA drivers installed (`nvidia-smi` works in terminal)
- Git installed and configured
- ~35 GB free disk space (dataset ~22 GB, processed data ~12 GB, model ~200 MB)

---

## Step 1 — Get Access to the Repository

You need to be added as a collaborator before you can push results back.

**Ask the repo owner to add you at:**
```
https://github.com/MordoTheHacker/Applied-ML-Audio-Denoising-RUG-2026/settings/access
```

Once added, you will receive an email invitation. Accept it before continuing.

---

## Step 2 — Generate a GitHub Personal Access Token

You need a token so the script can push results back to the repository on your behalf.

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Give it a name, e.g. `ml-replication`
4. Set expiration to **30 days** (enough for this task)
5. Under **Select scopes**, tick **`repo`** (the top-level checkbox)
6. Click **Generate token** at the bottom
7. **Copy the token immediately** — GitHub will not show it again

Your token will look like:
```
ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Keep it private. Do not share it or commit it to any file.

---

## Step 3 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/MordoTheHacker/Applied-ML-Audio-Denoising-RUG-2026
cd Applied-ML-Audio-Denoising-RUG-2026
```

---

## Step 4 — Run the Replication Script

Replace `YOUR_TOKEN_HERE` with the token you copied in Step 2:

```bash
python replicate.py --github_token YOUR_TOKEN_HERE
```

That's it. The script will now run through all stages automatically.

---

## What the Script Does

| Stage | What happens | Estimated time |
|-------|-------------|----------------|
| 1. Requirements | Detects your CUDA version and installs correct PyTorch + all dependencies | 2–5 min |
| 2. Download dataset | Downloads VoiceBank+DEMAND (~3 GB) from HuggingFace | 10–20 min |
| 3. Extract wavs | Converts dataset to wav files in `data/raw/wavs/` | 5–10 min |
| 4. Preprocess | Computes STFT spectrograms, saves to `data/processed/` | 20–40 min |
| 5. Train MLP | Trains model with early stopping, saves best weights | 30-60 min (GPU) |
| 6. Evaluate | Runs PESQ, STOI, and other metrics on test set | 5–10 min |
| 7. Push results | Commits model weights and metrics to GitHub | 1–2 min |
| **Total** | | **~1–4 hours** |

Each stage is skipped automatically if its output already exists, so you can safely re-run the script if something fails partway through.

---

## What Gets Pushed to GitHub

The script only pushes small result files — **not the dataset or processed data**:

```
outputs/models/mlp/best_model.pt          # Trained model weights (~16 MB)
outputs/models/mlp/norm_mean.npy          # Normalization stats (~3 MB)
outputs/models/mlp/norm_std.npy           # Normalization stats (~3 MB)
outputs/models/mlp/training_log.json      # Loss curves + hyperparameters
outputs/results/mlp.json                  # Final evaluation metrics
```

---

## Optional Flags

### Skip pushing to GitHub (run locally only)
```bash
python replicate.py --github_token YOUR_TOKEN_HERE --skip_push
```

### Use a different GitHub username or repo
```bash
python replicate.py --github_token YOUR_TOKEN_HERE \
                    --github_username YOUR_USERNAME \
                    --repo_name YOUR_REPO_NAME
```

---

## Verifying Your GPU is Detected

After Step 4 starts, you should see something like:

```
Device: cuda
GPU available: True
Device: NVIDIA GeForce RTX 3080
```

If you see `Device: cpu`, your GPU was not detected. Check that your NVIDIA drivers are installed and `nvidia-smi` works in terminal.

---

## Troubleshooting

**`nvidia-smi` not found**
Install NVIDIA drivers from https://www.nvidia.com/drivers

**`ModuleNotFoundError`**
Make sure you are running Python 3.9+ and in the repo root directory:
```bash
python --version
ls src/   # should show data.py, preprocess.py, etc.
```

**Dataset download fails**
HuggingFace may be slow. Re-run the script — it resumes from where it left off.

**Preprocessing runs out of disk space**
You need ~35 GB free. Clear space and re-run — preprocessing also resumes automatically.

**`git push` fails with 403**
Your token may have expired or lacks `repo` scope. Generate a new one following Step 2.

**Training loss is NaN**
This means the learning rate is too high. Open `src/models/mlp.py` and change `lr=1e-3` to `lr=1e-4` in `main()`, then re-run.

---

## After the Script Finishes

Results will appear in the GitHub repository under `outputs/results/mlp.json`. The repo owner can then pull the results:

```bash
git pull
cat outputs/results/mlp.json
```

---

## Contact

If anything goes wrong, open an issue on the repository or contact the team directly.