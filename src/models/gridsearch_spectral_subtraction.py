import json
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
from pathlib import Path
from itertools import product
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent)) # Add src to path
from models.spectral_subtraction import SpectralSubtraction
from evaluate import evaluate_all

"""
Grid search for Spectral Subtraction hyperparameters.

Searches over:
    alpha:    [1.0, 1.5, 2.0, 3.0]   — over-subtraction factor
    beta:     [0.002, 0.01, 0.02]     — spectral floor
    lambda_n: [0.95, 0.98, 0.99]      — noise smoothing factor

Priority hierarchy (per proposal):
    1. STOI must not drop below 0.90 (noisy baseline - 0.02)
    2. Maximize PESQ
    3. Maximize CBAK (baseline was low at 2.59, target +0.3)
"""

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CLEAN_DIR  = Path("data/raw/wavs/test/clean")
NOISY_DIR  = Path("data/raw/wavs/test/noisy")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter grid
ALPHAS    = [1.0, 1.5, 2.0, 3.0]
BETAS     = [0.002, 0.01, 0.02]
LAMBDA_NS = [0.95, 0.98, 0.99]

# Constraint: STOI must stay above this threshold
STOI_MIN = 0.90

# Load noisy baseline for comparison
baseline_path = RESULTS_DIR / "noisy_baseline.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        BASELINE = json.load(f)["metrics"]
    print(f"Loaded noisy baseline: PESQ={BASELINE['PESQ']:.4f}, STOI={BASELINE['STOI']:.4f}, CBAK={BASELINE['CBAK']:.4f}")
else:
    print("Warning: noisy_baseline.json not found. Run evaluate.py first.")
    BASELINE = {"PESQ": 1.97, "STOI": 0.92, "CBAK": 2.59}

# ─────────────────────────────────────────────
# Helper: evaluate one config on N files
# ─────────────────────────────────────────────

def evaluate_config(alpha, beta, lambda_n, clean_files, noisy_files):
    """Run one hyperparameter config and return averaged metrics."""
    model = SpectralSubtraction(
        sr=16000,
        frame_len=0.025,
        frame_shift=0.010,
        noise_frames=20,
        lambda_n=lambda_n,
        alpha=alpha,
        beta=beta,
    )

    all_results = {k: [] for k in ['PESQ', 'STOI', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'SI_SDR']}

    for cf, nf in zip(clean_files, noisy_files):
        clean, sr = sf.read(cf)
        noisy, _  = sf.read(nf)

        clean = clean.astype(np.float64)
        noisy = noisy.astype(np.float64)

        if len(clean.shape) > 1:
            clean = np.mean(clean, axis=1)
        if len(noisy.shape) > 1:
            noisy = np.mean(noisy, axis=1)

        enhanced = model.enhance(noisy)
        results = evaluate_all(clean, enhanced, sr)

        for k, v in results.items():
            if not np.isnan(v):
                all_results[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_results.items() if v}

# ─────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────

def run_grid_search(n_files: int = 100):
    """
    Run grid search over all hyperparameter combinations.

    Args:
        n_files: Number of test files to evaluate per config.
    """
    clean_files = sorted(CLEAN_DIR.glob("*.wav"))[:n_files]
    noisy_files = sorted(NOISY_DIR.glob("*.wav"))[:n_files]

    total_configs = len(ALPHAS) * len(BETAS) * len(LAMBDA_NS)
    print(f"\nGrid Search: {total_configs} configurations x {n_files} files")

    results_log = []

    with tqdm(total=total_configs, desc="Grid Search") as pbar:
        for alpha, beta, lambda_n in product(ALPHAS, BETAS, LAMBDA_NS):
            metrics = evaluate_config(alpha, beta, lambda_n, clean_files, noisy_files)

            entry = {
                "alpha": alpha,
                "beta": beta,
                "lambda_n": lambda_n,
                "metrics": metrics,
                # Priority flags
                "stoi_ok": metrics.get("STOI", 0) >= STOI_MIN,
                "pesq_delta": metrics.get("PESQ", 0) - BASELINE["PESQ"],
                "cbak_delta": metrics.get("CBAK", 0) - BASELINE["CBAK"],
            }
            results_log.append(entry)
            pbar.update(1)
            pbar.set_postfix({
                "a": alpha, "β": beta, "λ": lambda_n,
                "PESQ": f"{metrics.get('PESQ', 0):.3f}",
                "STOI": f"{metrics.get('STOI', 0):.3f}",
            })

    return results_log

# ─────────────────────────────────────────────
# Analysis & Best Config Selection
# ─────────────────────────────────────────────

def find_best_config(results_log: list) -> dict:
    """
    Find best config following the priority hierarchy:
    1. STOI must not drop below STOI_MIN
    2. Maximize PESQ
    3. Maximize CBAK as tiebreaker
    """
    # Filter: STOI constraint
    valid = [r for r in results_log if r["stoi_ok"]]

    if not valid:
        print("Warning: No config met STOI constraint. Relaxing to best STOI.")
        valid = sorted(results_log, key=lambda x: x["metrics"].get("STOI", 0), reverse=True)[:5]

    # Sort by PESQ (primary), then CBAK (secondary)
    valid.sort(key=lambda x: (
        x["metrics"].get("PESQ", 0),
        x["metrics"].get("CBAK", 0)
    ), reverse=True)

    return valid[0]

def print_grid_results(results_log: list, best: dict):
    """Print summary of grid search results."""
    print("\n" + "="*75)
    print("  GRID SEARCH RESULTS")
    print("="*75)
    print(f"  {'a':<6} {'β':<8} {'λ_n':<6} {'PESQ':>8} {'STOI':>8} {'CBAK':>8} {'ΔPESQ':>8} {'ΔCBAK':>8} {'OK':>4}")
    print(f"  {'-'*75}")

    # Sort by PESQ for display
    sorted_results = sorted(results_log,
                           key=lambda x: x["metrics"].get("PESQ", 0),
                           reverse=True)

    for r in sorted_results:
        m = r["metrics"]
        ok = "v" if r["stoi_ok"] else "x"
        pesq_d = f"{r['pesq_delta']:+.3f}"
        cbak_d = f"{r['cbak_delta']:+.3f}"
        print(f"  {r['alpha']:<6} {r['beta']:<8} {r['lambda_n']:<6} "
              f"{m.get('PESQ', 0):>8.4f} {m.get('STOI', 0):>8.4f} "
              f"{m.get('CBAK', 0):>8.4f} {pesq_d:>8} {cbak_d:>8} {ok:>4}")

    print("="*75)
    print(f"\n  Best Config:")
    print(f"    alpha    = {best['alpha']}")
    print(f"    beta     = {best['beta']}")
    print(f"    lambda_n = {best['lambda_n']}")
    print(f"    PESQ     = {best['metrics'].get('PESQ', 0):.4f}  (Δ {best['pesq_delta']:+.4f})")
    print(f"    STOI     = {best['metrics'].get('STOI', 0):.4f}")
    print(f"    CBAK     = {best['metrics'].get('CBAK', 0):.4f}  (Δ {best['cbak_delta']:+.4f})")
    print("="*75 + "\n")

# ─────────────────────────────────────────────
# Heatmap Visualization
# ─────────────────────────────────────────────

def plot_heatmaps(results_log: list):
    """
    Plot Alpha vs Beta heatmaps for each lambda_n value.
    PESQ intensity shows global maximum visually.
    """
    fig, axes = plt.subplots(1, len(LAMBDA_NS), figsize=(5 * len(LAMBDA_NS), 5))

    if len(LAMBDA_NS) == 1:
        axes = [axes]

    vmin = min(r["metrics"].get("PESQ", 0) for r in results_log)
    vmax = max(r["metrics"].get("PESQ", 0) for r in results_log)

    for ax, lambda_n in zip(axes, LAMBDA_NS):
        # Build PESQ matrix: rows=alpha, cols=beta
        matrix = np.zeros((len(ALPHAS), len(BETAS)))
        stoi_matrix = np.zeros((len(ALPHAS), len(BETAS)))

        for r in results_log:
            if r["lambda_n"] == lambda_n:
                i = ALPHAS.index(r["alpha"])
                j = BETAS.index(r["beta"])
                matrix[i, j] = r["metrics"].get("PESQ", 0)
                stoi_matrix[i, j] = r["metrics"].get("STOI", 0)

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect='auto')

        # Annotate each cell with PESQ value
        for i in range(len(ALPHAS)):
            for j in range(len(BETAS)):
                stoi_ok = stoi_matrix[i, j] >= STOI_MIN
                text = f"{matrix[i, j]:.3f}"
                color = "black" if stoi_ok else "red"
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)

        ax.set_xticks(range(len(BETAS)))
        ax.set_xticklabels([str(b) for b in BETAS])
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_yticklabels([str(a) for a in ALPHAS])
        ax.set_xlabel("Beta (spectral floor)")
        ax.set_ylabel("Alpha (over-subtraction)")
        ax.set_title(f"λ_n = {lambda_n}\nPESQ (red = STOI constraint failed)")

        plt.colorbar(im, ax=ax, label="PESQ")

    plt.suptitle("Spectral Subtraction: Alpha x Beta Heatmap (PESQ intensity)",
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = FIGURES_DIR / "spectral_subtraction_gridsearch.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {out_path}")
    plt.close()

# ─────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────

def save_results(results_log: list, best: dict):
    """Save full grid search log and best config to JSON."""

    # Full log
    log_path = RESULTS_DIR / "spectral_subtraction_gridsearch.json"
    with open(log_path, "w") as f:
        json.dump({
            "grid": {"alpha": ALPHAS, "beta": BETAS, "lambda_n": LAMBDA_NS},
            "baseline": BASELINE,
            "stoi_constraint": STOI_MIN,
            "results": results_log,
            "best": best,
        }, f, indent=2)
    print(f"Full grid search log saved to {log_path}")

    # Best config only
    best_path = RESULTS_DIR / "spectral_subtraction_best.json"
    with open(best_path, "w") as f:
        json.dump({
            "model": "Spectral Subtraction (best config)",
            "hyperparameters": {
                "alpha": best["alpha"],
                "beta": best["beta"],
                "lambda_n": best["lambda_n"],
            },
            "metrics": best["metrics"],
        }, f, indent=2)
    print(f"Best config saved to {best_path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 75)
    print("  SPECTRAL SUBTRACTION — HYPERPARAMETER GRID SEARCH")
    print("=" * 75)
    print(f"  Grid: alpha={ALPHAS}, beta={BETAS}, lambda_n={LAMBDA_NS}")
    print(f"  STOI constraint: >= {STOI_MIN}")
    print(f"  Noisy baseline: PESQ={BASELINE['PESQ']:.4f}, "
          f"STOI={BASELINE['STOI']:.4f}, CBAK={BASELINE['CBAK']:.4f}")

    results_log = run_grid_search(n_files=100)

    best = find_best_config(results_log)

    print_grid_results(results_log, best)

    plot_heatmaps(results_log)

    save_results(results_log, best)

    print(f"  alpha={best['alpha']}, beta={best['beta']}, lambda_n={best['lambda_n']}")

if __name__ == "__main__":
    main()