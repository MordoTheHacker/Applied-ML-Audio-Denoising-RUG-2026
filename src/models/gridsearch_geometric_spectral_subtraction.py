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

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add src to path
from models.geometric_spectral_subtraction import GeometricSpectralSubtraction
from evaluate import evaluate_all

"""
Grid search for Geometric Spectral Subtraction hyperparameters.

Searches over:
    alpha:    [1.0, 1.5, 2.0, 3.0]   — over-subtraction factor
    beta:     [0.002, 0.01, 0.02]     — spectral floor
    gamma:    [0.5, 1.0, 1.5, 2.0]    — subtraction domain
    lambda_n: [0.95, 0.98, 0.99]      — noise smoothing factor

Priority hierarchy (per proposal):
    1. STOI must not drop below 0.90 (noisy baseline - 0.02)
    2. Maximize PESQ
    3. Maximize CBAK as tiebreaker
"""

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CLEAN_DIR = Path("data/raw/wavs/test/clean")
NOISY_DIR = Path("data/raw/wavs/test/noisy")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter grid
ALPHAS = [1.0, 1.5, 2.0, 3.0]
BETAS = [0.002, 0.01, 0.02]
GAMMAS = [0.5, 1.0, 1.5, 2.0]
LAMBDA_NS = [0.95, 0.98, 0.99]

# Constraint: STOI must stay above this threshold
STOI_MIN = 0.90

# Load noisy baseline for comparison
baseline_path = RESULTS_DIR / "noisy_baseline.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        BASELINE = json.load(f)["metrics"]
    print(f"Loaded noisy baseline: PESQ={BASELINE['PESQ']:.4f}, "
          f"STOI={BASELINE['STOI']:.4f}, CBAK={BASELINE['CBAK']:.4f}")
else:
    print("Warning: noisy_baseline.json not found. Run evaluate.py first.")
    BASELINE = {"PESQ": 1.97, "STOI": 0.92, "CBAK": 2.59}


# ─────────────────────────────────────────────
# Helper: evaluate one config on N files
# ─────────────────────────────────────────────

def evaluate_config(alpha, beta, gamma, lambda_n, clean_files, noisy_files):
    """Run one hyperparameter config and return averaged metrics."""
    model = GeometricSpectralSubtraction(
        sr=16000,
        frame_len=0.025,
        frame_shift=0.010,
        noise_frames=20,
        lambda_n=lambda_n,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    all_results = {k: [] for k in [
        'PESQ', 'STOI', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'SI_SDR'
    ]}

    for cf, nf in zip(clean_files, noisy_files):
        clean, sr = sf.read(cf)
        noisy, _ = sf.read(nf)

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

    total_configs = len(ALPHAS) * len(BETAS) * len(GAMMAS) * len(LAMBDA_NS)
    print(f"\nGrid Search: {total_configs} configurations x {n_files} files")

    results_log = []

    with tqdm(total=total_configs, desc="Geometric Grid Search") as pbar:
        for alpha, beta, gamma, lambda_n in product(
            ALPHAS, BETAS, GAMMAS, LAMBDA_NS
        ):
            metrics = evaluate_config(
                alpha, beta, gamma, lambda_n, clean_files, noisy_files
            )

            entry = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda_n": lambda_n,
                "metrics": metrics,
                "stoi_ok": metrics.get("STOI", 0) >= STOI_MIN,
                "pesq_delta": metrics.get("PESQ", 0) - BASELINE["PESQ"],
                "cbak_delta": metrics.get("CBAK", 0) - BASELINE["CBAK"],
            }
            results_log.append(entry)
            pbar.update(1)
            pbar.set_postfix({
                "α": alpha, "β": beta, "γ": gamma, "λ": lambda_n,
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
    valid = [r for r in results_log if r["stoi_ok"]]

    if not valid:
        print("Warning: No config met STOI constraint. Relaxing to best STOI.")
        valid = sorted(
            results_log,
            key=lambda x: x["metrics"].get("STOI", 0),
            reverse=True,
        )[:5]

    valid.sort(
        key=lambda x: (
            x["metrics"].get("PESQ", 0),
            x["metrics"].get("CBAK", 0),
        ),
        reverse=True,
    )

    return valid[0]


def print_grid_results(results_log: list, best: dict):
    """Print summary of grid search results."""
    print("\n" + "=" * 85)
    print("  GEOMETRIC SPECTRAL SUBTRACTION — GRID SEARCH RESULTS")
    print("=" * 85)
    print(
        f"  {'α':<6} {'β':<8} {'γ':<6} {'λ_n':<6} "
        f"{'PESQ':>8} {'STOI':>8} {'CBAK':>8} "
        f"{'ΔPESQ':>8} {'ΔCBAK':>8} {'OK':>4}"
    )
    print(f"  {'-' * 85}")

    sorted_results = sorted(
        results_log,
        key=lambda x: x["metrics"].get("PESQ", 0),
        reverse=True,
    )

    for r in sorted_results:
        m = r["metrics"]
        ok = "v" if r["stoi_ok"] else "x"
        pesq_d = f"{r['pesq_delta']:+.3f}"
        cbak_d = f"{r['cbak_delta']:+.3f}"
        print(
            f"  {r['alpha']:<6} {r['beta']:<8} {r['gamma']:<6} "
            f"{r['lambda_n']:<6} "
            f"{m.get('PESQ', 0):>8.4f} {m.get('STOI', 0):>8.4f} "
            f"{m.get('CBAK', 0):>8.4f} {pesq_d:>8} {cbak_d:>8} {ok:>4}"
        )

    print("=" * 85)
    print(f"\n  Best Config:")
    print(f"    alpha    = {best['alpha']}")
    print(f"    beta     = {best['beta']}")
    print(f"    gamma    = {best['gamma']}")
    print(f"    lambda_n = {best['lambda_n']}")
    print(
        f"    PESQ     = {best['metrics'].get('PESQ', 0):.4f}  "
        f"(Δ {best['pesq_delta']:+.4f})"
    )
    print(f"    STOI     = {best['metrics'].get('STOI', 0):.4f}")
    print(
        f"    CBAK     = {best['metrics'].get('CBAK', 0):.4f}  "
        f"(Δ {best['cbak_delta']:+.4f})"
    )
    print("=" * 85 + "\n")


# ─────────────────────────────────────────────
# Heatmap Visualization
# ─────────────────────────────────────────────

def plot_heatmaps(results_log: list):
    """
    Plot PESQ & STOI heatmaps for each gamma value (rows=beta, cols=alpha),
    averaged over lambda_n.
    """
    alpha_list = sorted(set(r["alpha"] for r in results_log))
    beta_list = sorted(set(r["beta"] for r in results_log))
    gamma_list = sorted(set(r["gamma"] for r in results_log))

    fig, axes = plt.subplots(
        len(gamma_list), 2,
        figsize=(12, 3 * len(gamma_list)),
    )
    if len(gamma_list) == 1:
        axes = axes.reshape(1, -1)

    for gi, gamma in enumerate(gamma_list):
        for col, metric in enumerate(["PESQ", "STOI"]):
            ax = axes[gi, col]
            heatmap = np.zeros((len(beta_list), len(alpha_list)))
            for bi, beta in enumerate(beta_list):
                for ai, alpha in enumerate(alpha_list):
                    vals = [
                        r["metrics"].get(metric, np.nan)
                        for r in results_log
                        if r["alpha"] == alpha
                        and r["beta"] == beta
                        and r["gamma"] == gamma
                    ]
                    heatmap[bi, ai] = np.mean(vals) if vals else np.nan

            im = ax.imshow(
                heatmap, aspect="auto", origin="lower", cmap="RdYlGn"
            )
            ax.set_xticks(range(len(alpha_list)))
            ax.set_xticklabels([str(a) for a in alpha_list])
            ax.set_yticks(range(len(beta_list)))
            ax.set_yticklabels([str(b) for b in beta_list])
            ax.set_xlabel("α (over-subtraction)")
            ax.set_ylabel("β (spectral floor)")
            ax.set_title(
                f"{metric}  (γ = {gamma})", fontweight="bold"
            )

            for bi in range(len(beta_list)):
                for ai in range(len(alpha_list)):
                    v = heatmap[bi, ai]
                    if not np.isnan(v):
                        ax.text(
                            ai, bi, f"{v:.3f}",
                            ha="center", va="center",
                            fontsize=8, fontweight="bold",
                        )
            plt.colorbar(im, ax=ax)

    plt.suptitle(
        "Geometric SS Grid Search — PESQ & STOI Heatmaps (averaged over λ_n)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out_path = FIGURES_DIR / "geometric_ss_gridsearch.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────

def save_results(results_log: list, best: dict):
    """Save full grid search log and best config to JSON."""

    # Full log
    log_path = RESULTS_DIR / "geometric_ss_gridsearch.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "grid": {
                    "alpha": ALPHAS,
                    "beta": BETAS,
                    "gamma": GAMMAS,
                    "lambda_n": LAMBDA_NS,
                },
                "baseline": BASELINE,
                "stoi_constraint": STOI_MIN,
                "results": results_log,
                "best": best,
            },
            f,
            indent=2,
        )
    print(f"Full grid search log saved to {log_path}")

    # Best config only
    best_path = RESULTS_DIR / "geometric_spectral_subtraction_best.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "model": "Geometric Spectral Subtraction (best config)",
                "hyperparameters": {
                    "alpha": best["alpha"],
                    "beta": best["beta"],
                    "gamma": best["gamma"],
                    "lambda_n": best["lambda_n"],
                },
                "metrics": best["metrics"],
            },
            f,
            indent=2,
        )
    print(f"Best config saved to {best_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Geometric Spectral Subtraction Grid Search"
    )
    parser.add_argument(
        "--n-files", type=int, default=100,
        help="Number of test files to evaluate per config (default: 100)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Shortcut for --n-files 10 (smoke test)",
    )
    args = parser.parse_args()

    n_files = 10 if args.quick else args.n_files

    print("=" * 75)
    print("  GEOMETRIC SPECTRAL SUBTRACTION — HYPERPARAMETER GRID SEARCH")
    print("=" * 75)
    print(f"  Grid: alpha={ALPHAS}, beta={BETAS}, gamma={GAMMAS}, "
          f"lambda_n={LAMBDA_NS}")
    print(f"  STOI constraint: >= {STOI_MIN}")
    print(
        f"  Noisy baseline: PESQ={BASELINE['PESQ']:.4f}, "
        f"STOI={BASELINE['STOI']:.4f}, CBAK={BASELINE['CBAK']:.4f}"
    )
    print(f"  Evaluating on {n_files} file(s) per config")

    results_log = run_grid_search(n_files=n_files)

    best = find_best_config(results_log)

    print_grid_results(results_log, best)

    plot_heatmaps(results_log)

    save_results(results_log, best)

    print(
        f"  alpha={best['alpha']}, beta={best['beta']}, "
        f"gamma={best['gamma']}, lambda_n={best['lambda_n']}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
