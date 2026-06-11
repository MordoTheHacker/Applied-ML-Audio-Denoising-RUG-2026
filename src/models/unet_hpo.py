import sys
import json
from pathlib import Path

import torch
import optuna

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.unet import SpectrogramDataset, train_unet


def objective(trial):
    TRAIN_NPZ = Path("data/processed/train_spectrograms.npz")
    OUTPUT_ROOT = Path("outputs/hpo/unet")

    full_train_ds = SpectrogramDataset(TRAIN_NPZ)

    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size

    generator = torch.Generator().manual_seed(69)

    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds,
        [train_size, val_size],
        generator=generator,
    )

    base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])
    
    dropout_enc = trial.suggest_float("dropout_enc", 0.0, 0.3)
    dropout_bottleneck = trial.suggest_float("dropout_bottleneck", 0.2, 0.6)
    dropout_dec = trial.suggest_float("dropout_dec", 0.0, 0.3)

    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    trial_dir = OUTPUT_ROOT / f"trial_{trial.number}"

    train_unet(
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=trial_dir,
        base_filters=base_filters,
        dropout_enc=dropout_enc,
        dropout_bottleneck=dropout_bottleneck,
        dropout_dec=dropout_dec,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=30,
        patience=5,
    )

    log_path = trial_dir / "training_log.json"

    with open(log_path, "r") as f:
        log_data = json.load(f)

    best_val_loss = log_data["best_val_loss"]

    trial.set_user_attr("best_epoch", log_data["best_epoch"])
    trial.set_user_attr("trial_dir", str(trial_dir))

    return best_val_loss


def main():
    OUTPUT_ROOT = Path("outputs/hpo/unet")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(study_name="unet_hpo",direction="minimize")

    study.optimize(objective,n_trials=20)

    print(f"Trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_trial_summary = {
        "best_trial": study.best_trial.number,
        "best_val_loss": study.best_value,
        "best_params": study.best_params,
        "best_epoch": study.best_trial.user_attrs.get("best_epoch"),
        "trial_dir": study.best_trial.user_attrs.get("trial_dir"),
    }

    best_path = OUTPUT_ROOT / "best_trial.json"

    with open(best_path, "w") as f:
        json.dump(best_trial_summary, f, indent=2)

    all_trials = []

    for trial in study.trials:
        all_trials.append(
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "trial_dir": trial.user_attrs.get("trial_dir"),
            }
        )

    trials_path = OUTPUT_ROOT / "all_trials.json"

    with open(trials_path, "w") as f:
        json.dump(all_trials, f, indent=2)

    print(f"\nBest trial saved to: {best_path}")
    print(f"All trials saved to: {trials_path}")


if __name__ == "__main__":
    main()