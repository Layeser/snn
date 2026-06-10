from pathlib import Path

import mlflow


def setup_experiment(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)


def log_hyperparameters(params: dict) -> None:
    mlflow.log_params(params)


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        step=epoch,
    )


def log_final_metrics(best_val_acc: float, best_epoch: int) -> None:
    mlflow.log_metrics(
        {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }
    )


def log_artifacts(config_path: str | Path, checkpoint_path: str | Path) -> None:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    if config_path.exists():
        mlflow.log_artifact(str(config_path), artifact_path="config")
    if checkpoint_path.exists():
        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
