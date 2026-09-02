from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopConfig:
    """Configuration for validation/test error stagnation stopping."""

    enabled: bool = False
    start_epoch: int = 0
    patience: int = 20
    min_delta: float = 0.0


@dataclass
class EarlyStopState:
    """Mutable early-stop tracking state."""

    best_metric: float = float("inf")
    best_epoch: int = -1
    stale_epochs: int = 0
    should_stop: bool = False
    reason: str = ""


def validate_early_stop_config(
    config: EarlyStopConfig,
) -> None:
    """Validate early-stop settings."""
    if config.start_epoch < 0:
        raise ValueError(
            "early_stop_start_epoch must be >= 0. "
            f"Got {config.start_epoch}.",
        )

    if config.patience < 1:
        raise ValueError(
            "early_stop_patience must be >= 1. "
            f"Got {config.patience}.",
        )

    if config.min_delta < 0.0:
        raise ValueError(
            "early_stop_min_delta must be >= 0. "
            f"Got {config.min_delta}.",
        )


def update_early_stop(
    state: EarlyStopState,
    config: EarlyStopConfig,
    epoch: int,
    metric: float,
    metric_name: str = "top-1 error",
) -> EarlyStopState:
    """Update early-stop state for a lower-is-better metric."""
    if not config.enabled:
        return state

    if epoch < config.start_epoch:
        return state

    improvement = state.best_metric - metric

    if improvement > config.min_delta:
        state.best_metric = metric
        state.best_epoch = epoch
        state.stale_epochs = 0
        return state

    state.stale_epochs += 1

    if state.stale_epochs >= config.patience:
        state.should_stop = True
        state.reason = (
            f"Early stopping at epoch {epoch}: {metric_name} has not improved "
            f"by more than {config.min_delta * 100:.3f} percentage points "
            f"for {config.patience} checked epochs. "
            f"Best checked {metric_name}: {state.best_metric * 100:.2f}% "
            f"at epoch {state.best_epoch}."
        )

    return state
