from __future__ import annotations

import pytest

from allthemix.training.utils.early_stop import (
    EarlyStopConfig,
    EarlyStopState,
    update_early_stop,
    validate_early_stop_config,
)


def test_early_stop_ignores_epochs_before_start() -> None:
    """Verify early stopping waits until the configured start epoch."""
    config = EarlyStopConfig(
        enabled=True,
        start_epoch=5,
        patience=2,
        min_delta=0.01,
    )
    state = EarlyStopState()

    state = update_early_stop(
        state=state,
        config=config,
        epoch=1,
        metric=0.9,
    )

    assert state.best_metric == float("inf")
    assert state.stale_epochs == 0
    assert not state.should_stop


def test_early_stop_triggers_after_stale_patience() -> None:
    """Verify early stopping triggers after insufficient improvement."""
    config = EarlyStopConfig(
        enabled=True,
        start_epoch=1,
        patience=2,
        min_delta=0.01,
    )
    state = EarlyStopState()

    state = update_early_stop(
        state=state,
        config=config,
        epoch=1,
        metric=0.8,
    )
    state = update_early_stop(
        state=state,
        config=config,
        epoch=2,
        metric=0.795,
    )
    state = update_early_stop(
        state=state,
        config=config,
        epoch=3,
        metric=0.794,
    )

    assert state.should_stop
    assert state.best_metric == pytest.approx(0.8)
    assert state.best_epoch == 1


def test_early_stop_resets_after_large_enough_improvement() -> None:
    """Verify sufficient improvement resets the stale counter."""
    config = EarlyStopConfig(
        enabled=True,
        start_epoch=1,
        patience=2,
        min_delta=0.01,
    )
    state = EarlyStopState()

    for epoch, metric in (
        (1, 0.8),
        (2, 0.795),
        (3, 0.78),
    ):
        state = update_early_stop(
            state=state,
            config=config,
            epoch=epoch,
            metric=metric,
        )

    assert not state.should_stop
    assert state.stale_epochs == 0
    assert state.best_metric == pytest.approx(0.78)
    assert state.best_epoch == 3


def test_validate_early_stop_config_rejects_invalid_patience() -> None:
    """Verify invalid early-stop patience is rejected."""
    with pytest.raises(
        ValueError,
        match="early_stop_patience",
    ):
        validate_early_stop_config(
            EarlyStopConfig(
                enabled=True,
                patience=0,
            )
        )
