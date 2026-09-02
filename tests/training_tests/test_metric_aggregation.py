from __future__ import annotations

import pytest

from allthemix.training.utils.metric_aggregation import aggregate_epoch_metric_lists


def test_aggregate_epoch_metric_lists_preserves_mean_min_and_max() -> None:
    """Verify that metric suffixes select the intended epoch reduction."""
    metrics = aggregate_epoch_metric_lists(
        {
            "mix_lam_mean": [
                0.2,
                0.8,
            ],
            "mix_lam_min": [
                0.2,
                0.8,
            ],
            "mix_lam_max": [
                0.2,
                0.8,
            ],
            "mix_changed_ratio": [
                0.8,
                0.2,
            ],
        }
    )

    assert metrics["mix_lam_mean"] == pytest.approx(0.5)
    assert metrics["mix_lam_min"] == pytest.approx(0.2)
    assert metrics["mix_lam_max"] == pytest.approx(0.8)
    assert metrics["mix_changed_ratio"] == pytest.approx(0.5)


def test_aggregate_epoch_metric_lists_ignores_empty_metrics() -> None:
    """Verify that an empty metric series is not emitted."""
    metrics = aggregate_epoch_metric_lists(
        {
            "mix_lam_mean": [],
        }
    )

    assert metrics == {}
