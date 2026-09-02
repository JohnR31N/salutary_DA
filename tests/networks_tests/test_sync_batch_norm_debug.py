from __future__ import annotations

import numpy as np

from allthemix.debug.sync_batch_norm import (
    BATCH_NORM_EPSILON,
    compute_batch_statistics,
    expected_normalized_batch,
    make_probe_batch,
    run_exact_probe,
    update_expected_running_statistics,
)


def test_probe_batch_has_distinct_device_statistics() -> None:
    """Verify the synthetic probe can expose missing synchronization."""
    batch = make_probe_batch(
        num_devices=4,
        per_device_batch_size=2,
        height=3,
        width=3,
        channels=2,
    )
    local_mean, _ = compute_batch_statistics(
        batch=batch,
        synchronized=False,
    )

    assert batch.shape == (
        4,
        2,
        3,
        3,
        2,
    )
    assert not np.allclose(
        local_mean[0],
        local_mean[-1],
    )


def test_expected_sync_normalization_uses_global_statistics() -> None:
    """Verify the expected formula normalizes the full replica group."""
    batch = make_probe_batch(
        num_devices=4,
        per_device_batch_size=2,
        height=3,
        width=3,
        channels=2,
    )
    mean, variance = compute_batch_statistics(
        batch=batch,
        synchronized=True,
    )
    normalized = expected_normalized_batch(
        batch=batch,
        mean=mean,
        variance=variance,
        synchronized=True,
    )

    actual_mean = np.mean(
        normalized,
        axis=(
            0,
            1,
            2,
            3,
        ),
    )
    actual_variance = np.var(
        normalized,
        axis=(
            0,
            1,
            2,
            3,
        ),
    )
    expected_variance = variance / (
        variance + BATCH_NORM_EPSILON
    )

    np.testing.assert_allclose(
        actual_mean,
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual_variance,
        expected_variance,
        atol=1e-6,
    )


def test_running_statistics_use_flax_momentum_convention() -> None:
    """Verify momentum 0.9 gives ten percent weight to a new batch."""
    running_mean = np.zeros(
        (2,),
        dtype=np.float32,
    )
    running_variance = np.ones(
        (2,),
        dtype=np.float32,
    )
    batch_mean = np.asarray(
        [
            2.0,
            4.0,
        ],
        dtype=np.float32,
    )
    batch_variance = np.asarray(
        [
            3.0,
            5.0,
        ],
        dtype=np.float32,
    )

    new_mean, new_variance = update_expected_running_statistics(
        running_mean=running_mean,
        running_variance=running_variance,
        batch_mean=batch_mean,
        batch_variance=batch_variance,
    )

    np.testing.assert_allclose(
        new_mean,
        [
            0.2,
            0.4,
        ],
    )
    np.testing.assert_allclose(
        new_variance,
        [
            1.2,
            1.4,
        ],
    )


def test_exact_probe_runs_on_one_device() -> None:
    """Smoke-test the PMAP formula probe on the local JAX backend."""
    result = run_exact_probe(
        num_devices=1,
        per_device_batch_size=2,
        height=2,
        width=2,
        channels=2,
        steps=2,
        formula_tolerance=2e-3,
        replica_tolerance=1e-6,
    )

    assert result["passed"] is True
