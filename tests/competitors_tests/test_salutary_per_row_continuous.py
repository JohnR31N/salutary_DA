from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from salutary_da.policies.per_row_continuous import (
    DECISION_BUDGET_EXCLUDED,
    DECISION_GAIN_BELOW_THRESHOLD,
    DECISION_SELECTED,
    PerRowContinuousPolicyConfig,
    bounded_mean_one_weights,
    decide_per_row_continuous_device,
    decide_per_row_continuous_reference,
)


def _soft_targets(rows: int, classes: int) -> np.ndarray:
    return np.full((rows, classes), 1.0 / classes, dtype=np.float32)


def test_soft_label_policy_preserves_simplex_and_applies_all_eligible_rows() -> None:
    targets = _soft_targets(4, 3)
    raw = np.asarray(
        [
            [0.0, 0.4, 0.1],
            [0.0, 0.02, 0.01],
            [0.8, 0.1, 0.0],
            [0.0, 0.2, 0.19],
        ],
        dtype=np.float32,
    )
    config = PerRowContinuousPolicyConfig(
        mode="soft_label",
        soft_label_dose=0.1,
        minimum_gain=0.05,
        minimum_label_margin=0.01,
    )
    decision = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=1.0,
        config=config,
    )

    assert decision.eligible_rows.tolist() == [True, False, True, False]
    assert decision.applied_rows.tolist() == [True, False, True, False]
    assert decision.selected_labels.tolist() == [1, 1, 0, 1]
    assert decision.decision_codes.tolist() == [
        DECISION_SELECTED,
        DECISION_GAIN_BELOW_THRESHOLD,
        DECISION_SELECTED,
        DECISION_GAIN_BELOW_THRESHOLD,
    ]
    np.testing.assert_allclose(decision.targets_after.sum(axis=-1), 1.0)
    np.testing.assert_allclose(decision.targets_after[1], targets[1])
    np.testing.assert_allclose(decision.targets_after[3], targets[3])
    assert int(decision.applied_count) == 2
    assert bool(decision.scores_valid)


def test_origin_soft_label_masks_ground_truth_and_uses_best_positive_alternative(
) -> None:
    targets = np.eye(3, dtype=np.float32)[[0, 1]]
    raw = np.asarray(
        [
            [100.0, 4.0, 1.0],
            [3.0, 100.0, 2.0],
        ],
        dtype=np.float32,
    )
    decision = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=1.0,
        config=PerRowContinuousPolicyConfig(
            mode="soft_label",
            maximum_rows=2,
            soft_label_dose=0.1,
            minimum_gain=0.0,
            fallback_enabled=False,
        ),
    )

    assert decision.selected_labels.tolist() == [1, 0]
    assert decision.applied_rows.tolist() == [True, True]
    np.testing.assert_allclose(
        decision.targets_after,
        np.asarray([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=np.float32),
    )


def test_soft_label_fallback_changes_one_row_only_when_no_gate_passes() -> None:
    targets = _soft_targets(5, 4)
    raw = np.arange(20, dtype=np.float32).reshape(5, 4) / 1000.0
    config = PerRowContinuousPolicyConfig(
        mode="soft_label",
        minimum_gain=1.0,
        fallback_enabled=True,
        fallback_soft_label_dose=0.01,
    )
    decision = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=0.1,
        config=config,
    )

    assert int(decision.eligible_count) == 0
    assert int(decision.applied_count) == 1
    assert bool(decision.fallback_applied)
    assert np.flatnonzero(decision.applied_rows).tolist() == [4]
    assert float(decision.doses[4]) == np.float32(0.01)
    np.testing.assert_allclose(decision.targets_after.sum(axis=-1), 1.0)


def test_soft_label_budget_uses_stable_score_order() -> None:
    targets = _soft_targets(4, 3)
    raw = np.asarray(
        [
            [0.0, 0.4, 0.1],
            [0.0, 0.4, 0.2],
            [0.0, 0.3, 0.1],
            [0.0, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    decision = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=1.0,
        config=PerRowContinuousPolicyConfig(
            mode="soft_label",
            maximum_rows=2,
            soft_label_dose=0.1,
        ),
    )
    assert decision.eligible_rows.tolist() == [True, True, True, True]
    assert decision.applied_rows.tolist() == [True, True, False, False]
    assert decision.decision_codes.tolist() == [
        DECISION_SELECTED,
        DECISION_SELECTED,
        DECISION_BUDGET_EXCLUDED,
        DECISION_BUDGET_EXCLUDED,
    ]


def test_reweight_policy_is_bounded_mean_one_and_meets_ess_floor() -> None:
    rows = 128
    targets = _soft_targets(rows, 100)
    raw = np.linspace(-4.0, 5.0, rows * 100, dtype=np.float32).reshape(
        rows,
        100,
    )
    config = PerRowContinuousPolicyConfig(
        mode="reweight",
        maximum_weight_deviation=0.2,
        minimum_relative_ess=0.99,
    )
    decision = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=0.1,
        config=config,
        sample_weight_scores=np.linspace(-2.0, 3.0, rows, dtype=np.float32),
    )

    assert np.all(decision.weights >= 0.8 - 1e-6)
    assert np.all(decision.weights <= 1.2 + 1e-6)
    np.testing.assert_allclose(np.mean(decision.weights), 1.0, atol=1e-6)
    assert float(decision.relative_ess) >= 0.99 - 1e-6
    np.testing.assert_array_equal(decision.targets_after, targets)
    assert np.all(decision.eligible_rows)
    assert np.all(decision.decision_codes == DECISION_SELECTED)


def test_public_weight_map_preserves_mean_bounds_order_and_ess() -> None:
    scores = jnp.asarray([-2.0, -0.5, 0.0, 1.0, 4.0], dtype=jnp.float32)
    weights = np.asarray(
        bounded_mean_one_weights(
            scores,
            maximum_deviation=0.2,
            temperature=1.0,
            minimum_relative_ess=0.9,
        )
    )
    np.testing.assert_allclose(np.mean(weights), 1.0, atol=2e-6)
    assert np.min(weights) >= 0.8
    assert np.max(weights) <= 1.2
    assert np.all(np.diff(weights) >= 0.0)
    relative_ess = np.square(np.sum(weights)) / (
        weights.size * np.sum(np.square(weights))
    )
    assert relative_ess >= 0.9


def test_nonfinite_score_fails_closed() -> None:
    targets = _soft_targets(4, 3)
    raw = np.ones_like(targets)
    raw[2, 1] = np.nan
    config = PerRowContinuousPolicyConfig(
        mode="soft_label",
        fallback_enabled=True,
    )
    decision = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=0.1,
        config=config,
    )

    assert not bool(decision.scores_valid)
    assert int(decision.applied_count) == 0
    np.testing.assert_array_equal(decision.targets_after, targets)
    np.testing.assert_array_equal(decision.weights, np.ones(4, dtype=np.float32))


def test_device_policy_matches_global_reference() -> None:
    devices = jax.local_device_count()
    local_rows = 2
    classes = 5
    rows = devices * local_rows
    targets = _soft_targets(rows, classes)
    raw = np.linspace(-0.2, 0.8, rows * classes, dtype=np.float32).reshape(
        rows,
        classes,
    )
    config = PerRowContinuousPolicyConfig(
        mode="soft_label",
        soft_label_dose=0.025,
        minimum_gain=0.0,
        minimum_label_margin=0.0,
    )
    reference = decide_per_row_continuous_reference(
        raw,
        targets,
        learning_rate=0.1,
        config=config,
    )
    device = decide_per_row_continuous_device(
        jnp.asarray(raw.reshape(devices, local_rows, classes)),
        jnp.asarray(targets.reshape(devices, local_rows, classes)),
        jnp.full((devices,), 0.1, dtype=jnp.float32),
        config,
    )

    np.testing.assert_allclose(
        np.asarray(device.targets_after).reshape(rows, classes),
        reference.targets_after,
    )
    np.testing.assert_array_equal(
        np.asarray(device.selected_labels).reshape(rows),
        reference.selected_labels,
    )
    assert np.all(np.asarray(device.applied_count) == int(reference.applied_count))
