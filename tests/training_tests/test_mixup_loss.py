from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.training.losses.cross_entropy import (
    hard_cross_entropy,
    hard_cross_entropy_per_sample,
    soft_cross_entropy,
    soft_cross_entropy_per_sample,
)
from allthemix.training.losses.mixup_loss import mixup_loss


def _assert_allclose(
    actual,
    expected,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Assert allclose."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        atol=atol,
        rtol=rtol,
    )


def test_soft_cross_entropy_per_sample_matches_manual_formula() -> None:
    """Verify that soft cross entropy per sample matches manual formula."""
    logits = jnp.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.5, 1.5, -0.5],
        ],
        dtype=jnp.float32,
    )

    labels = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    log_probs = jax.nn.log_softmax(
        logits,
        axis=-1,
    )

    expected = -jnp.sum(
        labels * log_probs,
        axis=-1,
    )

    actual = soft_cross_entropy_per_sample(
        logits=logits,
        labels=labels,
    )

    _assert_allclose(
        actual,
        expected,
    )


def test_soft_cross_entropy_is_mean_of_per_sample_losses() -> None:
    """Verify that soft cross entropy is mean of per sample losses."""
    logits = jnp.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.5, 1.5, -0.5],
        ],
        dtype=jnp.float32,
    )

    labels = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    per_sample = soft_cross_entropy_per_sample(
        logits=logits,
        labels=labels,
    )

    actual = soft_cross_entropy(
        logits=logits,
        labels=labels,
    )

    expected = jnp.mean(
        per_sample,
    )

    _assert_allclose(
        actual,
        expected,
    )


def test_hard_cross_entropy_per_sample_matches_soft_version() -> None:
    """Verify that hard cross entropy per sample matches soft version."""
    logits = jnp.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.5, 1.5, -0.5],
        ],
        dtype=jnp.float32,
    )

    labels = jnp.asarray(
        [
            0,
            1,
        ],
        dtype=jnp.int32,
    )

    one_hot_labels = jax.nn.one_hot(
        labels,
        3,
    )

    expected = soft_cross_entropy_per_sample(
        logits=logits,
        labels=one_hot_labels,
    )

    actual = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels,
        num_classes=3,
    )

    _assert_allclose(
        actual,
        expected,
    )


def test_hard_cross_entropy_is_mean_of_per_sample_losses() -> None:
    """Verify that hard cross entropy is mean of per sample losses."""
    logits = jnp.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.5, 1.5, -0.5],
        ],
        dtype=jnp.float32,
    )

    labels = jnp.asarray(
        [
            0,
            1,
        ],
        dtype=jnp.int32,
    )

    per_sample = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels,
        num_classes=3,
    )

    actual = hard_cross_entropy(
        logits=logits,
        labels=labels,
        num_classes=3,
    )

    expected = jnp.mean(
        per_sample,
    )

    _assert_allclose(
        actual,
        expected,
    )


def test_mixup_loss_with_scalar_lambda_matches_manual_formula() -> None:
    """Verify that mixup loss with scalar lambda matches manual formula."""
    logits = jnp.asarray(
        [
            [3.0, 0.0, -1.0],
            [0.2, 2.0, -0.5],
            [0.1, -0.3, 1.7],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [
            0,
            1,
            2,
        ],
        dtype=jnp.int32,
    )

    labels_b = jnp.asarray(
        [
            1,
            2,
            0,
        ],
        dtype=jnp.int32,
    )

    lam = jnp.asarray(
        0.7,
        dtype=jnp.float32,
    )

    loss_a = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_a,
        num_classes=3,
    )

    loss_b = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_b,
        num_classes=3,
    )

    expected = jnp.mean(
        lam * loss_a + (
            1.0 - lam
        ) * loss_b,
    )

    actual = mixup_loss(
        logits=logits,
        labels_a=labels_a,
        labels_b=labels_b,
        num_classes=3,
        lam=lam,
    )

    _assert_allclose(
        actual,
        expected,
    )


def test_mixup_loss_with_per_sample_lambda_matches_manual_formula() -> None:
    """Verify that mixup loss with per sample lambda matches manual formula."""
    logits = jnp.asarray(
        [
            [3.0, 0.0, -1.0],
            [0.2, 2.0, -0.5],
            [0.1, -0.3, 1.7],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [
            0,
            1,
            2,
        ],
        dtype=jnp.int32,
    )

    labels_b = jnp.asarray(
        [
            1,
            2,
            0,
        ],
        dtype=jnp.int32,
    )

    lam = jnp.asarray(
        [
            0.9,
            0.5,
            0.1,
        ],
        dtype=jnp.float32,
    )

    loss_a = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_a,
        num_classes=3,
    )

    loss_b = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_b,
        num_classes=3,
    )

    expected = jnp.mean(
        lam * loss_a + (
            1.0 - lam
        ) * loss_b,
    )

    actual = mixup_loss(
        logits=logits,
        labels_a=labels_a,
        labels_b=labels_b,
        num_classes=3,
        lam=lam,
    )

    _assert_allclose(
        actual,
        expected,
    )


def test_scalar_lambda_is_equivalent_to_batch_level_formula() -> None:
    """Verify that scalar lambda is equivalent to batch level formula."""
    logits = jnp.asarray(
        [
            [3.0, 0.1, -1.0],
            [0.2, 2.0, -0.5],
            [0.1, -0.3, 1.7],
            [-1.0, 0.5, 2.5],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [
            0,
            1,
            2,
            2,
        ],
        dtype=jnp.int32,
    )

    labels_b = jnp.asarray(
        [
            1,
            2,
            0,
            0,
        ],
        dtype=jnp.int32,
    )

    lam = jnp.asarray(
        0.73,
        dtype=jnp.float32,
    )

    actual = mixup_loss(
        logits=logits,
        labels_a=labels_a,
        labels_b=labels_b,
        num_classes=3,
        lam=lam,
    )

    loss_a_mean = hard_cross_entropy(
        logits=logits,
        labels=labels_a,
        num_classes=3,
    )

    loss_b_mean = hard_cross_entropy(
        logits=logits,
        labels=labels_b,
        num_classes=3,
    )

    expected_old_batch_level_formula = lam * loss_a_mean + (
        1.0 - lam
    ) * loss_b_mean

    _assert_allclose(
        actual,
        expected_old_batch_level_formula,
        atol=1e-6,
    )


def test_constant_vector_lambda_is_equivalent_to_batch_level_formula() -> None:
    """Verify that constant vector lambda is equivalent to batch level formula."""
    logits = jnp.asarray(
        [
            [3.0, 0.1, -1.0],
            [0.2, 2.0, -0.5],
            [0.1, -0.3, 1.7],
            [-1.0, 0.5, 2.5],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [
            0,
            1,
            2,
            2,
        ],
        dtype=jnp.int32,
    )

    labels_b = jnp.asarray(
        [
            1,
            2,
            0,
            0,
        ],
        dtype=jnp.int32,
    )

    lam = jnp.asarray(
        [
            0.73,
            0.73,
            0.73,
            0.73,
        ],
        dtype=jnp.float32,
    )

    actual = mixup_loss(
        logits=logits,
        labels_a=labels_a,
        labels_b=labels_b,
        num_classes=3,
        lam=lam,
    )

    loss_a_mean = hard_cross_entropy(
        logits=logits,
        labels=labels_a,
        num_classes=3,
    )

    loss_b_mean = hard_cross_entropy(
        logits=logits,
        labels=labels_b,
        num_classes=3,
    )

    expected_old_batch_level_formula = 0.73 * loss_a_mean + (
        1.0 - 0.73
    ) * loss_b_mean

    _assert_allclose(
        actual,
        expected_old_batch_level_formula,
        atol=1e-6,
    )


def test_mixup_loss_per_sample_lambda_is_not_equivalent_to_mean_lambda_when_losses_vary() -> None:
    """Verify that mixup loss per sample lambda is not equivalent to mean lambda when losses vary."""
    logits = jnp.asarray(
        [
            [5.0, 0.0, -2.0],
            [0.1, 1.0, 3.5],
            [2.2, -1.0, 0.3],
            [-0.5, 4.0, 0.2],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [
            0,
            2,
            1,
            1,
        ],
        dtype=jnp.int32,
    )

    labels_b = jnp.asarray(
        [
            2,
            0,
            0,
            2,
        ],
        dtype=jnp.int32,
    )

    lam = jnp.asarray(
        [
            0.95,
            0.20,
            0.75,
            0.05,
        ],
        dtype=jnp.float32,
    )

    correct = mixup_loss(
        logits=logits,
        labels_a=labels_a,
        labels_b=labels_b,
        num_classes=3,
        lam=lam,
    )

    loss_a_per_sample = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_a,
        num_classes=3,
    )

    loss_b_per_sample = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_b,
        num_classes=3,
    )

    manual_correct = jnp.mean(
        lam * loss_a_per_sample + (
            1.0 - lam
        ) * loss_b_per_sample,
    )

    loss_a_mean = hard_cross_entropy(
        logits=logits,
        labels=labels_a,
        num_classes=3,
    )

    loss_b_mean = hard_cross_entropy(
        logits=logits,
        labels=labels_b,
        num_classes=3,
    )

    wrong_mean_lambda_formula = jnp.mean(lam) * loss_a_mean + (
        1.0 - jnp.mean(lam)
    ) * loss_b_mean

    _assert_allclose(
        correct,
        manual_correct,
    )

    assert not np.isclose(
        float(correct),
        float(wrong_mean_lambda_formula),
        atol=1e-4,
    )