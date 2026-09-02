from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from allthemix.training.losses.cross_entropy import hard_cross_entropy_per_sample
from allthemix.training.losses.sumix_loss import compute_sumix_lambda, sumix_loss


def test_sumix_loss_default_gamma_matches_official_cifar_config() -> None:
    """Verify that sumix loss default gamma matches official cifar config."""
    logits_original = jnp.asarray(
        [
            [2.0, 0.2, -0.5],
            [0.1, 1.8, -0.2],
            [-0.3, 0.4, 2.1],
            [0.3, -0.2, 1.2],
        ],
        dtype=jnp.float32,
    )

    logits_mixed = jnp.asarray(
        [
            [1.2, 0.8, -0.3],
            [0.2, 1.1, 0.4],
            [0.1, 0.7, 1.5],
            [0.5, 0.1, 1.0],
        ],
        dtype=jnp.float32,
    )

    uncertainty_original = jnp.asarray(
        [
            [0.8, 0.4, 0.2],
            [0.3, 0.9, 0.2],
            [0.2, 0.3, 0.9],
            [0.4, 0.2, 0.8],
        ],
        dtype=jnp.float32,
    )

    uncertainty_mixed = jnp.asarray(
        [
            [0.5, 0.6, 0.3],
            [0.4, 0.7, 0.5],
            [0.3, 0.5, 0.8],
            [0.6, 0.3, 0.7],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [0, 1, 2, 2],
        dtype=jnp.int32,
    )

    perm = jnp.asarray(
        [1, 2, 3, 0],
        dtype=jnp.int32,
    )

    labels_b = labels_a[perm]

    kwargs = {
        "logits_original": logits_original,
        "logits_mixed": logits_mixed,
        "uncertainty_original": uncertainty_original,
        "uncertainty_mixed": uncertainty_mixed,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "area_lam": jnp.asarray([0.65, 0.35, 0.75, 0.25], dtype=jnp.float32),
        "perm": perm,
        "num_classes": 3,
    }

    default_loss, _ = sumix_loss(**kwargs)
    explicit_official_loss, _ = sumix_loss(
        **kwargs,
        gamma=0.5,
    )
    old_default_loss, _ = sumix_loss(
        **kwargs,
        gamma=0.1,
    )

    np.testing.assert_allclose(
        np.asarray(default_loss),
        np.asarray(explicit_official_loss),
        atol=1e-6,
        rtol=1e-6,
    )

    assert not np.isclose(
        float(default_loss),
        float(old_default_loss),
        atol=1e-4,
    )


def test_sumix_classification_loss_uses_official_batch_mean_ce() -> None:
    """Verify that SUMix follows the official batch-mean CE weighting."""
    logits_original = jnp.asarray(
        [
            [3.0, 0.1, -0.5],
            [0.2, 2.6, -0.4],
            [-0.2, 0.4, 2.8],
            [1.0, -0.1, 0.3],
        ],
        dtype=jnp.float32,
    )

    logits_mixed = jnp.asarray(
        [
            [0.8, 1.5, -0.2],
            [0.3, 0.4, 2.0],
            [1.8, 0.2, 0.1],
            [0.1, 1.6, 0.5],
        ],
        dtype=jnp.float32,
    )

    uncertainty_original = jnp.asarray(
        [
            [0.8, 0.4, 0.2],
            [0.3, 0.9, 0.2],
            [0.2, 0.3, 0.9],
            [0.4, 0.2, 0.8],
        ],
        dtype=jnp.float32,
    )

    uncertainty_mixed = jnp.asarray(
        [
            [0.5, 0.6, 0.3],
            [0.4, 0.7, 0.5],
            [0.3, 0.5, 0.8],
            [0.6, 0.3, 0.7],
        ],
        dtype=jnp.float32,
    )

    labels_a = jnp.asarray(
        [0, 1, 2, 0],
        dtype=jnp.int32,
    )

    perm = jnp.asarray(
        [1, 2, 3, 0],
        dtype=jnp.int32,
    )

    labels_b = labels_a[perm]
    area_lam = jnp.asarray(
        [0.2, 0.8, 0.3, 0.7],
        dtype=jnp.float32,
    )

    _, metrics = sumix_loss(
        logits_original=logits_original,
        logits_mixed=logits_mixed,
        uncertainty_original=uncertainty_original,
        uncertainty_mixed=uncertainty_mixed,
        labels_a=labels_a,
        labels_b=labels_b,
        area_lam=area_lam,
        perm=perm,
        num_classes=3,
        gamma=0.0,
    )

    lam_a, lam_b, _, _ = compute_sumix_lambda(
        logits_original=logits_original,
        logits_mixed=logits_mixed,
        uncertainty_original=uncertainty_original,
        uncertainty_mixed=uncertainty_mixed,
        labels_a=labels_a,
        labels_b=labels_b,
        area_lam=area_lam,
        perm=perm,
    )

    ce_a = hard_cross_entropy_per_sample(
        logits=logits_mixed,
        labels=labels_a,
        num_classes=3,
    )

    ce_b = hard_cross_entropy_per_sample(
        logits=logits_mixed,
        labels=labels_b,
        num_classes=3,
    )

    expected_official = (
        jnp.mean(ce_a) * jnp.mean(lam_a)
        + jnp.mean(ce_b) * jnp.mean(lam_b)
    )

    per_sample_weighted = jnp.mean(
        lam_a * ce_a
        + lam_b * ce_b,
    )

    np.testing.assert_allclose(
        np.asarray(metrics["classification_loss"]),
        np.asarray(expected_official),
        atol=1e-6,
        rtol=1e-6,
    )

    assert not np.isclose(
        float(expected_official),
        float(per_sample_weighted),
        atol=1e-4,
    )


def test_sumix_semantic_scale_changes_adaptive_lambda() -> None:
    """Verify that semantic scale ablations affect SUMix adaptive lambda."""
    logits_original = jnp.asarray(
        [
            [2.0, 0.2, -0.5],
            [0.1, 1.8, -0.2],
            [-0.3, 0.4, 2.1],
            [0.3, -0.2, 1.2],
        ],
        dtype=jnp.float32,
    )

    logits_mixed = jnp.asarray(
        [
            [1.2, 0.8, -0.3],
            [0.2, 1.1, 0.4],
            [0.1, 0.7, 1.5],
            [0.5, 0.1, 1.0],
        ],
        dtype=jnp.float32,
    )

    uncertainty_original = jnp.ones_like(
        logits_original,
    ) * 0.5

    uncertainty_mixed = jnp.ones_like(
        logits_mixed,
    ) * 0.5

    labels_a = jnp.asarray(
        [0, 1, 2, 2],
        dtype=jnp.int32,
    )

    perm = jnp.asarray(
        [1, 2, 3, 0],
        dtype=jnp.int32,
    )

    labels_b = labels_a[perm]
    area_lam = jnp.asarray(
        [0.65, 0.35, 0.75, 0.25],
        dtype=jnp.float32,
    )

    official_lam_a, _, _, _ = compute_sumix_lambda(
        logits_original=logits_original,
        logits_mixed=logits_mixed,
        uncertainty_original=uncertainty_original,
        uncertainty_mixed=uncertainty_mixed,
        labels_a=labels_a,
        labels_b=labels_b,
        area_lam=area_lam,
        perm=perm,
        semantic_scale=-1.0,
    )

    smaller_scale_lam_a, _, _, _ = compute_sumix_lambda(
        logits_original=logits_original,
        logits_mixed=logits_mixed,
        uncertainty_original=uncertainty_original,
        uncertainty_mixed=uncertainty_mixed,
        labels_a=labels_a,
        labels_b=labels_b,
        area_lam=area_lam,
        perm=perm,
        semantic_scale=1.0,
    )

    assert not np.allclose(
        np.asarray(official_lam_a),
        np.asarray(smaller_scale_lam_a),
        atol=1e-5,
    )
