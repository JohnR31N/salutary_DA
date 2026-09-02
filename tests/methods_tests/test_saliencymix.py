from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.methods.saliencymix import saliencymix


def test_saliencymix_can_use_external_partner_batch() -> None:
    """Verify SaliencyMix can use externally paired images, labels, and maps."""
    images = jnp.ones(
        (
            4,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
        dtype=jnp.int32,
    )
    saliency_maps = jnp.arange(
        4 * 8 * 8,
        dtype=jnp.float32,
    ).reshape(
        4,
        8,
        8,
    )
    paired_labels = labels[::-1]

    _, labels_a, labels_b, _, _perm = saliencymix(
        rng=jax.random.PRNGKey(
            0,
        ),
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=10,
        alpha=1.0,
        prob=1.0,
        paired_images=images[::-1],
        paired_labels=paired_labels,
        paired_saliency_maps=saliency_maps[::-1],
    )

    np.testing.assert_array_equal(
        np.asarray(labels_a),
        np.asarray(labels),
    )
    np.testing.assert_array_equal(
        np.asarray(labels_b),
        np.asarray(paired_labels),
    )


def test_saliencymix_per_sample_returns_per_sample_lambdas() -> None:
    """Verify per-sample SaliencyMix builds one lambda per sample."""
    images = jnp.zeros(
        (
            4,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )
    paired_images = jnp.ones_like(
        images,
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
        dtype=jnp.int32,
    )
    paired_labels = labels[::-1]
    saliency_maps = jnp.zeros(
        (
            4,
            8,
            8,
        ),
        dtype=jnp.float32,
    )
    saliency_maps = saliency_maps.at[
        jnp.arange(4),
        jnp.asarray([1, 2, 5, 6]),
        jnp.asarray([1, 5, 2, 6]),
    ].set(
        1.0,
    )

    mixed_images, _, labels_b, lam, _perm = saliencymix(
        rng=jax.random.PRNGKey(
            2,
        ),
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=10,
        alpha=1.0,
        prob=1.0,
        per_sample=True,
        paired_images=paired_images,
        paired_labels=paired_labels,
        paired_saliency_maps=saliency_maps,
    )

    assert lam.shape == (4,)
    assert mixed_images.shape == images.shape
    assert jnp.any(
        mixed_images > 0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(labels_b),
        np.asarray(paired_labels),
    )


def test_saliencymix_lambda_matches_actual_clipped_patch_area() -> None:
    """Verify SaliencyMix lambda is recomputed after border clipping."""
    images = jnp.zeros(
        (
            4,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )
    paired_images = jnp.ones_like(
        images,
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
        dtype=jnp.int32,
    )
    saliency_maps = jnp.zeros(
        (
            4,
            8,
            8,
        ),
        dtype=jnp.float32,
    )
    saliency_maps = saliency_maps.at[
        jnp.arange(4),
        jnp.asarray([0, 0, 7, 7]),
        jnp.asarray([0, 7, 0, 7]),
    ].set(
        1.0,
    )

    mixed_images, _, _, lam, _perm = saliencymix(
        rng=jax.random.PRNGKey(
            4,
        ),
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=10,
        alpha=1.0,
        prob=1.0,
        per_sample=True,
        paired_images=paired_images,
        paired_labels=labels[::-1],
        paired_saliency_maps=saliency_maps,
    )

    changed_ratio = jnp.mean(
        mixed_images,
        axis=(
            1,
            2,
            3,
        ),
    )
    expected_lam = 1.0 - changed_ratio

    np.testing.assert_allclose(
        np.asarray(lam),
        np.asarray(expected_lam),
        atol=1e-6,
    )
