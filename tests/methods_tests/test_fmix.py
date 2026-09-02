from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.methods.fmix import _make_fmix_mask, fmix


def test_make_fmix_mask_is_binary_and_close_to_lambda() -> None:
    """Verify that FMix masks are binary and close to the sampled lambda."""
    target_lam = jnp.asarray(
        0.375,
        dtype=jnp.float32,
    )

    mask, returned_lam = _make_fmix_mask(
        rng=jax.random.PRNGKey(0),
        height=32,
        width=32,
        lam=target_lam,
        decay_power=3.0,
        dtype=jnp.float32,
    )

    assert mask.shape == (32, 32, 1)
    assert bool(
        jnp.all(
            (mask == 0.0)
            | (mask == 1.0),
        )
    )

    np.testing.assert_allclose(
        np.asarray(returned_lam),
        np.asarray(target_lam),
        atol=1e-6,
        rtol=1e-6,
    )

    assert abs(
        float(jnp.mean(mask))
        - float(target_lam)
    ) <= 1.0 / (32 * 32)


def test_fmix_prob_zero_returns_original_batch() -> None:
    """Verify that FMix can skip mixing with prob zero."""
    images = jnp.arange(
        4 * 8 * 8 * 3,
        dtype=jnp.float32,
    ).reshape(
        4,
        8,
        8,
        3,
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
        dtype=jnp.int32,
    )

    mixed_images, labels_a, labels_b, lam, _perm = fmix(
        rng=jax.random.PRNGKey(1),
        images=images,
        labels=labels,
        num_classes=10,
        alpha=1.0,
        decay_power=3.0,
        prob=0.0,
    )

    np.testing.assert_array_equal(
        np.asarray(mixed_images),
        np.asarray(images),
    )
    np.testing.assert_array_equal(
        np.asarray(labels_a),
        np.asarray(labels),
    )
    np.testing.assert_array_equal(
        np.asarray(labels_b),
        np.asarray(labels),
    )
    np.testing.assert_allclose(
        np.asarray(lam),
        np.asarray(1.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_fmix_uses_sampled_lambda_for_loss_weight() -> None:
    """Verify that FMix returns the sampled lambda like the reference code."""
    rng = jax.random.PRNGKey(2)
    images = jnp.arange(
        8 * 16 * 16 * 3,
        dtype=jnp.float32,
    ).reshape(
        8,
        16,
        16,
        3,
    )
    labels = jnp.arange(
        8,
        dtype=jnp.int32,
    )

    rng_lam, _, rng_mask, _ = jax.random.split(
        rng,
        4,
    )
    sampled_lam = jax.random.beta(
        rng_lam,
        1.0,
        1.0,
        shape=(),
    )
    _, expected_lam = _make_fmix_mask(
        rng=rng_mask,
        height=16,
        width=16,
        lam=sampled_lam,
        decay_power=3.0,
        dtype=images.dtype,
    )

    _, labels_a, labels_b, lam, _perm = fmix(
        rng=rng,
        images=images,
        labels=labels,
        num_classes=10,
        alpha=1.0,
        decay_power=3.0,
        prob=1.0,
    )

    np.testing.assert_array_equal(
        np.asarray(labels_a),
        np.asarray(labels),
    )
    assert labels_b.shape == labels.shape
    np.testing.assert_allclose(
        np.asarray(lam),
        np.asarray(expected_lam),
        atol=1e-6,
        rtol=1e-6,
    )


def test_fmix_per_sample_returns_per_sample_lambdas() -> None:
    """Verify that per-sample FMix samples one mask area per example."""
    images = jnp.arange(
        8 * 16 * 16 * 3,
        dtype=jnp.float32,
    ).reshape(
        8,
        16,
        16,
        3,
    )
    labels = jnp.arange(
        8,
        dtype=jnp.int32,
    )

    mixed_images, labels_a, labels_b, lam, _perm = fmix(
        rng=jax.random.PRNGKey(3),
        images=images,
        labels=labels,
        num_classes=10,
        alpha=1.0,
        decay_power=3.0,
        prob=1.0,
        per_sample=True,
    )

    assert mixed_images.shape == images.shape
    assert labels_a.shape == labels.shape
    assert labels_b.shape == labels.shape
    assert lam.shape == labels.shape
    assert bool(
        jnp.all(
            lam > 0.0,
        )
    )
    assert bool(
        jnp.all(
            lam < 1.0,
        )
    )
    assert float(
        jnp.std(
            lam,
        )
    ) > 0.0


def test_fmix_no_repeat_avoids_identity_partners() -> None:
    """Verify that FMix no-repeat pairing avoids self-pairs."""
    images = jnp.arange(
        8 * 16 * 16 * 3,
        dtype=jnp.float32,
    ).reshape(
        8,
        16,
        16,
        3,
    )
    labels = jnp.arange(
        8,
        dtype=jnp.int32,
    )

    _, labels_a, labels_b, _, _perm = fmix(
        rng=jax.random.PRNGKey(33),
        images=images,
        labels=labels,
        num_classes=10,
        alpha=1.0,
        decay_power=3.0,
        prob=1.0,
        no_repeat=True,
    )

    np.testing.assert_array_equal(
        np.asarray(labels_a),
        np.asarray(labels),
    )
    assert bool(
        jnp.all(
            labels_b != labels,
        )
    )


def test_fmix_accepts_external_paired_batch() -> None:
    """Verify that FMix can use externally paired samples."""
    images = jnp.arange(
        4 * 8 * 8 * 3,
        dtype=jnp.float32,
    ).reshape(
        4,
        8,
        8,
        3,
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
        dtype=jnp.int32,
    )
    paired_images = images[::-1]
    paired_labels = labels[::-1]

    _, labels_a, labels_b, _, _perm = fmix(
        rng=jax.random.PRNGKey(4),
        images=images,
        labels=labels,
        num_classes=10,
        alpha=1.0,
        decay_power=3.0,
        prob=1.0,
        paired_images=paired_images,
        paired_labels=paired_labels,
    )

    np.testing.assert_array_equal(
        np.asarray(labels_a),
        np.asarray(labels),
    )
    np.testing.assert_array_equal(
        np.asarray(labels_b),
        np.asarray(paired_labels),
    )


def test_selector_passes_fmix_per_sample_argument() -> None:
    """Verify that selector exposes the per-sample FMix option."""
    from allthemix.methods.selector import get_mixer

    images = jnp.arange(
        4 * 8 * 8 * 3,
        dtype=jnp.float32,
    ).reshape(
        4,
        8,
        8,
        3,
    )
    labels = jnp.arange(
        4,
        dtype=jnp.int32,
    )
    rng = jax.random.PRNGKey(
        4,
    )

    mixer = get_mixer(
        name="fmix",
        num_classes=10,
        fmix_alpha=1.0,
        fmix_decay=3.0,
        fmix_prob=1.0,
        fmix_per_sample=True,
    )

    selector_output = mixer(
        rng,
        images,
        labels,
        None,
    )

    direct_output = fmix(
        rng=rng,
        images=images,
        labels=labels,
        num_classes=10,
        alpha=1.0,
        decay_power=3.0,
        prob=1.0,
        per_sample=True,
        no_repeat=False,
    )

    for actual, expected in zip(
        selector_output,
        direct_output,
    ):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            atol=1e-6,
            rtol=1e-6,
        )


def test_selector_passes_fmix_external_paired_batch() -> None:
    """Verify that selector forwards externally paired FMix samples."""
    images = jnp.arange(
        4 * 8 * 8 * 3,
        dtype=jnp.float32,
    ).reshape(
        4,
        8,
        8,
        3,
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
        dtype=jnp.int32,
    )
    paired_images = images[::-1]
    paired_labels = labels[::-1]

    from allthemix.methods.selector import get_mixer

    mixer = get_mixer(
        name="fmix",
        num_classes=10,
        fmix_alpha=1.0,
        fmix_decay=3.0,
        fmix_prob=1.0,
    )

    _, _, labels_b, _, _perm = mixer(
        jax.random.PRNGKey(5),
        images,
        labels,
        {
            "paired_images": paired_images,
            "paired_labels": paired_labels,
        },
    )

    np.testing.assert_array_equal(
        np.asarray(labels_b),
        np.asarray(paired_labels),
    )
