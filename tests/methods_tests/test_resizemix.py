from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.methods.resizemix import _resize_source_to_box_nearest, resizemix


def test_resize_source_to_box_nearest_places_source_inside_box() -> None:
    """Verify that ResizeMix helper only fills pixels inside the paste box."""
    source_images = jnp.arange(
        2 * 4 * 4 * 1,
        dtype=jnp.float32,
    ).reshape(
        2,
        4,
        4,
        1,
    )

    resized_source, box_mask = _resize_source_to_box_nearest(
        source_images=source_images,
        x1=jnp.asarray(1),
        y1=jnp.asarray(1),
        x2=jnp.asarray(3),
        y2=jnp.asarray(3),
    )

    assert resized_source.shape == source_images.shape
    assert box_mask.shape == (2, 4, 4, 1)

    expected_mask = np.zeros(
        (
            2,
            4,
            4,
            1,
        ),
        dtype=bool,
    )
    expected_mask[:, 1:3, 1:3, :] = True

    np.testing.assert_array_equal(
        np.asarray(box_mask),
        expected_mask,
    )


def test_resizemix_prob_zero_returns_original_batch() -> None:
    """Verify that ResizeMix can skip mixing with prob zero."""
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

    mixed_images, labels_a, labels_b, lam, _perm = resizemix(
        rng=jax.random.PRNGKey(0),
        images=images,
        labels=labels,
        num_classes=10,
        scope_min=0.1,
        scope_max=0.8,
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


def test_resizemix_can_use_external_partner_batch() -> None:
    """Verify that ResizeMix can use partners supplied by distributed training."""
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

    _, labels_a, labels_b, _, _perm = resizemix(
        rng=jax.random.PRNGKey(0),
        images=images,
        labels=labels,
        num_classes=10,
        scope_min=0.1,
        scope_max=0.8,
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


def test_resizemix_lambda_matches_actual_paste_area() -> None:
    """Verify that ResizeMix lambda equals the retained target area."""
    rng = jax.random.PRNGKey(
        5,
    )
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

    rng_tao, _, rng_cx, rng_cy, _ = jax.random.split(
        rng,
        5,
    )
    tao = jax.random.uniform(
        rng_tao,
        shape=(),
        minval=0.1,
        maxval=0.8,
    )

    cut_width = jnp.maximum(
        (16 * tao).astype(jnp.int32),
        1,
    )
    cut_height = jnp.maximum(
        (16 * tao).astype(jnp.int32),
        1,
    )
    center_x = jax.random.randint(
        rng_cx,
        shape=(),
        minval=0,
        maxval=16,
    )
    center_y = jax.random.randint(
        rng_cy,
        shape=(),
        minval=0,
        maxval=16,
    )
    x1 = jnp.clip(
        center_x - cut_width // 2,
        0,
        16,
    )
    x2 = jnp.clip(
        center_x + cut_width // 2,
        0,
        16,
    )
    y1 = jnp.clip(
        center_y - cut_height // 2,
        0,
        16,
    )
    y2 = jnp.clip(
        center_y + cut_height // 2,
        0,
        16,
    )
    x2 = jnp.minimum(
        jnp.maximum(
            x2,
            x1 + 1,
        ),
        16,
    )
    y2 = jnp.minimum(
        jnp.maximum(
            y2,
            y1 + 1,
        ),
        16,
    )

    expected_lam = 1.0 - (
        ((x2 - x1) * (y2 - y1)).astype(jnp.float32)
        / float(16 * 16)
    )

    _, labels_a, labels_b, lam, _perm = resizemix(
        rng=rng,
        images=images,
        labels=labels,
        num_classes=10,
        scope_min=0.1,
        scope_max=0.8,
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


def test_resizemix_per_sample_returns_per_sample_lambdas() -> None:
    """Verify that ResizeMix can sample one clipped box per sample."""
    images = jnp.zeros((8, 32, 32, 1), dtype=jnp.float32)
    labels = jnp.arange(8, dtype=jnp.int32)
    paired_images = jnp.ones_like(images)
    paired_labels = labels[::-1]

    mixed_images, _, _, lam, _perm = resizemix(
        rng=jax.random.PRNGKey(7),
        images=images,
        labels=labels,
        num_classes=10,
        scope_min=0.1,
        scope_max=0.8,
        prob=1.0,
        per_sample=True,
        paired_images=paired_images,
        paired_labels=paired_labels,
    )

    changed_ratios = jnp.mean(
        mixed_images[..., 0] > 0.5,
        axis=(
            1,
            2,
        ),
    )

    assert lam.shape == (8,)
    np.testing.assert_allclose(
        np.asarray(lam),
        np.asarray(1.0 - changed_ratios),
        atol=1e-6,
        rtol=1e-6,
    )
    assert float(jnp.std(changed_ratios)) > 0.0


def test_resizemix_clipped_box_lambda_matches_pixel_fraction() -> None:
    """Verify lambda against pixel counts with an oracle independent of the box math."""
    batch_size = 4
    image_size = 16

    images = jnp.zeros(
        (
            batch_size,
            image_size,
            image_size,
            3,
        ),
        dtype=jnp.float32,
    )
    paired_images = jnp.ones(
        (
            batch_size,
            image_size,
            image_size,
            3,
        ),
        dtype=jnp.float32,
    )
    labels = jnp.zeros(
        (batch_size,),
        dtype=jnp.int32,
    )
    paired_labels = jnp.ones(
        (batch_size,),
        dtype=jnp.int32,
    )

    saw_clipped_box = False

    for seed in range(24):
        mixed_images, _, _, lam, _perm = resizemix(
            rng=jax.random.PRNGKey(
                seed,
            ),
            images=images,
            labels=labels,
            num_classes=2,
            scope_min=0.1,
            scope_max=0.8,
            prob=1.0,
            paired_images=paired_images,
            paired_labels=paired_labels,
        )

        mixed_np = np.asarray(
            mixed_images,
        )
        lam_value = float(
            np.asarray(
                lam,
            )
        )

        for sample_index in range(batch_size):
            pasted_fraction = float(
                mixed_np[sample_index, :, :, 0].mean(),
            )
            np.testing.assert_allclose(
                1.0 - lam_value,
                pasted_fraction,
                atol=1e-6,
                rtol=0.0,
            )

        box_rows, box_cols = np.where(
            mixed_np[0, :, :, 0] > 0.5,
        )
        box_height = box_rows.max() - box_rows.min() + 1
        box_width = box_cols.max() - box_cols.min() + 1

        if box_height != box_width:  # Unequal sides only happen when the box was clipped.
            saw_clipped_box = True

    assert saw_clipped_box
