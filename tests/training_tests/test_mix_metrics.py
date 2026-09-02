from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from allthemix.methods.output import MixOutput
from allthemix.training.utils.mix_metrics import (
    compute_mix_debug_metrics,
    unpack_mix_debug_inputs,
)


def test_unpack_mix_debug_inputs_reads_named_fields() -> None:
    """Verify the unified MixOutput container feeds debug metrics."""
    mixed_images = jnp.ones(
        (
            2,
            4,
            4,
            1,
        )
    )
    labels_a = jnp.asarray(
        [
            0,
            1,
        ]
    )
    labels_b = jnp.asarray(
        [
            1,
            0,
        ]
    )
    lam = jnp.asarray(
        [
            0.25,
            0.75,
        ]
    )

    perm = jnp.asarray(
        [
            1,
            0,
        ]
    )
    unpacked = unpack_mix_debug_inputs(
        MixOutput(
            images=mixed_images,
            labels_a=labels_a,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
        )
    )

    assert unpacked[0] is mixed_images
    np.testing.assert_array_equal(
        np.asarray(
            unpacked[1],
        ),
        np.asarray(
            labels_a,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            unpacked[2],
        ),
        np.asarray(
            labels_b,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(
            unpacked[3],
        ),
        np.asarray(
            lam,
        ),
    )
    assert unpacked[4] is perm


def test_unpack_mix_debug_inputs_accepts_mix_output() -> None:
    """Verify named MixOutput containers can feed debug metrics."""
    mixed_images = jnp.ones(
        (
            2,
            4,
            4,
            1,
        )
    )
    labels_a = jnp.asarray(
        [
            0,
            1,
        ]
    )
    labels_b = jnp.asarray(
        [
            1,
            0,
        ]
    )
    lam = jnp.asarray(
        [
            0.5,
            0.5,
        ]
    )
    output = MixOutput(
        images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=jnp.asarray(
            [
                1,
                0,
            ]
        ),
    )

    unpacked = unpack_mix_debug_inputs(
        output,
    )

    assert unpacked[0] is mixed_images
    np.testing.assert_array_equal(
        np.asarray(
            unpacked[1],
        ),
        np.asarray(
            labels_a,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            unpacked[2],
        ),
        np.asarray(
            labels_b,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(
            unpacked[3],
        ),
        np.asarray(
            lam,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            unpacked[4],
        ),
        np.asarray(
            [
                1,
                0,
            ]
        ),
    )


def test_unpack_mix_debug_inputs_accepts_baseline_identity() -> None:
    """Verify identity MixOutput produces no-op mix diagnostics."""
    images = jnp.ones(
        (
            2,
            4,
            4,
            1,
        )
    )
    labels = jnp.asarray(
        [
            0,
            1,
        ]
    )

    mixed_images, labels_a, labels_b, lam, perm = unpack_mix_debug_inputs(
        MixOutput(
            images=images,
            labels_a=labels,
            labels_b=labels,
            lam=jnp.ones(
                labels.shape,
                dtype=jnp.float32,
            ),
            perm=jnp.arange(
                labels.shape[0],
            ),
        )
    )
    metrics = compute_mix_debug_metrics(
        images=images,
        mixed_images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
    )

    np.testing.assert_array_equal(
        np.asarray(
            labels_a,
        ),
        np.asarray(
            labels,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            labels_b,
        ),
        np.asarray(
            labels,
        ),
    )
    assert float(
        metrics["mix_lam_mean"],
    ) == 1.0
    assert float(
        metrics["mix_apply_rate"],
    ) == 0.0
    assert float(
        metrics["mix_applied_lam_mean"],
    ) == 0.0
    assert float(
        metrics["mix_applied_changed_ratio"],
    ) == 0.0
    assert float(
        metrics["mix_applied_same_label_rate"],
    ) == 0.0
    assert float(
        metrics["mix_changed_ratio"],
    ) == 0.0
    # The unified contract gives baseline an explicit identity perm, so the
    # unconditional identity-pair rate is truthfully 1.0 (all self-pairs).
    assert float(
        metrics["mix_identity_pair_rate"],
    ) == 1.0
    assert float(
        metrics["mix_applied_identity_pair_rate"],
    ) == 0.0


def test_compute_mix_debug_metrics_reports_applied_only_rates() -> None:
    """Verify applied-only diagnostics ignore inactive mixed samples."""
    images = jnp.zeros(
        (
            4,
            2,
            2,
            1,
        )
    )
    mixed_images = images.at[
        0,
        0,
        0,
        0,
    ].set(
        1.0,
    ).at[
        1,
        :,
        :,
        :,
    ].set(
        1.0,
    )
    labels_a = jnp.asarray(
        [
            0,
            1,
            2,
            3,
        ]
    )
    labels_b = jnp.asarray(
        [
            1,
            1,
            2,
            3,
        ]
    )
    lam = jnp.asarray(
        [
            0.75,
            0.50,
            1.00,
            1.00,
        ]
    )

    metrics = compute_mix_debug_metrics(
        images=images,
        mixed_images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=jnp.asarray(
            [
                1,
                1,
                2,
                0,
            ]
        ),
    )

    assert float(
        metrics["mix_apply_rate"],
    ) == 0.5
    assert float(
        metrics["mix_same_label_rate"],
    ) == 0.75
    assert float(
        metrics["mix_applied_same_label_rate"],
    ) == 0.5
    assert float(
        metrics["mix_applied_lam_mean"],
    ) == 0.625
    assert float(
        metrics["mix_applied_changed_ratio"],
    ) == 0.625
    assert float(
        metrics["mix_identity_pair_rate"],
    ) == 0.5
    assert float(
        metrics["mix_applied_identity_pair_rate"],
    ) == 0.5
