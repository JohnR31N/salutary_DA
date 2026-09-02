from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from allthemix.visualize import mix_debug


def test_mixup_audit_reconstructs_linear_blend() -> None:
    """Verify MixUp debug checks the exact interpolation formula."""
    images = np.asarray(
        [
            np.zeros((4, 4, 1), dtype=np.float32),
            np.ones((4, 4, 1), dtype=np.float32),
        ]
    )
    labels = np.asarray([0, 1], dtype=np.int32)
    perm = np.asarray([1, 0], dtype=np.int32)
    lam = np.asarray([0.25, 0.25], dtype=np.float32)
    mixed = lam.reshape(2, 1, 1, 1) * images + (
        1.0 - lam.reshape(2, 1, 1, 1)
    ) * images[perm]

    row = mix_debug._audit_mixup_sample(
        images=images,
        mixed_images=mixed,
        labels=labels,
        labels_b=labels[perm],
        lam=lam,
        perm=perm,
        sample_index=0,
        threshold=1e-6,
    )

    assert row["label_b_ok"] is True
    assert row["lam_abs_diff"] == 0.0
    assert row["pixel_max_abs_error"] == 0.0


def test_hard_paste_audit_checks_area_and_sources() -> None:
    """Verify hard-paste debug checks lambda, source B, and source A regions."""
    images = np.asarray(
        [
            np.zeros((4, 4, 1), dtype=np.float32),
            np.ones((4, 4, 1), dtype=np.float32),
        ]
    )
    labels = np.asarray([0, 1], dtype=np.int32)
    perm = np.asarray([1, 0], dtype=np.int32)
    mixed = images.copy()
    mixed[0, 1:3, 1:3, :] = images[1, 1:3, 1:3, :]
    lam = np.asarray([0.75, 0.75], dtype=np.float32)

    row = mix_debug._audit_hard_paste_sample(
        images=images,
        mixed_images=mixed,
        labels=labels,
        labels_b=labels[perm],
        lam=lam,
        perm=perm,
        sample_index=0,
        threshold=1e-6,
        audit_type="hard_paste",
    )

    assert row["label_b_ok"] is True
    assert row["lam_abs_diff"] == 0.0
    assert row["inside_b_max_abs_error"] == 0.0
    assert row["outside_a_max_abs_error"] == 0.0


def test_hard_paste_audit_can_use_sampled_lambda() -> None:
    """Verify Torchbearer-style debug accepts sampled lambda."""
    images = np.asarray(
        [
            np.zeros((4, 4, 1), dtype=np.float32),
            np.ones((4, 4, 1), dtype=np.float32),
        ]
    )
    labels = np.asarray([0, 1], dtype=np.int32)
    perm = np.asarray([1, 0], dtype=np.int32)
    mixed = images.copy()
    mixed[0, 0:2, 0:2, :] = images[1, 0:2, 0:2, :]
    lam = np.asarray([0.4, 0.4], dtype=np.float32)

    row = mix_debug._audit_hard_paste_sample(
        images=images,
        mixed_images=mixed,
        labels=labels,
        labels_b=labels[perm],
        lam=lam,
        perm=perm,
        sample_index=0,
        threshold=1e-6,
        audit_type="hard_paste",
        lambda_mode="sampled",
    )

    assert row["label_b_ok"] is True
    assert row["expected_lam"] == row["lam"]
    assert row["lam_abs_diff"] == 0.0
    assert row["pixel_max_abs_error"] == 0.0


def test_resizemix_audit_reconstructs_nearest_resized_patch() -> None:
    """Verify ResizeMix debug reconstructs the resized source patch."""
    source_a = np.zeros((4, 4, 1), dtype=np.float32)
    source_b = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    images = np.stack([source_a, source_b], axis=0)
    labels = np.asarray([0, 1], dtype=np.int32)
    perm = np.asarray([1, 0], dtype=np.int32)
    mixed = images.copy()
    resized_source = mix_debug._resize_source_to_box_nearest_np(
        source_image=source_b,
        x1=1,
        y1=1,
        x2=3,
        y2=3,
    )
    mixed[0, 1:3, 1:3, :] = resized_source[1:3, 1:3, :]
    lam = np.asarray([0.75, 0.75], dtype=np.float32)

    row = mix_debug._audit_resizemix_sample(
        images=images,
        mixed_images=mixed,
        labels=labels,
        labels_b=labels[perm],
        lam=lam,
        perm=perm,
        sample_index=0,
        threshold=1e-6,
    )

    assert row["label_b_ok"] is True
    assert row["lam_abs_diff"] == 0.0
    assert row["inside_b_max_abs_error"] == 0.0
    assert row["outside_a_max_abs_error"] == 0.0
    assert row["pixel_max_abs_error"] == 0.0


def test_saliency_fallback_allows_debug_without_cache(tmp_path) -> None:
    """Verify SaliencyMix debug can inject fallback saliency maps."""
    args = argparse.Namespace(
        dataset="tiny_imagenet",
        saliency_dir=str(tmp_path),
        saliency_fallback="sr",
        guidedmixup_blur_kernel=7,
    )

    assert mix_debug._debug_skip_reason("saliencymix", args) is None

    images = jnp.ones(
        (
            2,
            8,
            8,
            3,
        ),
        dtype=jnp.float32,
    )
    aux_info, source = mix_debug._ensure_debug_aux_info(
        method="saliencymix",
        images=images,
        aux_info={},
        args=args,
    )

    assert source == "debug_sr_fallback"
    assert "saliency_maps" in aux_info
    assert aux_info["saliency_maps"].shape == (2, 8, 8, 1)
