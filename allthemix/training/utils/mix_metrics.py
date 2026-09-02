from __future__ import annotations

import jax.numpy as jnp

from allthemix.methods.output import MixOutput


def unpack_mix_debug_inputs(
    mixer_output: MixOutput,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract mixed images, labels, lambda, and perm from a MixOutput."""
    mixed_images = mixer_output.images

    return (
        mixed_images,
        mixer_output.labels_a,
        mixer_output.labels_b,
        mixer_output.lam,
        mixer_output.perm,
    )


def _as_batch_lambda(
    lam: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """Convert scalar or per-sample lambda into a batch vector."""
    lam = jnp.asarray(
        lam,
    )

    if lam.ndim == 0:
        return jnp.full(
            (
                batch_size,
            ),
            lam,
        )

    if lam.ndim == 2 and lam.shape[-1] == 1:
        lam = lam.squeeze(
            axis=-1,
        )

    return lam.reshape(
        -1,
    )


def compute_mix_debug_metrics(
    images: jnp.ndarray,
    mixed_images: jnp.ndarray,
    labels_a: jnp.ndarray,
    labels_b: jnp.ndarray,
    lam: jnp.ndarray,
    perm: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute lightweight diagnostics for mix-based augmentations."""
    batch_lam = _as_batch_lambda(
        lam=lam,
        batch_size=images.shape[0],
    )
    changed_pixels = jnp.abs(
        mixed_images - images,
    ) > 1e-6
    changed_per_sample = jnp.mean(
        changed_pixels.astype(
            jnp.float32,
        ),
        axis=tuple(
            range(
                1,
                changed_pixels.ndim,
            )
        ),
    )
    applied_mask = batch_lam < 1.0 - 1e-6
    applied_float = applied_mask.astype(
        jnp.float32,
    )
    applied_count = jnp.sum(
        applied_float,
    )
    applied_denominator = jnp.maximum(
        applied_count,
        1.0,
    )
    same_label = (
        labels_a == labels_b
    ).astype(
        jnp.float32,
    )
    identity_pair = jnp.zeros_like(
        same_label,
        dtype=jnp.float32,
    )

    if perm is not None:
        identity_pair = (
            perm == jnp.arange(
                labels_a.shape[0],
            )
        ).astype(
            jnp.float32,
        )

    return {
        "mix_lam_mean": jnp.mean(
            batch_lam,
        ),
        "mix_lam_min": jnp.min(
            batch_lam,
        ),
        "mix_lam_max": jnp.max(
            batch_lam,
        ),
        "mix_lam_std": jnp.std(
            batch_lam,
        ),
        "mix_changed_ratio": jnp.mean(
            changed_pixels.astype(
                jnp.float32,
            )
        ),
        "mix_applied_lam_mean": jnp.sum(
            batch_lam * applied_float,
        )
        / applied_denominator,
        "mix_applied_changed_ratio": jnp.sum(
            changed_per_sample * applied_float,
        )
        / applied_denominator,
        "mix_apply_rate": jnp.mean(
            applied_float,
        ),
        "mix_same_label_rate": jnp.mean(
            same_label,
        ),
        "mix_applied_same_label_rate": jnp.sum(
            same_label * applied_float,
        )
        / applied_denominator,
        "mix_identity_pair_rate": jnp.mean(
            identity_pair,
        ),
        "mix_applied_identity_pair_rate": jnp.sum(
            identity_pair * applied_float,
        )
        / applied_denominator,
        "mix_image_mean": jnp.mean(
            mixed_images,
        ),
        "mix_image_std": jnp.std(
            mixed_images,
        ),
    }
