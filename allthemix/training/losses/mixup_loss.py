from __future__ import annotations

import jax.numpy as jnp

from allthemix.training.losses.cross_entropy import hard_cross_entropy_per_sample


def mixup_loss(
    logits: jnp.ndarray,
    labels_a: jnp.ndarray,
    labels_b: jnp.ndarray,
    num_classes: int,
    lam: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Mixup-style loss.

    Supports both:
        1. batch-level scalar lambda: shape ()
        2. per-sample lambda: shape (batch_size,)

    Correct formula:
        mean_i[
            lam_i * CE(logit_i, label_a_i)
            + (1 - lam_i) * CE(logit_i, label_b_i)
        ]

    For scalar lambda, this is mathematically equivalent to the common
    batch-level Mixup/CutMix formula.

    For per-sample lambda, this avoids the incorrect mean-lambda behavior.
    """
    loss_a = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_a,
        num_classes=num_classes,
    )

    loss_b = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels_b,
        num_classes=num_classes,
    )

    if lam.ndim == 0:
        mixed_loss = lam * loss_a + (  # Scalar-lambda mix of both CE terms.
            1.0 - lam
        ) * loss_b

    else:
        lam = lam.reshape(
            -1,
        )

        mixed_loss = lam * loss_a + (  # Per-sample lambda mix of both CE terms.
            1.0 - lam
        ) * loss_b

    return jnp.mean(  # Mean mixed loss over the batch.
        mixed_loss,
    )
