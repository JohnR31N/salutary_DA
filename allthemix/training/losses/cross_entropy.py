from __future__ import annotations

import jax
import jax.numpy as jnp


def soft_cross_entropy_per_sample(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-sample cross entropy for soft labels."""
    log_probs = jax.nn.log_softmax(  # Convert logits to log probabilities.
        logits,
        axis=-1,
    )

    losses = -jnp.sum(  # Sum -label * log(probability) over classes.
        labels * log_probs,
        axis=-1,
    )

    return losses


def soft_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Average soft-label cross entropy over the batch."""
    losses = soft_cross_entropy_per_sample(
        logits=logits,
        labels=labels,
    )

    return jnp.mean(  # Batch mean cross entropy.
        losses,
    )


def hard_cross_entropy_per_sample(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """Compute per-sample cross entropy for integer labels."""
    one_hot_labels = jax.nn.one_hot(  # Convert hard labels to one-hot labels.
        labels,
        num_classes,
    )

    return soft_cross_entropy_per_sample(
        logits=logits,
        labels=one_hot_labels,
    )


def hard_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """Average hard-label cross entropy over the batch."""
    losses = hard_cross_entropy_per_sample(
        logits=logits,
        labels=labels,
        num_classes=num_classes,
    )

    return jnp.mean(  # Batch mean cross entropy.
        losses,
    )


def cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """Dispatch to hard-label or soft-label cross entropy."""
    if labels.ndim == 1:
        return hard_cross_entropy(
            logits=logits,
            labels=labels,
            num_classes=num_classes,
        )

    if labels.ndim == 2:
        return soft_cross_entropy(
            logits=logits,
            labels=labels,
        )

    raise ValueError(
        f"Unsupported labels shape: {labels.shape}"
    )
