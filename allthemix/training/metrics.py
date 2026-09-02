from __future__ import annotations

import jax.numpy as jnp


def top_k_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
    """Compute top-k classification accuracy."""
    num_classes = logits.shape[-1]
    k = min(k, num_classes)

    top_k_predictions = jnp.argsort(  # Sort logits and keep the k largest classes.
        logits,
        axis=-1,
    )[:, -k:]

    labels = labels[:, None]

    correct = jnp.any(  # A sample is correct when its label appears in top-k.
        top_k_predictions == labels,
        axis=-1,
    )

    accuracy = jnp.mean(correct)  # Average correctness over the batch.

    return accuracy


def top_1_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Compute top-1 classification accuracy."""
    return top_k_accuracy(
        logits=logits,
        labels=labels,
        k=1,
    )


def top_5_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Compute top-5 classification accuracy."""
    return top_k_accuracy(
        logits=logits,
        labels=labels,
        k=5,
    )


def accuracy_to_error(
    accuracy: jnp.ndarray,
) -> jnp.ndarray:
    """Convert accuracy into classification error."""
    return 1.0 - accuracy  # Error is the complement of accuracy.
