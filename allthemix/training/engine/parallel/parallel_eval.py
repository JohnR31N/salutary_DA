from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from allthemix.training.engine.state import TrainStateWithBatchStats
from allthemix.training.losses.cross_entropy import (
    hard_cross_entropy_per_sample,
    soft_cross_entropy_per_sample,
)


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(4,),
)
def parallel_eval_step(
    state: TrainStateWithBatchStats,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    valid_mask: jnp.ndarray,
    num_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one PMAP evaluation step across local devices."""
    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }

    logits = state.apply_fn(
        variables,
        images,
        training=False,
        mutable=False,
    )

    if labels.ndim == 1:
        per_sample_loss = hard_cross_entropy_per_sample(
            logits=logits,
            labels=labels,
            num_classes=num_classes,
        )
        hard_labels = labels
    elif labels.ndim == 2:
        per_sample_loss = soft_cross_entropy_per_sample(
            logits=logits,
            labels=labels,
        )
        hard_labels = jnp.argmax(  # Convert soft labels to class ids for accuracy.
            labels,
            axis=-1,
        )
    else:
        raise ValueError(
            f"Unsupported labels shape: {labels.shape}"
        )

    top1_predictions = jnp.argmax(  # Select the largest-logit class.
        logits,
        axis=-1,
    )

    top5_predictions = jnp.argsort(  # Keep the five largest-logit classes.
        logits,
        axis=-1,
    )[:, -min(5, logits.shape[-1]) :]

    top1_correct = top1_predictions == hard_labels  # Mark top-1 correct samples.
    top5_correct = jnp.any(  # Mark top-5 correct samples.
        top5_predictions == hard_labels[:, None],
        axis=-1,
    )

    valid_mask = valid_mask.astype(
        per_sample_loss.dtype,
    )

    loss_sum = jnp.sum(  # Sum loss over real, non-padded samples.
        per_sample_loss * valid_mask,
    )

    top1_correct_sum = jnp.sum(  # Count valid top-1 correct samples.
        top1_correct.astype(
            per_sample_loss.dtype,
        )
        * valid_mask,
    )

    top5_correct_sum = jnp.sum(  # Count valid top-5 correct samples.
        top5_correct.astype(
            per_sample_loss.dtype,
        )
        * valid_mask,
    )

    valid_count = jnp.sum(valid_mask)  # Count real samples in this shard.

    loss_sum = jax.lax.psum(  # Aggregate loss sum across devices.
        loss_sum,
        axis_name="batch",
    )

    top1_correct_sum = jax.lax.psum(  # Aggregate top-1 count across devices.
        top1_correct_sum,
        axis_name="batch",
    )

    top5_correct_sum = jax.lax.psum(  # Aggregate top-5 count across devices.
        top5_correct_sum,
        axis_name="batch",
    )

    valid_count = jax.lax.psum(  # Aggregate sample count across devices.
        valid_count,
        axis_name="batch",
    )

    return loss_sum, top1_correct_sum, top5_correct_sum, valid_count
