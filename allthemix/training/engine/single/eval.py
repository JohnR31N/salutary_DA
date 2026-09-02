from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from allthemix.training.engine.state import TrainStateWithBatchStats
from allthemix.training.losses.cross_entropy import cross_entropy
from allthemix.training.metrics import accuracy_to_error, top_1_accuracy, top_5_accuracy


@partial(
    jax.jit,
    static_argnames=("num_classes",),
)
def eval_step(
    state: TrainStateWithBatchStats,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one JIT-compiled single-device evaluation step."""
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

    loss = cross_entropy(
        logits=logits,
        labels=labels,
        num_classes=num_classes,
    )

    top1_acc = top_1_accuracy(
        logits=logits,
        labels=labels,
    )

    top5_acc = top_5_accuracy(
        logits=logits,
        labels=labels,
    )

    top1_error = accuracy_to_error(top1_acc)  # Error is one minus top-1 accuracy.
    top5_error = accuracy_to_error(top5_acc)  # Error is one minus top-5 accuracy.

    return loss, top1_acc, top5_acc, top1_error, top5_error
