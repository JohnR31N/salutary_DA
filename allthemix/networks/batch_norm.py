from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def batch_norm(
    x: jnp.ndarray,
    training: bool = True,
    sync_batch_stats: bool = False,
) -> jnp.ndarray:
    """Apply BatchNorm with optional cross-replica training statistics."""
    axis_name = "batch" if training and sync_batch_stats else None

    return nn.BatchNorm(
        use_running_average=not training,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=axis_name,
    )(x)
