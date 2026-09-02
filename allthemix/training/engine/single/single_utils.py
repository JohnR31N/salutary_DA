"""Small host-side helpers for the single-device engine loop."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

AuxInfo = dict[str, jnp.ndarray]


def to_jax_aux_info(
    aux_info: dict[str, Any],
) -> AuxInfo:
    """Convert auxiliary batch data to JAX arrays."""
    return {
        key: jnp.asarray(value)
        for key, value in aux_info.items()
    }


def append_step_metrics(
    metric_lists: dict[str, list[float]],
    metrics: dict[str, jnp.ndarray],
) -> None:
    """Append scalar JAX metrics into Python lists."""
    for key, value in metrics.items():
        metric_lists.setdefault(
            key,
            [],
        ).append(
            float(value),
        )
