from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

FeatureHook = Callable[[jnp.ndarray, int], jnp.ndarray]


def apply_feature_hook(
    features: jnp.ndarray,
    feature_hook: FeatureHook | None,
    layer_index: int,
) -> jnp.ndarray:
    """Apply an optional feature hook at a numbered feature layer."""
    if feature_hook is None:
        return features

    return feature_hook(
        features,
        layer_index,
    )
