from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.utils.feature_hooks import FeatureHook, apply_feature_hook


class SimpleCNNBackbone(nn.Module):
    feature_dim: int = 128

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        feature_hook: FeatureHook | None = None,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Extract image features with a compact CNN backbone."""
        del training
        del sync_batch_stats

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            padding="SAME",
        )(x)
        x = nn.relu(x)
        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=1,
        )
        x = nn.avg_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            padding="SAME",
        )(x)
        x = nn.relu(x)
        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=2,
        )
        x = nn.avg_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
        )

        x = x.reshape((x.shape[0], -1))  # Flatten spatial features for dense head.

        x = nn.Dense(
            features=self.feature_dim,
        )(x)
        x = nn.relu(x)

        return x
