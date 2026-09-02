from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.batch_norm import batch_norm
from allthemix.networks.utils.feature_hooks import FeatureHook, apply_feature_hook


class WideBasicBlock(nn.Module):
    features: int
    stride: int = 1
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Apply one WideResNet residual block."""
        residual = x

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = nn.relu(x)

        if residual.shape[-1] != self.features or self.stride != 1:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = nn.relu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(
                rate=self.dropout_rate,
            )(
                x,
                deterministic=not training,
            )

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        x = x + residual  # Add residual shortcut to the transformed path.

        return x


class WideResNetBackbone(nn.Module):
    depth: int
    widen_factor: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        feature_hook: FeatureHook | None = None,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Extract image features with a WideResNet backbone."""
        if (self.depth - 4) % 6 != 0:
            raise ValueError(
                f"WideResNet depth should satisfy depth = 6n + 4, "
                f"but got depth={self.depth}."
            )

        num_blocks_per_group = (self.depth - 4) // 6

        features = (
            16,
            16 * self.widen_factor,
            32 * self.widen_factor,
            64 * self.widen_factor,
        )

        x = nn.Conv(
            features=features[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=1,
        )

        x = self._make_group(
            x=x,
            num_blocks=num_blocks_per_group,
            features=features[1],
            stride=1,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=2,
        )

        x = self._make_group(
            x=x,
            num_blocks=num_blocks_per_group,
            features=features[2],
            stride=2,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=3,
        )

        x = self._make_group(
            x=x,
            num_blocks=num_blocks_per_group,
            features=features[3],
            stride=2,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=4,
        )

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = nn.relu(x)

        x = jnp.mean(  # Global average pool spatial dimensions.
            x,
            axis=(1, 2),
        )

        return x

    def _make_group(
        self,
        x: jnp.ndarray,
        num_blocks: int,
        features: int,
        stride: int,
        training: bool,
        sync_batch_stats: bool,
    ) -> jnp.ndarray:
        """Support make group."""
        for block_index in range(num_blocks):
            block_stride = stride if block_index == 0 else 1

            x = WideBasicBlock(
                features=features,
                stride=block_stride,
                dropout_rate=self.dropout_rate,
            )(
                x,
                training=training,
                sync_batch_stats=sync_batch_stats,
            )

        return x


def wide_resnet28_10_backbone(
    dropout_rate: float = 0.0,
) -> WideResNetBackbone:
    """Build a WideResNet-28-10 backbone."""
    return WideResNetBackbone(
        depth=28,
        widen_factor=10,
        dropout_rate=dropout_rate,
    )
