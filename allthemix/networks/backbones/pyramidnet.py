from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.batch_norm import batch_norm
from allthemix.networks.utils.feature_hooks import FeatureHook, apply_feature_hook


class PyramidBottleneckBlock(nn.Module):
    in_channels: int
    out_channels: int
    bottleneck_channels: int
    stride: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Apply one PyramidNet bottleneck residual block."""
        residual = x

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )
        x = nn.relu(x)

        x = nn.Conv(
            features=self.bottleneck_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )
        x = nn.relu(x)

        x = nn.Conv(
            features=self.bottleneck_channels,
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

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        if self.stride != 1:
            residual = nn.avg_pool(
                residual,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            )

        channel_pad = self.out_channels - residual.shape[-1]  # Match shortcut channels.

        if channel_pad > 0:
            residual = jnp.pad(
                residual,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, channel_pad),
                ),
            )

        return x + residual  # Add residual shortcut to the transformed path.


@dataclass(frozen=True)
class PyramidNetChannelSchedule:
    channels: list[int]
    bottleneck_channels: list[int]


def build_pyramidnet_channel_schedule(
    depth: int,
    alpha: int,
    bottleneck: bool = True,
    initial_channels: int = 16,
) -> PyramidNetChannelSchedule:
    """Build pyramidnet channel schedule."""
    if not bottleneck:
        raise ValueError("This implementation currently supports bottleneck PyramidNet only.")

    if (depth - 2) % 9 != 0:
        raise ValueError(
            "For bottleneck PyramidNet on CIFAR, depth should satisfy (depth - 2) % 9 == 0."
        )

    blocks_per_group = (depth - 2) // 9
    total_blocks = blocks_per_group * 3
    add_rate = alpha / total_blocks  # Increase channels linearly across blocks.

    channels = []
    bottleneck_channels = []

    current_channels = initial_channels

    for block_index in range(total_blocks):
        current_channels += add_rate

        bottleneck_channel = int(round(current_channels))
        out_channel = bottleneck_channel * 4

        bottleneck_channels.append(bottleneck_channel)
        channels.append(out_channel)

    return PyramidNetChannelSchedule(
        channels=channels,
        bottleneck_channels=bottleneck_channels,
    )


class PyramidNetBackbone(nn.Module):
    depth: int = 200
    alpha: int = 240
    initial_channels: int = 16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        feature_hook: FeatureHook | None = None,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Extract image features with a PyramidNet backbone."""
        if (self.depth - 2) % 9 != 0:
            raise ValueError(
                "For bottleneck PyramidNet on CIFAR, depth should satisfy (depth - 2) % 9 == 0."
            )

        blocks_per_group = (self.depth - 2) // 9

        schedule = build_pyramidnet_channel_schedule(
            depth=self.depth,
            alpha=self.alpha,
            bottleneck=True,
            initial_channels=self.initial_channels,
        )

        x = nn.Conv(
            features=self.initial_channels,
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

        in_channels = self.initial_channels
        schedule_index = 0

        for group_index in range(3):
            for block_index in range(blocks_per_group):
                stride = 2 if group_index > 0 and block_index == 0 else 1

                out_channels = schedule.channels[schedule_index]
                bottleneck_channels = schedule.bottleneck_channels[schedule_index]

                x = PyramidBottleneckBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    stride=stride,
                )(
                    x,
                    training=training,
                    sync_batch_stats=sync_batch_stats,
                )

                in_channels = out_channels
                schedule_index += 1

            x = apply_feature_hook(
                features=x,
                feature_hook=feature_hook,
                layer_index=group_index + 2,
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


def pyramidnet200_backbone(
    alpha: int = 240,
) -> PyramidNetBackbone:
    """Build a PyramidNet-200 backbone."""
    return PyramidNetBackbone(
        depth=200,
        alpha=alpha,
        initial_channels=16,
    )
