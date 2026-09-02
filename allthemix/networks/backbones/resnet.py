from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.batch_norm import batch_norm
from allthemix.networks.utils.feature_hooks import FeatureHook, apply_feature_hook


class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    expansion: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Apply one ResNet basic residual block."""
        residual = x
        out_features = self.features * self.expansion

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

        x = nn.Conv(
            features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        if residual.shape != x.shape:
            residual = nn.Conv(
                features=out_features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)

            residual = batch_norm(
                residual,
                training=training,
                sync_batch_stats=sync_batch_stats,
            )

        x = x + residual  # Add residual shortcut to the transformed path.
        x = nn.relu(x)

        return x


class BottleneckBlock(nn.Module):
    features: int
    stride: int = 1
    expansion: int = 4

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Apply one ResNet bottleneck residual block."""
        residual = x
        out_features = self.features * self.expansion

        x = nn.Conv(
            features=self.features,
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

        x = nn.Conv(
            features=out_features,
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

        if residual.shape != x.shape:
            residual = nn.Conv(
                features=out_features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)

            residual = batch_norm(
                residual,
                training=training,
                sync_batch_stats=sync_batch_stats,
            )

        x = x + residual  # Add residual shortcut to the transformed path.
        x = nn.relu(x)

        return x


class ResNetBackbone(nn.Module):
    block_cls: type[nn.Module]
    block_sizes: Sequence[int]
    features: Sequence[int] = (64, 128, 256, 512)
    stem_type: str = "cifar"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        feature_hook: FeatureHook | None = None,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Extract image features with a ResNet backbone."""
        x = self._apply_stem(
            x=x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = apply_feature_hook(
            features=x,
            feature_hook=feature_hook,
            layer_index=1,
        )

        for stage_index, num_blocks in enumerate(self.block_sizes):
            for block_index in range(num_blocks):
                stride = 1

                if stage_index > 0 and block_index == 0:
                    stride = 2

                x = self.block_cls(
                    features=self.features[stage_index],
                    stride=stride,
                )(
                    x,
                    training=training,
                    sync_batch_stats=sync_batch_stats,
                )

            x = apply_feature_hook(
                features=x,
                feature_hook=feature_hook,
                layer_index=stage_index + 2,
            )

        x = jnp.mean(  # Global average pool spatial dimensions.
            x,
            axis=(1, 2),
        )

        return x

    def _apply_stem(
        self,
        x: jnp.ndarray,
        training: bool,
        sync_batch_stats: bool,
    ) -> jnp.ndarray:
        """Support apply stem."""
        if self.stem_type == "cifar":
            return self._apply_cifar_stem(
                x=x,
                training=training,
                sync_batch_stats=sync_batch_stats,
            )

        if self.stem_type == "imagenet":
            return self._apply_imagenet_stem(
                x=x,
                training=training,
                sync_batch_stats=sync_batch_stats,
            )

        raise ValueError(f"Unknown stem_type: {self.stem_type}")

    def _apply_cifar_stem(
        self,
        x: jnp.ndarray,
        training: bool,
        sync_batch_stats: bool,
    ) -> jnp.ndarray:
        """Support apply cifar stem."""
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
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

        return x

    def _apply_imagenet_stem(
        self,
        x: jnp.ndarray,
        training: bool,
        sync_batch_stats: bool,
    ) -> jnp.ndarray:
        """Support apply imagenet stem."""
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )(x)

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = nn.relu(x)

        x = nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )

        return x


def resnet18_backbone(stem_type: str = "cifar") -> ResNetBackbone:
    """Build a ResNet-18 backbone."""
    return ResNetBackbone(
        block_cls=BasicBlock,
        block_sizes=(2, 2, 2, 2),
        stem_type=stem_type,
    )


def resnet34_backbone(stem_type: str = "cifar") -> ResNetBackbone:
    """Build a ResNet-34 backbone."""
    return ResNetBackbone(
        block_cls=BasicBlock,
        block_sizes=(3, 4, 6, 3),
        stem_type=stem_type,
    )


def resnet50_backbone(stem_type: str = "cifar") -> ResNetBackbone:
    """Build a ResNet-50 backbone."""
    return ResNetBackbone(
        block_cls=BottleneckBlock,
        block_sizes=(3, 4, 6, 3),
        stem_type=stem_type,
    )


def resnet101_backbone(stem_type: str = "cifar") -> ResNetBackbone:
    """Build a ResNet-101 backbone."""
    return ResNetBackbone(
        block_cls=BottleneckBlock,
        block_sizes=(3, 4, 23, 3),
        stem_type=stem_type,
    )


def resnet152_backbone(stem_type: str = "cifar") -> ResNetBackbone:
    """Build a ResNet-152 backbone."""
    return ResNetBackbone(
        block_cls=BottleneckBlock,
        block_sizes=(3, 8, 36, 3),
        stem_type=stem_type,
    )
