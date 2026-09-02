from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.batch_norm import batch_norm
from allthemix.networks.torch_compat import (
    PYTORCH_3X3_PADDING,
    PYTORCH_7X7_PADDING,
    PYTORCH_KAIMING_CONV_INIT,
    PYTORCH_MODULE_DEFAULT_CONV_INIT,
)
from allthemix.networks.utils.feature_hooks import FeatureHook, apply_feature_hook


def _select_conv_kernel_init(
    pytorch_default_init: bool,
):
    """Return the requested PyTorch-compatible convolution initializer."""
    if pytorch_default_init:
        return PYTORCH_MODULE_DEFAULT_CONV_INIT

    return PYTORCH_KAIMING_CONV_INIT


class PreActBasicBlock(nn.Module):
    features: int
    stride: int = 1
    expansion: int = 1
    pytorch_default_init: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Apply one pre-activation residual basic block."""
        out_features = self.features * self.expansion
        conv_kernel_init = _select_conv_kernel_init(
            pytorch_default_init=self.pytorch_default_init,
        )

        residual = x

        x = batch_norm(
            x,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )

        x = nn.relu(x)

        if residual.shape[-1] != out_features or self.stride != 1:
            shortcut = nn.Conv(
                features=out_features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
                kernel_init=conv_kernel_init,
            )(x)
        else:
            shortcut = residual

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=PYTORCH_3X3_PADDING,
            use_bias=False,
            kernel_init=conv_kernel_init,
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
            padding=PYTORCH_3X3_PADDING,
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)

        x = x + shortcut  # Add the residual shortcut path.

        return x


class PreActResNetBackbone(nn.Module):
    block_sizes: Sequence[int]
    features: Sequence[int] = (64, 128, 256, 512)
    stem_type: str = "cifar"
    stem_bn_relu: bool = False
    pytorch_default_init: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        feature_hook: FeatureHook | None = None,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray:
        """Extract image features with a pre-activation ResNet backbone."""
        x = self._apply_stem(
            x,
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

                x = PreActBasicBlock(
                    features=self.features[stage_index],
                    stride=stride,
                    pytorch_default_init=self.pytorch_default_init,
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

    def _apply_stem(
        self,
        x: jnp.ndarray,
        training: bool,
        sync_batch_stats: bool,
    ) -> jnp.ndarray:
        """Support apply stem."""
        conv_kernel_init = _select_conv_kernel_init(
            pytorch_default_init=self.pytorch_default_init,
        )

        if self.stem_type == "cifar":
            x = nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=PYTORCH_3X3_PADDING,
                use_bias=False,
                kernel_init=conv_kernel_init,
            )(x)

            if self.stem_bn_relu:
                x = batch_norm(
                    x,
                    training=training,
                    sync_batch_stats=sync_batch_stats,
                )

                x = nn.relu(x)

            return x

        if self.stem_type == "imagenet":
            x = nn.Conv(
                features=64,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding=PYTORCH_7X7_PADDING,
                use_bias=False,
                kernel_init=conv_kernel_init,
            )(x)

            if self.stem_bn_relu:
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

        raise ValueError(f"Unknown stem_type: {self.stem_type}")


def preact_resnet18_backbone(
    stem_type: str = "cifar",
    stem_bn_relu: bool = False,
    pytorch_default_init: bool = False,
) -> PreActResNetBackbone:
    """Build a PreAct-ResNet-18 backbone."""
    return PreActResNetBackbone(
        block_sizes=(2, 2, 2, 2),
        stem_type=stem_type,
        stem_bn_relu=stem_bn_relu,
        pytorch_default_init=pytorch_default_init,
    )
