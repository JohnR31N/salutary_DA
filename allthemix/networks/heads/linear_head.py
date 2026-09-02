from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.torch_compat import (
    pytorch_default_linear_bias_init,
    pytorch_default_linear_kernel_init,
)


class LinearHead(nn.Module):
    num_classes: int
    init_style: str = "normal_0.01"

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Project features into class logits."""
        del training

        if self.init_style == "pytorch_default":
            logits = nn.Dense(  # PyTorch nn.Linear default initialization.
                features=self.num_classes,
                kernel_init=pytorch_default_linear_kernel_init,
                bias_init=pytorch_default_linear_bias_init(
                    fan_in=features.shape[-1],
                ),
            )(features)

            return logits

        if self.init_style != "normal_0.01":
            raise ValueError(
                "LinearHead init_style must be one of "
                "('normal_0.01', 'pytorch_default'). "
                f"Got {self.init_style}.",
            )

        logits = nn.Dense(  # Linear class projection.
            features=self.num_classes,
            kernel_init=nn.initializers.normal(
                stddev=0.01,
            ),
            bias_init=nn.initializers.zeros,
        )(features)

        return logits
