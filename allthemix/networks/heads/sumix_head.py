from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def l2_normalize(
    x: jnp.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Normalize vectors by their L2 norm."""
    norm = jnp.sqrt(  # Compute sqrt(sum(x^2) + eps) for stable normalization.
        jnp.sum(
            jnp.square(
                x,
            ),
            axis=axis,
            keepdims=True,
        )
        + eps
    )

    return x / norm  # Scale each vector to unit L2 length.


class SUMixUncertaintyHead(nn.Module):
    """
    Official-style SUMix uncertainty estimation head.

    This head predicts a class-wise uncertainty vector for each sample:

        features -> uncertainty_vector

    Shape:
        features:            [batch_size, ...]
        uncertainty_vector:  [batch_size, num_classes]

    The output follows the official idea:

        uncertainty = l2_normalize(softmax(linear(features)))
    """

    num_classes: int
    hidden_dim: int = 128
    dropout_rate: float = 0.0
    use_hidden_layer: bool = False

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """Predict class-wise SUMix uncertainty from features."""
        x = features.reshape(
            (
                features.shape[0],
                -1,
            )
        )

        if self.use_hidden_layer:
            x = nn.Dense(
                self.hidden_dim,
            )(
                x,
            )

            x = nn.relu(
                x,
            )

            if self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate,
                )(
                    x,
                    deterministic=not training,
                )

        uncertainty_logits = nn.Dense(
            self.num_classes,
        )(
            x,
        )

        uncertainty = nn.softmax(  # Convert uncertainty logits to class probabilities.
            uncertainty_logits,
            axis=-1,
        )

        uncertainty = l2_normalize(  # Match SUMix class-wise uncertainty normalization.
            uncertainty,
            axis=-1,
        )

        return uncertainty
