from __future__ import annotations

# Adapted from the Apache-2.0 MetaAugment controller implementation. The task
# classifier remains the ordinary AllTheMix ImageClassifier.
import flax.linen as nn
import jax.numpy as jnp


class MetaAugmentPolicy(nn.Module):
    """Predict one sample weight from task features and an operation embedding."""

    hidden_size: int = 100

    @nn.compact
    def __call__(
        self,
        image_features: jnp.ndarray,
        transform_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return sigmoid policy weights for augmented examples."""
        feature_branch = nn.relu(
            nn.Dense(
                self.hidden_size,
                name="feature_fc",
            )(
                image_features,
            )
        )
        transform_branch = nn.relu(
            nn.Dense(
                self.hidden_size,
                name="transform_fc",
            )(
                transform_embedding,
            )
        )
        features = jnp.concatenate(
            [
                feature_branch,
                transform_branch,
            ],
            axis=-1,
        )
        weights = nn.sigmoid(
            nn.Dense(
                1,
                name="weight_fc",
            )(
                features,
            )
        )

        return jnp.squeeze(
            weights,
            axis=-1,
        )
