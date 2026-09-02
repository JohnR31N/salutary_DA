from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from allthemix.networks.utils.feature_hooks import FeatureHook


class ImageClassifier(nn.Module):
    backbone: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        return_features: bool = False,
        feature_hook: FeatureHook | None = None,
        sync_batch_stats: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Run backbone feature extraction followed by the classifier head."""
        features = self.backbone(
            x,
            training=training,
            feature_hook=feature_hook,
            sync_batch_stats=sync_batch_stats,
        )

        logits = self.head(
            features,
            training=training,
        )

        if return_features:
            return logits, features

        return logits
