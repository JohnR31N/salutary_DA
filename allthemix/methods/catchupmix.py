from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from allthemix.methods.cutmix import _no_repeat_permutation, cutmix
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_no_repeat_batch_size,
    validate_num_classes,
    validate_positive,
    validate_positive_int,
)


class CatchupMixOutput(NamedTuple):
    images: jnp.ndarray
    labels_a: jnp.ndarray
    labels_b: jnp.ndarray
    lam: jnp.ndarray
    perm: jnp.ndarray
    layer: jnp.ndarray


class CatchupMixContext(NamedTuple):
    layer: jnp.ndarray
    lam: jnp.ndarray
    perm: jnp.ndarray


def _normalize_filter_influence(
    influence: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Normalize per-channel influence into a probability-like vector."""
    return influence / (  # Divide each channel influence by the sample total.
        jnp.sum(
            influence,
            axis=-1,
            keepdims=True,
        )
        + eps
    )


def catchup_mix_features(
    features: jnp.ndarray,
    lam: jnp.ndarray,
    perm: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply Catch-Up Mix to NHWC feature maps.

    For each pair, the source keeps floor(lambda * C) channels with the
    lowest relative filter influence (RFI), and the target supplies the rest.
    """

    if features.ndim != 4:
        raise ValueError(
            "Catch-Up Mix feature mixing expects NHWC feature maps.",
        )

    num_channels = features.shape[-1]
    target_features = features[perm]

    source_influence = jnp.sqrt(  # Compute source channel energy over spatial axes.
        jnp.sum(
            jnp.square(
                features,
            ),
            axis=(
                1,
                2,
            ),
        )
    )

    target_influence = jnp.sqrt(  # Compute target channel energy over spatial axes.
        jnp.sum(
            jnp.square(
                target_features,
            ),
            axis=(
                1,
                2,
            ),
        )
    )

    relative_influence = (  # Compare normalized source and target channel influence.
        _normalize_filter_influence(source_influence)
        - _normalize_filter_influence(target_influence)
    )

    relative_influence = jax.lax.stop_gradient(  # Rank channels without backprop through RFI.
        relative_influence,
    )

    channel_rank = jnp.argsort(  # Convert relative influence order into per-channel ranks.
        jnp.argsort(
            relative_influence,
            axis=-1,
        ),
        axis=-1,
    )

    num_source_channels = jnp.floor(  # Keep floor(lambda * C) source channels.
        lam * float(num_channels),
    ).astype(
        jnp.int32,
    )

    num_source_channels = jnp.clip(
        num_source_channels,
        0,
        num_channels,
    )

    source_threshold = (
        num_source_channels
        if num_source_channels.ndim == 0
        else num_source_channels[:, None]
    )

    source_mask = channel_rank < source_threshold  # Select lowest-RFI source channels.
    source_mask = source_mask[:, None, None, :]

    return jnp.where(  # Choose source or target features channel-wise.
        source_mask,
        features,
        target_features,
    )


def maybe_apply_catchup_mix(
    features: jnp.ndarray,
    catchup_mix: CatchupMixContext | None,
    layer_index: int,
) -> jnp.ndarray:
    """Apply CatchupMix features when the current layer matches the context."""
    if catchup_mix is None:
        return features

    return jax.lax.cond(
        catchup_mix.layer == layer_index,
        lambda current_features: catchup_mix_features(
            features=current_features,
            lam=catchup_mix.lam,
            perm=catchup_mix.perm,
        ),
        lambda current_features: current_features,
        features,
    )


def make_catchup_mix_feature_hook(
    catchup_mix: CatchupMixContext,
) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """Make catchup mix feature hook."""
    def feature_hook(
        features: jnp.ndarray,
        layer_index: int,
    ) -> jnp.ndarray:
        """Apply CatchupMix only at the selected feature layer."""
        return maybe_apply_catchup_mix(
            features=features,
            catchup_mix=catchup_mix,
            layer_index=layer_index,
        )

    return feature_hook


def catchupmix(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    cutmix_alpha: float = 1.0,
    cutmix_variant: str = "standard",
    cutmix_per_sample_lam: bool = False,
    cutmix_min_lam: float = 0.0,
    num_feature_layers: int = 5,
    no_repeat: bool = False,
) -> CatchupMixOutput:
    """
    Sample Catch-Up Mix metadata and optionally apply input-level CutMix.

    The sampled layer follows the paper's K = {0, 1, ..., num_feature_layers}.
    Layer 0 uses CutMix; feature layers are mixed inside the model forward.
    """

    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="catchupmix_alpha",
        value=alpha,
    )
    validate_positive(
        name="catchupmix_cutmix_alpha",
        value=cutmix_alpha,
    )
    validate_positive_int(
        name="catchupmix_num_layers",
        value=num_feature_layers,
    )
    validate_nhwc_images(
        images=images,
        method_name="CatchUpMix",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="CatchUpMix",
    )

    batch_size = images.shape[0]

    if no_repeat:
        validate_no_repeat_batch_size(
            batch_size=batch_size,
            method_name="CatchUpMix",
        )

    rng_layer, rng_lam, rng_perm, rng_cutmix = jax.random.split(
        rng,
        4,
    )

    layer = jax.random.randint(  # Sample input layer 0 or one feature layer.
        rng_layer,
        shape=(),
        minval=0,
        maxval=num_feature_layers + 1,
    )

    lam_shape = (
        (batch_size,)
        if cutmix_per_sample_lam
        else ()
    )

    lam = jax.random.beta(  # Sample feature/input mixing strength.
        rng_lam,
        alpha,
        alpha,
        shape=lam_shape,
    )

    if no_repeat:
        perm = _no_repeat_permutation(
            rng_perm,
            batch_size,
        )
    else:
        perm = jax.random.permutation(
            rng_perm,
            batch_size,
        )

    feature_output = CatchupMixOutput(
        images=images,
        labels_a=labels,
        labels_b=labels[perm],
        lam=lam,
        perm=perm,
        layer=layer,
    )

    cutmix_output = cutmix(
        rng=rng_cutmix,
        images=images,
        labels=labels,
        num_classes=num_classes,
        alpha=cutmix_alpha,
        prob=1.0,
        no_repeat=no_repeat,
        variant=cutmix_variant,
        per_sample_lam=cutmix_per_sample_lam,
        min_lam=cutmix_min_lam,
    )

    input_output = CatchupMixOutput(
        images=cutmix_output.images,
        labels_a=cutmix_output.labels_a,
        labels_b=cutmix_output.labels_b,
        lam=cutmix_output.lam,
        perm=cutmix_output.perm,
        layer=layer,
    )

    return jax.lax.cond(
        layer == 0,
        lambda _: input_output,
        lambda _: feature_output,
        operand=None,
    )
