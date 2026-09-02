from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.cutmix import _no_repeat_permutation
from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_no_repeat_batch_size,
    validate_num_classes,
    validate_positive,
    validate_probability,
)


def _make_fmix_mask(
    rng: jax.Array,
    height: int,
    width: int,
    lam: jnp.ndarray,
    decay_power: float,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample an official-style low-frequency binary FMix mask."""
    fy = jnp.fft.fftfreq(height).reshape(height, 1)  # Vertical frequency grid.
    fx = jnp.fft.rfftfreq(width).reshape(1, -1)  # Real-FFT horizontal grid.

    frequency = jnp.sqrt(fx * fx + fy * fy)  # Radial spatial frequency.
    min_frequency = 1.0 / float(max(height, width))
    frequency = jnp.maximum(frequency, min_frequency)

    scale = 1.0 / (frequency ** decay_power)  # Favor low frequencies by power decay.

    rng_spectrum, rng_round = jax.random.split(
        rng,
    )

    param = jax.random.normal(
        rng_spectrum,
        shape=(
            1,
            frequency.shape[0],
            frequency.shape[1],
            2,
        ),
    )

    raw_spectrum = scale[None, :, :, None] * param  # Shape random noise in Fourier space.

    # Match the reference FMix implementation exactly:
    # spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    spectrum = raw_spectrum[:, 0] + 1j * raw_spectrum[:, 1]

    low_frequency_image = jnp.real(  # Transform back to a smooth spatial field.
        jnp.fft.irfftn(
            spectrum,
            s=(
                height,
                width,
            ),
        ),
    )

    low_frequency_image = low_frequency_image[
        0,
        :height,
        :width,
    ]

    low_frequency_image = low_frequency_image - jnp.min(low_frequency_image)
    low_frequency_image = low_frequency_image / jnp.maximum(
        jnp.max(low_frequency_image),
        jnp.asarray(
            1e-12,
            dtype=low_frequency_image.dtype,
        ),
    )

    flat = low_frequency_image.reshape(-1)
    num_pixels = height * width

    lam_pixels = lam * num_pixels
    use_ceil = jax.random.bernoulli(
        rng_round,
        p=0.5,
        shape=(),
    )
    num_keep = jnp.where(  # Officially choose ceil/floor randomly.
        use_ceil,
        jnp.ceil(lam_pixels),
        jnp.floor(lam_pixels),
    ).astype(jnp.int32)

    num_keep = jnp.clip(
        num_keep,
        1,
        num_pixels - 1,
    )

    sorted_flat = jnp.sort(flat)
    threshold_index = num_pixels - num_keep
    threshold = sorted_flat[threshold_index]  # Pick quantile threshold for mask area.

    mask = (low_frequency_image >= threshold).astype(dtype)  # Binarize smooth field.

    mask = mask[:, :, None]

    return mask, lam


def fmix(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    decay_power: float = 3.0,
    prob: float = 1.0,
    per_sample: bool = False,
    no_repeat: bool = False,
    paired_images: jnp.ndarray | None = None,
    paired_labels: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply FMix using a low-frequency binary mask."""
    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="fmix_alpha",
        value=alpha,
    )
    validate_positive(
        name="fmix_decay",
        value=decay_power,
    )
    validate_probability(
        name="fmix_prob",
        value=prob,
    )
    validate_nhwc_images(
        images=images,
        method_name="FMix",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="FMix",
    )

    if (paired_images is None) != (paired_labels is None):
        raise ValueError(
            "FMix paired_images and paired_labels must be provided together.",
        )

    if paired_images is not None:
        validate_nhwc_images(
            images=paired_images,
            method_name="FMix paired",
        )
        validate_labels_match_images(
            labels=paired_labels,
            images=paired_images,
            method_name="FMix paired",
        )

        if paired_images.shape != images.shape:
            raise ValueError(
                "FMix paired_images must have the same shape as images. "
                f"Got {paired_images.shape} and {images.shape}.",
            )

        if paired_labels.shape != labels.shape:
            raise ValueError(
                "FMix paired_labels must have the same shape as labels. "
                f"Got {paired_labels.shape} and {labels.shape}.",
            )

    batch_size = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]

    if no_repeat and paired_images is None:
        validate_no_repeat_batch_size(
            batch_size=batch_size,
            method_name="FMix",
        )

    if image_height * image_width <= 1:
        raise ValueError(
            "FMix requires at least two spatial pixels to build a binary mask.",
        )

    rng_lam, rng_perm, rng_mask, rng_prob = jax.random.split(
        rng,
        4,
    )

    lam_shape = (
        (batch_size,)
        if per_sample
        else ()
    )

    lam = jax.random.beta(  # Sample target mask fraction from Beta(alpha, alpha).
        rng_lam,
        alpha,
        alpha,
        shape=lam_shape,
    )

    do_fmix = jax.random.bernoulli(
        rng_prob,
        p=prob,
        shape=(),
    )

    if paired_images is None:
        if no_repeat:
            permutation = _no_repeat_permutation(
                rng_perm,
                batch_size,
            )

        else:
            permutation = jax.random.permutation(
                rng_perm,
                batch_size,
            )

        shuffled_images = images[permutation]
        shuffled_labels = labels[permutation]

    else:
        permutation = jnp.arange(
            batch_size,
            dtype=jnp.int32,
        )
        shuffled_images = paired_images
        shuffled_labels = paired_labels

    if per_sample:
        mask_rngs = jax.random.split(
            rng_mask,
            batch_size,
        )

        mask, adjusted_lam = jax.vmap(
            lambda mask_rng, sample_lam: _make_fmix_mask(
                rng=mask_rng,
                height=image_height,
                width=image_width,
                lam=sample_lam,
                decay_power=decay_power,
                dtype=images.dtype,
            )
        )(
            mask_rngs,
            lam,
        )

    else:
        mask, adjusted_lam = _make_fmix_mask(
            rng=rng_mask,
            height=image_height,
            width=image_width,
            lam=lam,
            decay_power=decay_power,
            dtype=images.dtype,
        )

    fmixed_images = mask * images + (1.0 - mask) * shuffled_images  # Blend by mask.

    mixed_images = jnp.where(
        do_fmix,
        fmixed_images,
        images,
    )

    labels_a = labels

    labels_b = jnp.where(
        do_fmix,
        shuffled_labels,
        labels,
    )

    lam = jnp.where(
        do_fmix,
        adjusted_lam,
        1.0,
    )

    return MixOutput(
        images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=permutation,
    )
