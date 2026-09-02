from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_num_classes,
    validate_positive,
    validate_probability,
    validate_saliency_maps_match_images,
)


def _build_saliency_box_mask(
    saliency_map: jnp.ndarray,
    cut_width: jnp.ndarray,
    cut_height: jnp.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build one SaliencyMix bbox mask.

    This follows the official CIFAR-style implementation:
    one saliency map is used to choose one bbox, and the same bbox
    is applied to the whole batch.
    """
    flat_index = jnp.argmax(  # Choose the most salient pixel as the box center.
        saliency_map.reshape(-1),
    )

    center_y = flat_index // image_width  # Convert flat index to row coordinate.
    center_x = flat_index % image_width  # Convert flat index to column coordinate.

    x1 = jnp.clip(
        center_x - cut_width // 2,
        0,
        image_width,
    )

    x2 = jnp.clip(
        center_x + cut_width // 2,
        0,
        image_width,
    )

    y1 = jnp.clip(
        center_y - cut_height // 2,
        0,
        image_height,
    )

    y2 = jnp.clip(
        center_y + cut_height // 2,
        0,
        image_height,
    )

    y_positions = jnp.arange(
        image_height,
    )[:, None]

    x_positions = jnp.arange(
        image_width,
    )[None, :]

    mask = (  # Select pixels inside the saliency-centered rectangle.
        (y_positions >= y1)
        & (y_positions < y2)
        & (x_positions >= x1)
        & (x_positions < x2)
    )

    mask = mask[None, :, :, None]

    patch_area = (  # Measure the clipped pasted rectangle area.
        (x2 - x1)
        * (y2 - y1)
    )

    return mask, patch_area


def saliencymix(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    saliency_maps: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    prob: float = 0.5,
    per_sample: bool = False,
    paired_images: jnp.ndarray | None = None,
    paired_labels: jnp.ndarray | None = None,
    paired_saliency_maps: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply SaliencyMix using saliency-centered pasted boxes.

    By default this uses one shared saliency-guided bbox for the whole batch,
    matching the official CIFAR implementation style. When per_sample is true,
    each sample receives its own beta lambda and saliency-guided bbox.

    Steps:
        1. Shuffle the batch.
        2. Use shuffled-image saliency to choose one or more bboxes.
        3. Apply the bbox mask to paste shuffled pixels.
        4. Paste shuffled image patches into original images.

    Returns:
        mixed_images, labels_a, labels_b, lam
    """
    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="saliencymix_alpha",
        value=alpha,
    )
    validate_probability(
        name="saliencymix_prob",
        value=prob,
    )
    validate_nhwc_images(
        images=images,
        method_name="SaliencyMix",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="SaliencyMix",
    )
    validate_saliency_maps_match_images(
        saliency_maps=saliency_maps,
        images=images,
        method_name="SaliencyMix",
    )

    if (paired_images is None) != (paired_labels is None):
        raise ValueError(
            "SaliencyMix paired_images and paired_labels must be provided together.",
        )

    if paired_images is not None:
        if paired_saliency_maps is None:
            raise ValueError(
                "SaliencyMix paired_saliency_maps is required when paired_images "
                "are provided.",
            )

        validate_nhwc_images(
            images=paired_images,
            method_name="SaliencyMix paired",
        )
        validate_labels_match_images(
            labels=paired_labels,
            images=paired_images,
            method_name="SaliencyMix paired",
        )
        validate_saliency_maps_match_images(
            saliency_maps=paired_saliency_maps,
            images=paired_images,
            method_name="SaliencyMix paired",
        )

        if paired_images.shape != images.shape:
            raise ValueError(
                "SaliencyMix paired_images must have the same shape as images. "
                f"Got {paired_images.shape} and {images.shape}.",
            )

        if paired_labels.shape != labels.shape:
            raise ValueError(
                "SaliencyMix paired_labels must have the same shape as labels. "
                f"Got {paired_labels.shape} and {labels.shape}.",
            )

    batch_size = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]

    rng_apply, rng_lam, rng_perm = jax.random.split(
        rng,
        3,
    )

    apply_mix = jax.random.uniform(
        rng_apply,
        shape=(),
    ) < prob

    lam_shape = (
        (batch_size,)
        if per_sample
        else ()
    )

    lam = jax.random.beta(  # Sample retained-area ratio from Beta(alpha, alpha).
        rng_lam,
        alpha,
        alpha,
        shape=lam_shape,
    )

    if paired_images is None:
        permutation = jax.random.permutation(
            rng_perm,
            batch_size,
        )

        shuffled_images = images[
            permutation,
        ]

        shuffled_labels = labels[
            permutation,
        ]

        shuffled_saliency_maps = saliency_maps[
            permutation,
        ]

    else:
        del rng_perm
        permutation = jnp.arange(
            batch_size,
            dtype=jnp.int32,
        )
        shuffled_images = paired_images
        shuffled_labels = paired_labels
        shuffled_saliency_maps = paired_saliency_maps

    cut_ratio = jnp.sqrt(  # Convert pasted-area ratio to side-length ratio.
        1.0 - lam,
    )

    cut_width = (  # Convert sampled side ratio to box width.
        image_width * cut_ratio
    ).astype(jnp.int32)

    cut_height = (  # Convert sampled side ratio to box height.
        image_height * cut_ratio
    ).astype(jnp.int32)

    if per_sample:
        mask, patch_area = jax.vmap(
            lambda saliency_map, sample_width, sample_height: _build_saliency_box_mask(
                saliency_map=saliency_map,
                cut_width=sample_width,
                cut_height=sample_height,
                image_height=image_height,
                image_width=image_width,
            )
        )(
            shuffled_saliency_maps,
            cut_width,
            cut_height,
        )

        mask = jnp.squeeze(  # Remove the singleton batch axis from each vmapped mask.
            mask,
            axis=1,
        )

    else:
        saliency_map_for_bbox = shuffled_saliency_maps[0]

        mask, patch_area = _build_saliency_box_mask(
            saliency_map=saliency_map_for_bbox,
            cut_width=cut_width,
            cut_height=cut_height,
            image_height=image_height,
            image_width=image_width,
        )

    mixed_candidate = jnp.where(  # Paste shuffled pixels inside the saliency box.
        mask,
        shuffled_images,
        images,
    )

    adjusted_lam = 1.0 - (  # Recompute lambda from actual pasted area.
        patch_area.astype(jnp.float32)
        / float(image_height * image_width)
    )

    mixed_images = jnp.where(
        apply_mix,
        mixed_candidate,
        images,
    )

    labels_a = labels

    labels_b = jnp.where(
        apply_mix,
        shuffled_labels,
        labels,
    )

    final_lam = jnp.where(
        apply_mix,
        adjusted_lam,
        jnp.array(
            1.0,
            dtype=jnp.float32,
        ),
    )

    return MixOutput(
        images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=final_lam,
        perm=permutation,
    )
