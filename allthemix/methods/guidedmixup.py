from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_num_classes,
    validate_odd_positive_int,
    validate_positive,
    validate_probability,
    validate_saliency_maps_match_images,
)

INF = 1e9


def _normalize_saliency_maps(
    saliency_maps: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Normalize saliency maps into nonnegative per-sample distributions."""
    if saliency_maps.ndim == 3:
        saliency_maps = saliency_maps[..., None]

    saliency_maps = jnp.maximum(  # Drop negative saliency values before normalization.
        saliency_maps,
        0.0,
    )

    saliency_sum = jnp.sum(  # Sum each saliency map over spatial and channel axes.
        saliency_maps,
        axis=(1, 2, 3),
        keepdims=True,
    )

    saliency_maps = saliency_maps / (  # Normalize each map to unit total mass.
        saliency_sum + eps
    )

    return saliency_maps


def _make_gaussian_kernel_1d(
    kernel_size: int,
    sigma: float = 3.0,
) -> jnp.ndarray:
    """Build a normalized one-dimensional Gaussian kernel."""
    if kernel_size <= 1:
        return jnp.ones(
            (
                1,
            ),
            dtype=jnp.float32,
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "guidedmixup_blur_kernel must be odd."
        )

    radius = kernel_size // 2

    coords = jnp.arange(
        -radius,
        radius + 1,
        dtype=jnp.float32,
    )

    kernel = jnp.exp(  # Evaluate the Gaussian density at discrete offsets.
        -(
            coords * coords
        )
        / (
            2.0 * sigma * sigma
        )
    )

    kernel = kernel / jnp.sum(  # Normalize kernel weights to preserve mass.
        kernel,
    )

    return kernel


def _gaussian_blur_2d_single_channel(
    saliency_maps: jnp.ndarray,
    kernel_size: int,
    sigma: float = 3.0,
) -> jnp.ndarray:
    """Apply torchvision-style reflect-padded Gaussian blur."""
    if kernel_size <= 1:
        return saliency_maps

    if kernel_size % 2 == 0:
        raise ValueError(
            "guidedmixup_blur_kernel must be odd."
        )

    kernel = _make_gaussian_kernel_1d(
        kernel_size=kernel_size,
        sigma=sigma,
    )

    pad = kernel_size // 2

    padded_y = jnp.pad(
        saliency_maps,
        pad_width=(
            (0, 0),
            (pad, pad),
            (0, 0),
            (0, 0),
        ),
        mode="reflect",
    )

    blurred_y = jnp.zeros_like(
        saliency_maps,
    )

    for dy in range(kernel_size):
        blurred_y = blurred_y + kernel[dy] * padded_y[
            :,
            dy : dy + saliency_maps.shape[1],
            :,
            :,
        ]

    padded_x = jnp.pad(
        blurred_y,
        pad_width=(
            (0, 0),
            (0, 0),
            (pad, pad),
            (0, 0),
        ),
        mode="reflect",
    )

    blurred = jnp.zeros_like(
        saliency_maps,
    )

    for dx in range(kernel_size):
        blurred = blurred + kernel[dx] * padded_x[
            :,
            :,
            dx : dx + saliency_maps.shape[2],
            :,
        ]

    return blurred


# Backward-compatible alias.
_box_blur_2d_single_channel = _gaussian_blur_2d_single_channel


def _mean_filter_2d_single_channel(
    values: jnp.ndarray,
    kernel_size: int = 3,
) -> jnp.ndarray:
    """Apply a replicate-padded mean filter to single-channel maps."""
    if kernel_size <= 1:
        return values

    if kernel_size % 2 == 0:
        raise ValueError(
            "spectral residual kernel_size must be odd."
        )

    pad = kernel_size // 2

    padded = jnp.pad(
        values,
        pad_width=(
            (0, 0),
            (pad, pad),
            (pad, pad),
            (0, 0),
        ),
        mode="edge",
    )

    filtered = jnp.zeros_like(
        values,
    )

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            filtered = filtered + padded[
                :,
                dy : dy + values.shape[1],
                dx : dx + values.shape[2],
                :,
            ]

    return filtered / float(kernel_size * kernel_size)


def _rgb_to_grayscale(
    images: jnp.ndarray,
) -> jnp.ndarray:
    """Convert NHWC RGB images to one-channel grayscale maps."""
    if images.shape[-1] == 1:
        return images

    if images.shape[-1] != 3:
        raise ValueError(
            "Guided-SR expects images with either 1 or 3 channels."
        )

    weights = jnp.asarray(
        [
            0.2989,
            0.5870,
            0.1140,
        ],
        dtype=images.dtype,
    )

    grayscale = jnp.sum(  # Match torchvision RGB-to-gray luminance weighting.
        images * weights,
        axis=-1,
        keepdims=True,
    )

    return grayscale


def _compute_spectral_residual_saliency_maps(
    images: jnp.ndarray,
    blur_kernel: int = 7,
    blur_sigma: float = 3.0,
    spectral_kernel_size: int = 3,
    max_size: int = 128,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Compute official Guided-SR style spectral residual saliency maps."""
    grayscale = _rgb_to_grayscale(
        images,
    )

    batch_size = grayscale.shape[0]
    image_height = grayscale.shape[1]
    image_width = grayscale.shape[2]

    needs_resize = max(
        image_height,
        image_width,
    ) > max_size

    if needs_resize:
        grayscale = jax.image.resize(
            grayscale,
            shape=(
                batch_size,
                max_size,
                max_size,
                1,
            ),
            method="bilinear",
        )

    frequency = jnp.fft.fft2(  # Transform grayscale images into frequency space.
        grayscale,
        axes=(
            1,
            2,
        ),
    )

    magnitude = jnp.sqrt(  # Stable Fourier magnitude for spectral residual scaling.
        jnp.real(
            frequency,
        )
        ** 2
        + jnp.imag(
            frequency,
        )
        ** 2
        + eps
    )

    log_magnitude = jnp.log(  # Work in log amplitude as in spectral residual saliency.
        magnitude,
    )

    local_average = _mean_filter_2d_single_channel(
        values=log_magnitude,
        kernel_size=spectral_kernel_size,
    )

    residual_scale = jnp.exp(  # Remove local average spectrum to emphasize novelty.
        log_magnitude - local_average,
    )

    residual_frequency = frequency * (  # Preserve phase while replacing amplitude.
        residual_scale
        / magnitude
    )

    saliency_response = jnp.fft.ifft2(
        residual_frequency,
        axes=(
            1,
            2,
        ),
    )

    saliency_maps = jnp.abs(  # Magnitude gives the spatial saliency response.
        saliency_response,
    )

    saliency_maps = _gaussian_blur_2d_single_channel(
        saliency_maps=saliency_maps,
        kernel_size=blur_kernel,
        sigma=blur_sigma,
    )

    if needs_resize:
        saliency_maps = jax.image.resize(
            saliency_maps,
            shape=(
                batch_size,
                image_height,
                image_width,
                1,
            ),
            method="bilinear",
        )

    return jnp.maximum(
        saliency_maps,
        0.0,
    )


def _compute_l2_distance_matrix(
    saliency_maps: jnp.ndarray,
) -> jnp.ndarray:
    """Compute pairwise L2 distances between flattened saliency maps."""
    batch_size = saliency_maps.shape[0]

    flat_maps = saliency_maps.reshape(
        batch_size,
        -1,
    )

    diff = flat_maps[:, None, :] - flat_maps[None, :, :]  # Pairwise map residuals.

    distance_matrix = jnp.sqrt(  # Euclidean distance between every pair of maps.
        jnp.sum(
            diff * diff,
            axis=-1,
        )
        + 1e-12
    )

    return distance_matrix


def _onecycle_cover(
    distance_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Official GuidedMixup greedy pairing logic.

    This follows the one-cycle cover procedure used in the PyTorch
    implementation:

        1. Set diagonal to -INF to avoid self-pairing.
        2. Select the global maximum pair (row, col).
        3. Set permutation[row] = col.
        4. Remove column row from future selection.
        5. Repeatedly set row = col and select the largest remaining
           entry in that row.
        6. Close the cycle by mapping the final col back to first_row.

    This is different from a standard max-weight matching.
    """
    batch_size = distance_matrix.shape[0]

    distance_matrix = jnp.where(
        jnp.eye(
            batch_size,
            dtype=bool,
        ),
        -INF,
        distance_matrix,
    )

    max_idx = jnp.argmax(
        distance_matrix,
    )

    row = max_idx // batch_size
    col = max_idx % batch_size
    first_row = row

    permutation = jnp.zeros(
        (
            batch_size,
        ),
        dtype=jnp.int32,
    )

    permutation = permutation.at[row].set(
        col.astype(jnp.int32),
    )

    distance_matrix = distance_matrix.at[:, row].set(
        -INF,
    )

    for _ in range(batch_size - 2):
        row = col
        col = jnp.argmax(
            distance_matrix[row],
        )

        permutation = permutation.at[row].set(
            col.astype(jnp.int32),
        )

        distance_matrix = distance_matrix.at[:, row].set(
            -INF,
        )

    permutation = permutation.at[col].set(
        first_row.astype(jnp.int32),
    )

    return permutation


def _pair_by_random(
    rng: jax.Array,
    batch_size: int,
) -> jnp.ndarray:
    """
    Official random condition behavior is random permutation.

    This can contain fixed points, matching torch.randperm /
    np.random.permutation behavior.
    """
    return jax.random.permutation(
        rng,
        batch_size,
    ).astype(
        jnp.int32,
    )


def _pair_by_greedy_onecycle(
    saliency_maps: jnp.ndarray,
) -> jnp.ndarray:
    """Pair samples with the greedy one-cycle saliency-distance rule."""
    distance_matrix = _compute_l2_distance_matrix(
        saliency_maps,
    )

    permutation = _onecycle_cover(
        distance_matrix,
    )

    return permutation


# Backward-compatible public helper names used by tests and debug scripts.
def _greedy_pairing_matrix(
    distance_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the backward-compatible greedy pairing helper."""
    return _onecycle_cover(
        distance_matrix,
    )


def _pair_by_greedy_max_distance(
    saliency_maps: jnp.ndarray,
) -> jnp.ndarray:
    """Pair samples by maximum saliency-map distance."""
    return _pair_by_greedy_onecycle(
        saliency_maps,
    )


def _build_pairing(
    rng: jax.Array,
    saliency_maps: jnp.ndarray,
    condition: str,
) -> jnp.ndarray:
    """Build GuidedMixup pair indices for the selected condition."""
    condition = condition.lower()

    batch_size = saliency_maps.shape[0]

    if condition == "random":
        return _pair_by_random(
            rng=rng,
            batch_size=batch_size,
        )

    if condition == "greedy":
        return _pair_by_greedy_onecycle(
            saliency_maps,
        )

    raise ValueError(
        "Unsupported guidedmixup_condition: "
        f"{condition}. Expected one of: random, greedy."
    )


def _guidedmixup_from_saliency(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    saliency_maps: jnp.ndarray,
    blur_kernel: int = 7,
    condition: str = "greedy",
    eps: float = 1e-8,
) -> MixOutput:
    """Apply the GuidedMixup core formula using provided saliency maps."""
    validate_odd_positive_int(
        name="guidedmixup_blur_kernel",
        value=blur_kernel,
    )
    validate_nhwc_images(
        images=images,
        method_name="GuidedMixup",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="GuidedMixup",
    )
    validate_saliency_maps_match_images(
        saliency_maps=saliency_maps,
        images=images,
        method_name="GuidedMixup",
    )

    saliency_maps = _normalize_saliency_maps(
        saliency_maps,
        eps=eps,
    )

    saliency_maps = _gaussian_blur_2d_single_channel(
        saliency_maps,
        kernel_size=blur_kernel,
        sigma=3.0,
    )

    saliency_maps = _normalize_saliency_maps(
        saliency_maps,
        eps=eps,
    )

    permutation = _build_pairing(
        rng=rng,
        saliency_maps=saliency_maps,
        condition=condition,
    )

    paired_images = images[permutation]
    paired_labels = labels[permutation]
    paired_saliency_maps = saliency_maps[permutation]

    pixel_mask = saliency_maps / (  # Allocate each pixel by relative saliency mass.
        saliency_maps + paired_saliency_maps + eps
    )

    guided_images = pixel_mask * images + (  # Blend paired images per pixel.
        1.0 - pixel_mask
    ) * paired_images

    lam = jnp.mean(  # Average pixel mask to get each sample's target lambda.
        pixel_mask,
        axis=(1, 2, 3),
    )

    labels_a = labels
    labels_b = paired_labels

    return MixOutput(
        images=guided_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=permutation,
    )


def guidedmixup(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    saliency_maps: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    prob: float = 1.0,
    blur_kernel: int = 7,
    condition: str = "greedy",
    eps: float = 1e-8,
) -> MixOutput:
    """Apply GuidedMixup with official batch-level mix probability."""
    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="guidedmixup_alpha",
        value=alpha,
    )
    validate_probability(
        name="guidedmixup_prob",
        value=prob,
    )

    rng_apply, rng_pairing = jax.random.split(
        rng,
    )

    apply_mix = jax.random.bernoulli(
        rng_apply,
        p=prob,
        shape=(),
    )

    mixed = _guidedmixup_from_saliency(
        rng=rng_pairing,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        blur_kernel=blur_kernel,
        condition=condition,
        eps=eps,
    )

    labels_b = jnp.where(
        apply_mix,
        mixed.labels_b,
        labels,
    )

    guided_images = jnp.where(
        apply_mix,
        mixed.images,
        images,
    )

    lam = jnp.where(
        apply_mix,
        mixed.lam,
        jnp.ones_like(
            mixed.lam,
        ),
    )

    return MixOutput(
        images=guided_images,
        labels_a=mixed.labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=mixed.perm,
    )


def guided_sr(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    prob: float = 1.0,
    blur_kernel: int = 7,
    condition: str = "greedy",
    eps: float = 1e-8,
) -> MixOutput:
    """Apply official Guided-SR with online spectral residual saliency."""
    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="guidedmixup_alpha",
        value=alpha,
    )
    validate_probability(
        name="guidedmixup_prob",
        value=prob,
    )
    validate_odd_positive_int(
        name="guidedmixup_blur_kernel",
        value=blur_kernel,
    )
    validate_nhwc_images(
        images=images,
        method_name="Guided-SR",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="Guided-SR",
    )

    rng_apply, rng_pairing = jax.random.split(
        rng,
    )

    apply_mix = jax.random.bernoulli(
        rng_apply,
        p=prob,
        shape=(),
    )

    def mix_branch(
        _,
    ) -> MixOutput:
        """Compute online SR saliency only when Guided-SR is applied."""
        saliency_maps = _compute_spectral_residual_saliency_maps(
            images=images,
        )

        return _guidedmixup_from_saliency(
            rng=rng_pairing,
            images=images,
            labels=labels,
            saliency_maps=saliency_maps,
            blur_kernel=blur_kernel,
            condition=condition,
            eps=eps,
        )

    def clean_branch(
        _,
    ) -> MixOutput:
        """Return an unmixed batch when official mix_prob skips Guided-SR."""
        lam = jnp.ones(
            (
                labels.shape[0],
            ),
            dtype=images.dtype,
        )

        return MixOutput(
            images=images,
            labels_a=labels,
            labels_b=labels,
            lam=lam,
            perm=jnp.arange(
                labels.shape[0],
                dtype=jnp.int32,
            ),
        )

    return jax.lax.cond(
        apply_mix,
        mix_branch,
        clean_branch,
        operand=None,
    )
