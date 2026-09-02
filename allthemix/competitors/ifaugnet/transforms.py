from __future__ import annotations

import jax
import jax.numpy as jnp


def average_pool_same(
    values: jnp.ndarray,
    kernel_size: int = 4,
) -> jnp.ndarray:
    """Smooth dense transform fields without changing their spatial size."""
    if kernel_size <= 1:
        return values

    pad_before = (kernel_size - 1) // 2
    pad_after = kernel_size // 2
    padded = jnp.pad(
        values,
        (
            (0, 0),
            (pad_before, pad_after),
            (pad_before, pad_after),
            (0, 0),
        ),
        mode="edge",
    )
    pooled = jax.lax.reduce_window(
        padded,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, kernel_size, kernel_size, 1),
        window_strides=(1, 1, 1, 1),
        padding="VALID",
    )

    return pooled / float(kernel_size * kernel_size)


def _base_grid(
    height: int,
    width: int,
) -> jnp.ndarray:
    """Create a normalized sampling grid in y/x coordinate order."""
    ys = jnp.linspace(-1.0, 1.0, height)
    xs = jnp.linspace(-1.0, 1.0, width)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

    return jnp.stack(
        [yy, xx],
        axis=-1,
    )


def _pixel_base_grid(
    height: int,
    width: int,
) -> jnp.ndarray:
    """Create the source-coordinate grid used by the paper's affine rule."""
    ys = jnp.arange(
        height,
        dtype=jnp.float32,
    )
    xs = jnp.arange(
        width,
        dtype=jnp.float32,
    )
    yy, xx = jnp.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    return jnp.stack(
        [yy, xx],
        axis=-1,
    )


def _to_normalized_coordinates(
    grid: jnp.ndarray,
    height: int,
    width: int,
) -> jnp.ndarray:
    """Convert y/x pixel coordinates to the sampler's normalized domain."""
    height_scale = max(
        height - 1,
        1,
    )
    width_scale = max(
        width - 1,
        1,
    )
    y = 2.0 * grid[..., 0] / float(height_scale) - 1.0
    x = 2.0 * grid[..., 1] / float(width_scale) - 1.0

    return jnp.stack(
        [y, x],
        axis=-1,
    )


def _to_pixel_coordinates(
    grid: jnp.ndarray,
    height: int,
    width: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert normalized grid coordinates to floating pixel coordinates."""
    y = (grid[..., 0] + 1.0) * 0.5 * (height - 1)
    x = (grid[..., 1] + 1.0) * 0.5 * (width - 1)

    return y, x


def _sample_single(
    image: jnp.ndarray,
    grid: jnp.ndarray,
) -> jnp.ndarray:
    """Sample one image at a dense grid with bilinear interpolation."""
    height, width, _ = image.shape
    y, x = _to_pixel_coordinates(
        grid=grid,
        height=height,
        width=width,
    )
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    y0_clipped = jnp.clip(y0, 0, height - 1)
    x0_clipped = jnp.clip(x0, 0, width - 1)
    y1_clipped = jnp.clip(y1, 0, height - 1)
    x1_clipped = jnp.clip(x1, 0, width - 1)

    # Bilinear weights correspond to top-left, top-right, bottom-left, bottom-right.
    weight_a = (y1.astype(jnp.float32) - y) * (
        x1.astype(jnp.float32) - x
    )
    weight_b = (y1.astype(jnp.float32) - y) * (
        x - x0.astype(jnp.float32)
    )
    weight_c = (y - y0.astype(jnp.float32)) * (
        x1.astype(jnp.float32) - x
    )
    weight_d = (y - y0.astype(jnp.float32)) * (
        x - x0.astype(jnp.float32)
    )
    pixel_a = image[y0_clipped, x0_clipped]
    pixel_b = image[y0_clipped, x1_clipped]
    pixel_c = image[y1_clipped, x0_clipped]
    pixel_d = image[y1_clipped, x1_clipped]
    sampled = (
        pixel_a * weight_a[..., None]
        + pixel_b * weight_b[..., None]
        + pixel_c * weight_c[..., None]
        + pixel_d * weight_d[..., None]
    )
    in_bounds = (
        (y >= 0)
        & (y <= height - 1)
        & (x >= 0)
        & (x <= width - 1)
    )

    return jnp.where(
        in_bounds[..., None],
        sampled,
        0.0,
    )


def bilinear_sample(
    images: jnp.ndarray,
    grid: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorize bilinear image sampling over a batch."""
    return jax.vmap(
        _sample_single,
    )(
        images,
        grid,
    )


def apply_spatial_transform(
    images: jnp.ndarray,
    spatial_params: jnp.ndarray,
    spatial_scale: float = 0.20,
    smoothing_kernel: int = 4,
    parameterization: str = "guarded",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the guarded or paper-form dense local affine transform."""
    height, width = images.shape[1:3]
    spatial_params = jnp.nan_to_num(
        spatial_params,
        nan=0.0,
        posinf=(1.0 if parameterization == "guarded" else 0.0),
        neginf=(-1.0 if parameterization == "guarded" else 0.0),
    )
    if parameterization == "guarded":
        params = average_pool_same(
            spatial_params,
            smoothing_kernel,
        )
        params = jnp.tanh(
            params,
        )
        base = _base_grid(
            height=height,
            width=width,
        )
    elif parameterization == "paper":
        params = spatial_params
        base = _pixel_base_grid(
            height=height,
            width=width,
        )
    else:
        raise ValueError(
            "parameterization must be 'guarded' or 'paper'."
        )

    weights = params[..., :4].reshape(
        params.shape[0],
        height,
        width,
        2,
        2,
    )
    bias = params[..., 4:6]
    base_batch = jnp.broadcast_to(
        base,
        (images.shape[0], height, width, 2),
    )
    delta = jnp.einsum(  # delta(p) = A(p) * grid(p) + b(p).
        "bhwij,bhwj->bhwi",
        weights,
        base_batch,
    ) + bias

    if parameterization == "guarded":
        sample_grid = jnp.clip(
            base_batch + spatial_scale * delta,
            -1.5,
            1.5,
        )
    else:
        # Eq. 20 constructs a dense pixel-coordinate flow before smoothing.
        # Pool the displacement so an identity flow remains exactly identity.
        smoothed_delta = average_pool_same(
            delta,
            smoothing_kernel,
        )
        sample_grid = _to_normalized_coordinates(
            base_batch + smoothed_delta,
            height=height,
            width=width,
        )

    sample_grid = jnp.nan_to_num(
        sample_grid,
        nan=0.0,
        posinf=(1.5 if parameterization == "guarded" else 0.0),
        neginf=(-1.5 if parameterization == "guarded" else 0.0),
    )

    return bilinear_sample(
        images=images,
        grid=sample_grid,
    ), sample_grid


def apply_appearance_transform(
    images: jnp.ndarray,
    appearance_params: jnp.ndarray,
    appearance_scale: float = 0.25,
    smoothing_kernel: int = 4,
    parameterization: str = "guarded",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the guarded or paper-form dense local color transform."""
    channels = images.shape[-1]
    appearance_params = jnp.nan_to_num(
        appearance_params,
        nan=0.0,
        posinf=(1.0 if parameterization == "guarded" else 0.0),
        neginf=(-1.0 if parameterization == "guarded" else 0.0),
    )
    appearance_params = average_pool_same(
        appearance_params,
        smoothing_kernel,
    )
    weights = appearance_params[..., : channels * channels]
    bias = appearance_params[..., channels * channels :]
    weights = weights.reshape(
        *images.shape[:3],
        channels,
        channels,
    )

    if parameterization == "guarded":
        weights = jnp.tanh(
            weights,
        )
        bias = jnp.tanh(
            bias,
        )
    elif parameterization != "paper":
        raise ValueError(
            "parameterization must be 'guarded' or 'paper'."
        )

    delta = jnp.einsum(  # delta(p) = W(p) * image(p) + b(p).
        "bhwij,bhwj->bhwi",
        weights,
        images,
    ) + bias

    if parameterization == "guarded":
        transformed = jnp.clip(
            images + appearance_scale * delta,
            0.0,
            1.0,
        )
    else:
        transformed = images + delta

    return transformed, delta


def combine_transforms(
    images: jnp.ndarray,
    spatial_images: jnp.ndarray,
    appearance_images: jnp.ndarray,
    composition: str,
    clip_output: bool,
) -> jnp.ndarray:
    """Compose spatial output and appearance residual serially or in parallel."""
    if composition == "serial":
        combined = appearance_images
    elif composition == "parallel":
        combined = spatial_images + appearance_images - images
    else:
        raise ValueError(
            "composition must be 'serial' or 'parallel'."
        )

    if clip_output:
        combined = jnp.clip(
            combined,
            0.0,
            1.0,
        )

    return combined
