"""Saliency map computation backends."""

from __future__ import annotations

import numpy as np

COLUMN_STRIPE_STD_RATIO = 2.5

OPENCV_METHOD_NAMES = (
    "opencv",
    "opencv_finegrained",
    "finegrained",
)

SPECTRAL_RESIDUAL_METHOD_NAMES = (
    "spectral_residual",
    "sr",
)

SPECTRAL_RESIDUAL_NOFALLBACK_METHOD_NAMES = (
    "spectral_residual_nofallback",
    "sr_nofallback",
    "sr_no_fallback",
)

GRADIENT_METHOD_NAMES = (
    "gradient",
    "grad",
)

SALIENCY_METHOD_CHOICES = (
    *OPENCV_METHOD_NAMES,
    *SPECTRAL_RESIDUAL_METHOD_NAMES,
    *SPECTRAL_RESIDUAL_NOFALLBACK_METHOD_NAMES,
    *GRADIENT_METHOD_NAMES,
)


def _import_cv2():
    """Import OpenCV lazily for backends that require it."""
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise ImportError(
            "OpenCV is required for this saliency backend."
        ) from error

    return cv2


def normalize_saliency_map(
    saliency_map: np.ndarray,
) -> np.ndarray:
    """Normalize a saliency map to the [0, 1] range."""
    saliency_map = saliency_map.astype(np.float32)

    saliency_map = saliency_map - np.min(  # Shift minimum saliency to zero.
        saliency_map,
    )

    max_value = np.max(  # Find scale after min-shift.
        saliency_map,
    )

    if max_value < 1e-8:
        return np.zeros_like(
            saliency_map,
            dtype=np.float32,
        )

    saliency_map = saliency_map / max_value  # Scale maximum saliency to one.

    return saliency_map.astype(np.float32)


def compute_gradient_saliency_map(
    image: np.ndarray,
) -> np.ndarray:
    """Compute a simple edge-gradient saliency map."""
    image_float = image.astype(np.float32)

    if image_float.max() > 1.0:
        image_float = image_float / 255.0  # Convert uint8-like pixels to [0, 1].

    gray = np.mean(  # Average RGB channels into a grayscale map.
        image_float,
        axis=-1,
    )

    dx = np.zeros_like(
        gray,
        dtype=np.float32,
    )

    dy = np.zeros_like(
        gray,
        dtype=np.float32,
    )

    dx[:, 1:] = np.abs(  # Horizontal finite-difference magnitude.
        gray[:, 1:] - gray[:, :-1],
    )

    dy[1:, :] = np.abs(  # Vertical finite-difference magnitude.
        gray[1:, :] - gray[:-1, :],
    )

    saliency_map = dx + dy  # Combine horizontal and vertical edge energy.

    return normalize_saliency_map(
        saliency_map,
    )


def is_saliency_map_suspicious(
    saliency_map: np.ndarray,
) -> bool:
    """Return whether a saliency map is invalid or degenerate."""
    if saliency_map.ndim != 2:
        return True

    if not np.all(
        np.isfinite(
            saliency_map,
        )
    ):
        return True

    if np.max(saliency_map) - np.min(saliency_map) < 1e-8:  # Reject near-constant maps.
        return True

    row_mean_std = np.std(  # Measure variation across row-wise saliency means.
        np.mean(
            saliency_map,
            axis=1,
        )
    )

    col_mean_std = np.std(  # Measure variation across column-wise saliency means.
        np.mean(
            saliency_map,
            axis=0,
        )
    )

    if col_mean_std > row_mean_std * COLUMN_STRIPE_STD_RATIO:
        return True

    return False


def compute_opencv_finegrained_saliency_map(
    image: np.ndarray,
) -> np.ndarray:
    """Compute OpenCV fine-grained saliency with gradient fallback."""
    try:
        cv2 = _import_cv2()
    except ImportError:
        return compute_gradient_saliency_map(
            image,
        )

    if not hasattr(cv2, "saliency") or not hasattr(
        cv2.saliency,
        "StaticSaliencyFineGrained_create",
    ):
        return compute_gradient_saliency_map(
            image,
        )

    image_uint8 = image.astype(
        np.uint8,
    )

    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR,
    )

    detector = cv2.saliency.StaticSaliencyFineGrained_create()

    success, saliency_map = detector.computeSaliency(
        image_bgr,
    )

    if not success:
        return compute_gradient_saliency_map(
            image,
        )

    saliency_map = normalize_saliency_map(
        saliency_map,
    )

    if is_saliency_map_suspicious(
        saliency_map,
    ):
        return compute_gradient_saliency_map(
            image,
        )

    return saliency_map.astype(np.float32)


def _to_grayscale_float(
    image: np.ndarray,
) -> np.ndarray:
    """Convert an image to float grayscale in [0, 1] when needed."""
    image_float = image.astype(
        np.float32,
    )

    if image_float.max() > 1.0:
        image_float = image_float / 255.0  # Convert uint8-like pixels to [0, 1].

    if image_float.ndim == 2:
        gray = image_float
    elif image_float.ndim == 3 and image_float.shape[-1] == 3:
        gray = (  # Luma-weighted RGB to grayscale conversion.
            0.299 * image_float[..., 0]
            + 0.587 * image_float[..., 1]
            + 0.114 * image_float[..., 2]
        )
    else:
        raise ValueError(
            f"Unsupported image shape for grayscale conversion: {image.shape}"
        )

    return gray.astype(
        np.float32,
    )


def compute_spectral_residual_saliency_map_core(
    image: np.ndarray,
    avg_kernel_size: int = 3,
    blur_kernel_size: int = 7,
    blur_sigma: float = 3.0,
    max_size: int = 128,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute pure Spectral Residual saliency map.

    This function does NOT apply suspicious-map fallback.

    Pipeline:
        image -> grayscale -> FFT
        -> log amplitude
        -> log amplitude - local average log amplitude
        -> inverse FFT
        -> magnitude saliency
        -> Gaussian blur
        -> normalize to [0, 1]
    """
    if avg_kernel_size % 2 == 0:
        raise ValueError("avg_kernel_size must be odd.")

    if blur_kernel_size != 0 and blur_kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be odd or 0.")

    cv2 = _import_cv2()

    original_height = image.shape[0]
    original_width = image.shape[1]

    gray = _to_grayscale_float(
        image,
    )

    resized = False

    if max(
        original_height,
        original_width,
    ) > max_size:
        gray = cv2.resize(
            gray,
            dsize=(
                max_size,
                max_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        resized = True

    fft = np.fft.fft2(  # Transform grayscale image into frequency domain.
        gray,
    )

    amplitude = np.abs(  # Magnitude of each Fourier coefficient.
        fft,
    )

    phase = np.angle(  # Phase of each Fourier coefficient.
        fft,
    )

    log_amplitude = np.log(  # Log amplitude stabilizes multiplicative spectra.
        amplitude + eps,
    ).astype(
        np.float32,
    )

    average_log_amplitude = cv2.blur(  # Local average log spectrum.
        log_amplitude,
        ksize=(
            avg_kernel_size,
            avg_kernel_size,
        ),
        borderType=cv2.BORDER_REPLICATE,
    )

    spectral_residual = np.exp(  # Highlight frequency components above local average.
        log_amplitude - average_log_amplitude,
    )

    reconstructed = np.fft.ifft2(  # Reconstruct saliency response with original phase.
        spectral_residual * np.exp(
            1j * phase,
        )
    )

    saliency_map = np.abs(  # Magnitude gives the spatial saliency response.
        reconstructed,
    )

    if blur_kernel_size > 0:
        saliency_map = cv2.GaussianBlur(
            saliency_map.astype(np.float32),
            ksize=(
                blur_kernel_size,
                blur_kernel_size,
            ),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

    if resized:
        saliency_map = cv2.resize(
            saliency_map.astype(np.float32),
            dsize=(
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    saliency_map = normalize_saliency_map(
        saliency_map,
    )

    return saliency_map.astype(
        np.float32,
    )


def compute_spectral_residual_saliency_map(
    image: np.ndarray,
    avg_kernel_size: int = 3,
    blur_kernel_size: int = 7,
    blur_sigma: float = 3.0,
    max_size: int = 128,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute Spectral Residual saliency map with the old fallback behavior.

    This is kept for backward compatibility with previous experiments:
        if the SR map is suspicious, use gradient saliency instead.
    """
    saliency_map = compute_spectral_residual_saliency_map_core(
        image=image,
        avg_kernel_size=avg_kernel_size,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        max_size=max_size,
        eps=eps,
    )

    if is_saliency_map_suspicious(
        saliency_map,
    ):
        return compute_gradient_saliency_map(
            image,
        )

    return saliency_map.astype(
        np.float32,
    )


def compute_spectral_residual_saliency_map_nofallback(
    image: np.ndarray,
    avg_kernel_size: int = 3,
    blur_kernel_size: int = 7,
    blur_sigma: float = 3.0,
    max_size: int = 128,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute pure Spectral Residual saliency map without fallback.

    Use this for GuidedMixup SR experiments that should match the
    OpenMixUp / original GuidedMixup saliency recipe more closely.
    """
    return compute_spectral_residual_saliency_map_core(
        image=image,
        avg_kernel_size=avg_kernel_size,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        max_size=max_size,
        eps=eps,
    )


def compute_saliency_map(
    image: np.ndarray,
    method: str = "opencv",
) -> np.ndarray:
    """Compute a saliency map with the selected backend."""
    method_name = method.lower()

    if method_name in OPENCV_METHOD_NAMES:
        return compute_opencv_finegrained_saliency_map(
            image,
        )

    if method_name in SPECTRAL_RESIDUAL_METHOD_NAMES:
        return compute_spectral_residual_saliency_map(
            image,
        )

    if method_name in SPECTRAL_RESIDUAL_NOFALLBACK_METHOD_NAMES:
        return compute_spectral_residual_saliency_map_nofallback(
            image,
        )

    if method_name in GRADIENT_METHOD_NAMES:
        return compute_gradient_saliency_map(
            image,
        )

    raise ValueError(
        f"Unsupported saliency method: {method}"
    )


__all__ = [
    "COLUMN_STRIPE_STD_RATIO",
    "GRADIENT_METHOD_NAMES",
    "OPENCV_METHOD_NAMES",
    "SALIENCY_METHOD_CHOICES",
    "SPECTRAL_RESIDUAL_METHOD_NAMES",
    "SPECTRAL_RESIDUAL_NOFALLBACK_METHOD_NAMES",
    "compute_gradient_saliency_map",
    "compute_opencv_finegrained_saliency_map",
    "compute_saliency_map",
    "compute_spectral_residual_saliency_map",
    "compute_spectral_residual_saliency_map_core",
    "compute_spectral_residual_saliency_map_nofallback",
    "is_saliency_map_suspicious",
    "normalize_saliency_map",
]
