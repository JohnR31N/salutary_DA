from __future__ import annotations

from typing import Any


def normalize_method_name(
    name: str,
) -> str:
    """Normalize user-facing method names into internal identifiers."""
    return name.lower().replace("-", "_").replace(" ", "_")


def format_shape(
    value: Any,
) -> str:
    """Format an array-like shape for validation errors."""
    shape = getattr(
        value,
        "shape",
        None,
    )

    return str(shape) if shape is not None else "<unknown>"


def validate_positive(
    name: str,
    value: float,
) -> None:
    """Require a strictly positive scalar hyperparameter."""
    if value <= 0.0:
        raise ValueError(
            f"{name} must be > 0. Got {value}.",
        )


def validate_probability(
    name: str,
    value: float,
) -> None:
    """Require a probability in the closed unit interval."""
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]. Got {value}.",
        )


def validate_positive_int(
    name: str,
    value: int,
) -> None:
    """Require a strictly positive integer hyperparameter."""
    if value < 1:
        raise ValueError(
            f"{name} must be >= 1. Got {value}.",
        )


def validate_odd_positive_int(
    name: str,
    value: int,
) -> None:
    """Require an odd positive integer hyperparameter."""
    validate_positive_int(
        name=name,
        value=value,
    )

    if value % 2 == 0:
        raise ValueError(
            f"{name} must be odd. Got {value}.",
        )


def validate_scope_range(
    min_name: str,
    min_value: float,
    max_name: str,
    max_value: float,
) -> None:
    """Require a ResizeMix scope interval inside (0, 1]."""
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError(
            f"{min_name} and {max_name} must be > 0. "
            f"Got {min_name}={min_value}, {max_name}={max_value}.",
        )

    if min_value > max_value:
        raise ValueError(
            f"{min_name} must be <= {max_name}. "
            f"Got {min_name}={min_value}, {max_name}={max_value}.",
        )

    if max_value > 1.0:
        raise ValueError(
            f"{max_name} must be <= 1. Got {max_value}.",
        )


def validate_num_classes(
    num_classes: int,
) -> None:
    """Require a positive number of classes."""
    validate_positive_int(
        name="num_classes",
        value=num_classes,
    )


def validate_nhwc_images(
    images: Any,
    method_name: str,
) -> None:
    """Require image batches with NHWC shape."""
    if getattr(images, "ndim", None) != 4:
        raise ValueError(
            f"{method_name} expects images with shape "
            f"(batch, height, width, channels), got {format_shape(images)}.",
        )

    batch_size, image_height, image_width, channels = images.shape

    if batch_size < 1:
        raise ValueError(
            f"{method_name} requires batch size >= 1, got {batch_size}.",
        )

    if image_height < 1 or image_width < 1:
        raise ValueError(
            f"{method_name} requires positive image height and width, "
            f"got {format_shape(images)}.",
        )

    if channels < 1:
        raise ValueError(
            f"{method_name} requires at least one image channel, "
            f"got {format_shape(images)}.",
        )


def validate_labels_match_images(
    labels: Any,
    images: Any,
    method_name: str,
) -> None:
    """Require labels to share the image batch dimension."""
    if getattr(labels, "ndim", None) < 1:
        raise ValueError(
            f"{method_name} expects labels with a batch dimension, "
            f"got {format_shape(labels)}.",
        )

    if labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"{method_name} image/label batch mismatch: "
            f"images shape {format_shape(images)}, "
            f"labels shape {format_shape(labels)}.",
        )


def validate_saliency_maps_match_images(
    saliency_maps: Any,
    images: Any,
    method_name: str,
) -> None:
    """Require saliency maps to match image batch and spatial dimensions."""
    if getattr(saliency_maps, "ndim", None) not in (3, 4):
        raise ValueError(
            f"{method_name} expects saliency maps with shape "
            f"(batch, height, width) or (batch, height, width, channels), "
            f"got {format_shape(saliency_maps)}.",
        )

    if saliency_maps.shape[0] != images.shape[0]:
        raise ValueError(
            f"{method_name} saliency/image batch mismatch: "
            f"images shape {format_shape(images)}, "
            f"saliency_maps shape {format_shape(saliency_maps)}.",
        )

    if saliency_maps.shape[1] != images.shape[1] or saliency_maps.shape[2] != images.shape[2]:
        raise ValueError(
            f"{method_name} saliency maps must match image height and width: "
            f"images shape {format_shape(images)}, "
            f"saliency_maps shape {format_shape(saliency_maps)}.",
        )


def validate_no_repeat_batch_size(
    batch_size: int,
    method_name: str,
) -> None:
    """Require enough examples to create non-self pairs."""
    if batch_size < 2:
        raise ValueError(
            f"{method_name} no_repeat requires batch size >= 2, "
            f"got {batch_size}.",
        )
