from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from allthemix.config import load_optional_yaml_config, load_yaml_config
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.methods.cutmix import normalize_cutmix_variant
from allthemix.methods.guidedmixup import (
    _compute_spectral_residual_saliency_maps,
    _gaussian_blur_2d_single_channel,
    _normalize_saliency_maps,
)
from allthemix.utils.cli import str2bool
from allthemix.visualize.mix_samples import (
    DEFAULTS,
    _as_numpy_output,
    _canonical_method_name,
    _denormalize_images,
    _force_mix_args_for_visualization,
    _load_visual_batch,
    _make_mixer,
    _parse_methods,
    _skip_reason,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the mix debug CLI parser."""
    parser = argparse.ArgumentParser(
        description="Audit one pixel-level mix batch and save visual debug files.",
    )

    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--config_root", type=str, default="")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--methods", type=str, default="config")
    parser.add_argument("--output_dir", type=str, default="./outputs/visualize/debug")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--force_mix", type=str2bool, default=None)
    parser.add_argument(
        "--saliency_fallback",
        type=str,
        choices=("sr", "none"),
        default="sr",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=None)
    parser.add_argument("--basic_aug", type=str2bool, default=None)
    parser.add_argument(
        "--aug_recipe",
        type=str,
        choices=(
            "none",
            "basic",
            "hflip",
            "horizontal_flip",
            "cub",
            "fine_grained",
            "imagenet",
            "tiny_official",
            "tiny_openmixup",
        ),
        default=None,
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse debug args and merge them with a training config."""
    parser = _build_parser()
    cli_args = parser.parse_args()
    config = (
        {}
        if cli_args.config_dir or cli_args.config_root
        else load_optional_yaml_config(cli_args.config)
    )

    values = dict(
        DEFAULTS,
    )
    values.update(
        config,
    )

    if cli_args.method:
        values["method"] = cli_args.method

    values["methods"] = [
        _canonical_method_name(
            values["method"],
        )
    ]
    if cli_args.methods:
        values["methods"] = _parse_methods(
            methods_arg=cli_args.methods,
            config_method=values["method"],
        )
    values["methods_arg"] = cli_args.methods

    for key in (
        "output_dir",
        "num_samples",
        "dpi",
        "threshold",
        "saliency_fallback",
    ):
        values[key] = getattr(
            cli_args,
            key,
        )

    if cli_args.force_mix is not None:
        values["force_mix"] = cli_args.force_mix
        cli_overrides = {
            "force_mix",
        }
    else:
        cli_overrides = set()

    if cli_args.seed is not None:
        values["seed"] = cli_args.seed
        cli_overrides.add(
            "seed",
        )

    for key in (
        "dataset",
        "data_dir",
        "batch_size",
        "shuffle_buffer_size",
        "basic_aug",
        "aug_recipe",
    ):
        value = getattr(
            cli_args,
            key,
        )
        if value is not None:
            values[key] = value
            cli_overrides.add(
                key,
            )

    if cli_args.shuffle_buffer_size is None and values.get(
        "shuffle_buffer_size",
        1,
    ) == 1:
        values["shuffle_buffer_size"] = 10000
        cli_overrides.add(
            "shuffle_buffer_size",
        )

    values["config"] = cli_args.config
    values["config_dir"] = cli_args.config_dir
    values["config_root"] = cli_args.config_root
    values["_cli_overrides"] = sorted(
        cli_overrides,
    )

    return argparse.Namespace(
        **values,
    )


def _requires_saliency_maps(
    method: str,
) -> bool:
    """Return whether a method needs saliency maps in aux_info."""
    return _canonical_method_name(
        method,
    ) in {
        "saliencymix",
        "guidedmixup",
    }


def _debug_skip_reason(
    method: str,
    args: argparse.Namespace,
) -> str | None:
    """Return skip reason, allowing debug saliency fallback when configured."""
    reason = _skip_reason(
        method=method,
        args=args,
    )
    if (
        reason is not None
        and _requires_saliency_maps(
            method,
        )
        and args.saliency_fallback != "none"
    ):
        return None

    return reason


def _debug_saliency_maps(
    images: jnp.ndarray,
    args: argparse.Namespace,
) -> jnp.ndarray:
    """Generate debug-only saliency maps for methods missing cache input."""
    if args.saliency_fallback == "sr":
        return _compute_spectral_residual_saliency_maps(
            images=images,
            blur_kernel=args.guidedmixup_blur_kernel,
        )

    raise ValueError(
        f"Unsupported saliency_fallback: {args.saliency_fallback}"
    )


def _ensure_debug_aux_info(
    method: str,
    images: jnp.ndarray,
    aux_info: dict[str, jnp.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, jnp.ndarray], str]:
    """Ensure debug saliency maps exist when a method needs them."""
    if not _requires_saliency_maps(
        method,
    ):
        return aux_info, "not_required"

    if "saliency_maps" in aux_info:
        return aux_info, "cache"

    aux_info = dict(
        aux_info,
    )
    aux_info["saliency_maps"] = _debug_saliency_maps(
        images=images,
        args=args,
    )

    return aux_info, f"debug_{args.saliency_fallback}_fallback"


def _as_flat_lambda(
    lam: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Convert scalar or per-sample lambda into a batch vector."""
    lam = np.asarray(
        lam,
        dtype=np.float32,
    )

    if lam.ndim == 0:
        return np.full(
            batch_size,
            float(lam),
            dtype=np.float32,
        )

    return lam.reshape(
        -1,
    ).astype(
        np.float32,
    )


def _infer_changed_mask(
    source_a: np.ndarray,
    mixed: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a spatial mask where mixed pixels differ from source A."""
    return np.max(
        np.abs(
            mixed - source_a,
        ),
        axis=-1,
    ) > threshold


def _mask_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return x1, y1, x2, y2 for the changed mask."""
    if not np.any(
        mask,
    ):
        return 0, 0, 0, 0

    ys, xs = np.where(
        mask,
    )
    x1 = int(
        xs.min(),
    )
    y1 = int(
        ys.min(),
    )
    x2 = int(
        xs.max() + 1,
    )
    y2 = int(
        ys.max() + 1,
    )

    return x1, y1, x2, y2


def _masked_max_abs_error(
    lhs: np.ndarray,
    rhs: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute max absolute RGB error on a spatial mask."""
    if not np.any(
        mask,
    ):
        return 0.0

    mask = mask[..., None]

    return float(
        np.max(
            np.abs(
                lhs - rhs,
            )[mask.repeat(lhs.shape[-1], axis=-1)]
        )
    )


def _base_audit_row(
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    sample_index: int,
) -> dict[str, Any]:
    """Build common audit fields for one sample."""
    label_a = int(
        labels[sample_index],
    )
    label_b = int(
        labels_b[sample_index],
    )
    expected_label_b = int(
        labels[perm[sample_index]],
    )
    lam_value = float(
        lam[sample_index],
    )

    return {
        "sample_index": sample_index,
        "perm": int(
            perm[sample_index],
        ),
        "label_a": label_a,
        "label_b": label_b,
        "expected_label_b": expected_label_b,
        "label_b_ok": label_b == expected_label_b,
        "lam": lam_value,
    }


def _max_abs_error(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Compute max absolute error between two arrays."""
    return float(
        np.max(
            np.abs(
                lhs - rhs,
            )
        )
    )


def _resize_source_to_box_nearest_np(
    source_image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Resize one source image into a box using ResizeMix nearest mapping."""
    image_height, image_width = source_image.shape[:2]
    box_width = max(
        x2 - x1,
        1,
    )
    box_height = max(
        y2 - y1,
        1,
    )

    y_positions = np.arange(
        image_height,
    )[:, None]
    x_positions = np.arange(
        image_width,
    )[None, :]
    relative_y = y_positions - y1
    relative_x = x_positions - x1
    source_y = np.floor(
        relative_y.astype(np.float32)
        * image_height
        / float(box_height)
    ).astype(
        np.int32,
    )
    source_x = np.floor(
        relative_x.astype(np.float32)
        * image_width
        / float(box_width)
    ).astype(
        np.int32,
    )
    source_y = np.clip(
        source_y,
        0,
        image_height - 1,
    )
    source_x = np.clip(
        source_x,
        0,
        image_width - 1,
    )

    return source_image[
        source_y,
        source_x,
        :,
    ]


def _audit_hard_paste_sample(
    images: np.ndarray,
    mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    sample_index: int,
    threshold: float,
    audit_type: str,
    lambda_mode: str = "area",
) -> dict[str, Any]:
    """Audit a hard mask paste where changed pixels should come from source B."""
    source_a = images[sample_index]
    source_b = images[perm[sample_index]]
    mixed = mixed_images[sample_index]
    mask = _infer_changed_mask(
        source_a=source_a,
        mixed=mixed,
        threshold=threshold,
    )

    x1, y1, x2, y2 = _mask_bbox(
        mask,
    )
    changed_ratio = float(
        np.mean(
            mask,
        )
    )
    row = _base_audit_row(
        labels=labels,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
        sample_index=sample_index,
    )
    expected_lam = (
        row["lam"]
        if lambda_mode == "sampled"
        else 1.0 - changed_ratio
    )

    inside_b_error = _masked_max_abs_error(
        lhs=mixed,
        rhs=source_b,
        mask=mask,
    )
    outside_a_error = _masked_max_abs_error(
        lhs=mixed,
        rhs=source_a,
        mask=np.logical_not(
            mask,
        ),
    )

    row.update(
        {
            "audit_type": audit_type,
            "expected_lam": expected_lam,
            "lam_abs_diff": abs(
                row["lam"] - expected_lam,
            ),
            "changed_area_ratio": changed_ratio,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "box_width": x2 - x1,
            "box_height": y2 - y1,
            "pixel_max_abs_error": max(
                inside_b_error,
                outside_a_error,
            ),
            "inside_b_max_abs_error": inside_b_error,
            "outside_a_max_abs_error": outside_a_error,
        }
    )

    return row


def _audit_mixup_sample(
    images: np.ndarray,
    mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    sample_index: int,
    threshold: float,
) -> dict[str, Any]:
    """Audit MixUp by reconstructing the linear interpolation."""
    source_a = images[sample_index]
    source_b = images[perm[sample_index]]
    mixed = mixed_images[sample_index]
    row = _base_audit_row(
        labels=labels,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
        sample_index=sample_index,
    )
    expected = row["lam"] * source_a + (
        1.0 - row["lam"]
    ) * source_b
    changed_mask = _infer_changed_mask(
        source_a=source_a,
        mixed=mixed,
        threshold=threshold,
    )
    row.update(
        {
            "audit_type": "linear_mixup",
            "expected_lam": row["lam"],
            "lam_abs_diff": 0.0,
            "changed_area_ratio": float(
                np.mean(
                    changed_mask,
                )
            ),
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "box_width": "",
            "box_height": "",
            "pixel_max_abs_error": _max_abs_error(
                mixed,
                expected,
            ),
            "inside_b_max_abs_error": "",
            "outside_a_max_abs_error": "",
        }
    )

    return row


def _audit_resizemix_sample(
    images: np.ndarray,
    mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    sample_index: int,
    threshold: float,
) -> dict[str, Any]:
    """Audit ResizeMix by reconstructing the nearest-neighbor resized paste."""
    source_a = images[sample_index]
    source_b = images[perm[sample_index]]
    mixed = mixed_images[sample_index]
    mask = _infer_changed_mask(
        source_a=source_a,
        mixed=mixed,
        threshold=threshold,
    )
    x1, y1, x2, y2 = _mask_bbox(
        mask,
    )
    box_mask = np.zeros(
        mask.shape,
        dtype=bool,
    )
    if x2 > x1 and y2 > y1:
        box_mask[y1:y2, x1:x2] = True

    resized_source = _resize_source_to_box_nearest_np(
        source_image=source_b,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    expected = np.where(
        box_mask[..., None],
        resized_source,
        source_a,
    )
    changed_ratio = float(
        np.mean(
            box_mask,
        )
    )
    expected_lam = 1.0 - changed_ratio
    row = _base_audit_row(
        labels=labels,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
        sample_index=sample_index,
    )
    row.update(
        {
            "audit_type": "resized_paste",
            "expected_lam": expected_lam,
            "lam_abs_diff": abs(
                row["lam"] - expected_lam,
            ),
            "changed_area_ratio": changed_ratio,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "box_width": x2 - x1,
            "box_height": y2 - y1,
            "pixel_max_abs_error": _max_abs_error(
                mixed,
                expected,
            ),
            "inside_b_max_abs_error": _masked_max_abs_error(
                mixed,
                resized_source,
                box_mask,
            ),
            "outside_a_max_abs_error": _masked_max_abs_error(
                mixed,
                source_a,
                np.logical_not(
                    box_mask,
                ),
            ),
        }
    )

    return row


def _guided_soft_masks(
    method: str,
    images: jnp.ndarray,
    aux_info: dict[str, jnp.ndarray],
    perm: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Rebuild GuidedMixup/Guided-SR per-pixel soft masks."""
    method = _canonical_method_name(
        method,
    )

    if method in {
        "guided_sr",
        "guidedmixup_sr",
    }:
        saliency_maps = _compute_spectral_residual_saliency_maps(
            images=images,
        )
    else:
        saliency_maps = aux_info["saliency_maps"]

    saliency_maps = _normalize_saliency_maps(
        saliency_maps,
    )
    saliency_maps = _gaussian_blur_2d_single_channel(
        saliency_maps,
        kernel_size=args.guidedmixup_blur_kernel,
    )
    saliency_maps = _normalize_saliency_maps(
        saliency_maps,
    )
    paired_saliency_maps = saliency_maps[
        jnp.asarray(
            perm,
        )
    ]

    return np.asarray(
        saliency_maps
        / (
            saliency_maps
            + paired_saliency_maps
            + 1e-8
        )
    )


def _audit_soft_mask_sample(
    images: np.ndarray,
    mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    soft_masks: np.ndarray,
    sample_index: int,
    threshold: float,
) -> dict[str, Any]:
    """Audit per-pixel soft masks such as Guided-SR."""
    source_a = images[sample_index]
    source_b = images[perm[sample_index]]
    mixed = mixed_images[sample_index]
    soft_mask = soft_masks[sample_index]
    expected = soft_mask * source_a + (
        1.0 - soft_mask
    ) * source_b
    expected_lam = float(
        np.mean(
            soft_mask,
        )
    )
    changed_mask = _infer_changed_mask(
        source_a=source_a,
        mixed=mixed,
        threshold=threshold,
    )
    row = _base_audit_row(
        labels=labels,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
        sample_index=sample_index,
    )
    row.update(
        {
            "audit_type": "soft_saliency_mix",
            "expected_lam": expected_lam,
            "lam_abs_diff": abs(
                row["lam"] - expected_lam,
            ),
            "changed_area_ratio": float(
                np.mean(
                    changed_mask,
                )
            ),
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "box_width": "",
            "box_height": "",
            "pixel_max_abs_error": _max_abs_error(
                mixed,
                expected,
            ),
            "inside_b_max_abs_error": "",
            "outside_a_max_abs_error": "",
        }
    )

    return row


def _audit_feature_metadata_sample(
    images: np.ndarray,
    mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    sample_index: int,
) -> dict[str, Any]:
    """Audit feature-level methods whose input image should remain unchanged."""
    row = _base_audit_row(
        labels=labels,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
        sample_index=sample_index,
    )
    row.update(
        {
            "audit_type": "feature_mix_metadata",
            "expected_lam": row["lam"],
            "lam_abs_diff": 0.0,
            "changed_area_ratio": 0.0,
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "box_width": "",
            "box_height": "",
            "pixel_max_abs_error": _max_abs_error(
                mixed_images[sample_index],
                images[sample_index],
            ),
            "inside_b_max_abs_error": "",
            "outside_a_max_abs_error": "",
        }
    )

    return row


def _audit_sample_by_method(
    method: str,
    images: np.ndarray,
    images_jax: jnp.ndarray,
    mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    lam: np.ndarray,
    perm: np.ndarray,
    aux_info: dict[str, jnp.ndarray],
    args: argparse.Namespace,
    output: dict[str, np.ndarray],
    sample_index: int,
    threshold: float,
) -> dict[str, Any]:
    """Dispatch one sample through the method-specific audit formula."""
    method = _canonical_method_name(
        method,
    )

    if method == "baseline":
        return _audit_feature_metadata_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            sample_index=sample_index,
        )

    if method == "mixup":
        return _audit_mixup_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            sample_index=sample_index,
            threshold=threshold,
        )

    if method in {
        "cutmix",
        "cutmix_sumix",
        "saliencymix",
    }:
        lambda_mode = (
            "sampled"
            if method in {
                "cutmix",
                "cutmix_sumix",
            }
            and str(
                normalize_cutmix_variant(
                    getattr(
                        args,
                        "cutmix_variant",
                        "standard",
                    )
                )
            )
            in {
                "torchbearer",
                "fmix_repo",
            }
            else "area"
        )
        return _audit_hard_paste_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            sample_index=sample_index,
            threshold=threshold,
            audit_type="hard_paste",
            lambda_mode=lambda_mode,
        )

    if method == "fmix":
        return _audit_hard_paste_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            sample_index=sample_index,
            threshold=threshold,
            audit_type="low_frequency_mask",
        )

    if method == "resizemix":
        return _audit_resizemix_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            sample_index=sample_index,
            threshold=threshold,
        )

    if method in {
        "guidedmixup",
        "guided_sr",
        "guidedmixup_sr",
    }:
        soft_masks = _guided_soft_masks(
            method=method,
            images=images_jax,
            aux_info=aux_info,
            perm=perm,
            args=args,
        )
        return _audit_soft_mask_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            soft_masks=soft_masks,
            sample_index=sample_index,
            threshold=threshold,
        )

    if method in {
        "catchupmix",
        "catchup_mix",
        "catch_up_mix",
    }:
        if int(
            np.asarray(
                output.get(
                    "layer",
                    1,
                )
            )
        ) == 0:
            return _audit_hard_paste_sample(
                images=images,
                mixed_images=mixed_images,
                labels=labels,
                labels_b=labels_b,
                lam=lam,
                perm=perm,
                sample_index=sample_index,
                threshold=threshold,
                audit_type="catchup_input_cutmix",
            )

        return _audit_feature_metadata_sample(
            images=images,
            mixed_images=mixed_images,
            labels=labels,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
            sample_index=sample_index,
        )

    raise ValueError(
        f"Unsupported debug method: {method}"
    )


def _row_mask_for_display(
    row: dict[str, Any],
    source_a: np.ndarray,
    mixed: np.ndarray,
    soft_mask: np.ndarray | None,
    threshold: float,
) -> np.ndarray:
    """Return the third-column map for visualization."""
    if soft_mask is not None:
        value = soft_mask
        if value.ndim == 3:
            value = value[..., 0]
        return value

    return _infer_changed_mask(
        source_a=source_a,
        mixed=mixed,
        threshold=threshold,
    ).astype(
        np.float32,
    )


def _legacy_row_keys() -> tuple[str, ...]:
    """Keep historical CSV columns readable when users open old outputs."""
    return (
        "expected_lam_from_changed_area",
        "lam_area_abs_diff",
    )


def _normalize_legacy_fields(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Populate old lambda column names for compatibility."""
    row = dict(
        row,
    )
    row["expected_lam_from_changed_area"] = row.get(
        "expected_lam",
        "",
    )
    row["lam_area_abs_diff"] = row.get(
        "lam_abs_diff",
        "",
    )

    return row


def _draw_bbox(
    axis,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    """Draw a rectangle on an axis when a non-empty box exists."""
    if x2 <= x1 or y2 <= y1:
        return

    rectangle = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor="lime",
        linewidth=1.5,
    )
    axis.add_patch(
        rectangle,
    )


def _save_debug_grid(
    method: str,
    display_images: np.ndarray,
    display_mixed: np.ndarray,
    masks: list[np.ndarray],
    rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    """Save source, partner, mask, and mixed debug grid."""
    num_samples = len(
        rows,
    )
    figure, axes = plt.subplots(
        num_samples,
        4,
        figsize=(
            8,
            max(
                2.0,
                2.0 * num_samples,
            ),
        ),
        squeeze=False,
    )
    figure.suptitle(
        f"{method} debug",
        fontsize=12,
    )

    for column, title in enumerate(
        (
            "source A",
            "source B",
            "changed mask",
            "mixed",
        )
    ):
        axes[0, column].set_title(
            title,
            fontsize=8,
        )

    for row_index, row in enumerate(
        rows,
    ):
        sample_index = int(
            row["sample_index"],
        )
        perm_index = int(
            row["perm"],
        )

        axes[row_index, 0].imshow(
            display_images[sample_index],
        )
        axes[row_index, 0].set_ylabel(
            f"y={row['label_a']}",
            fontsize=7,
        )

        axes[row_index, 1].imshow(
            display_images[perm_index],
        )
        axes[row_index, 1].set_ylabel(
            f"y={row['label_b']}",
            fontsize=7,
        )

        axes[row_index, 2].imshow(
            masks[row_index],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 3].imshow(
            display_mixed[sample_index],
        )
        if row.get(
            "x1",
            "",
        ) != "":
            _draw_bbox(
                axis=axes[row_index, 3],
                x1=int(row["x1"]),
                y1=int(row["y1"]),
                x2=int(row["x2"]),
                y2=int(row["y2"]),
            )
        axes[row_index, 3].set_xlabel(
            f"lam={float(row['lam']):.3f}, err={float(row['pixel_max_abs_error']):.2g}",
            fontsize=7,
        )

        for column_index in range(
            4,
        ):
            axes[row_index, column_index].set_xticks(
                [],
            )
            axes[row_index, column_index].set_yticks(
                [],
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(
        figure,
    )


def _write_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write debug audit rows to CSV."""
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key,
                )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows,
        )


def _summarize_rows(
    rows: list[dict[str, Any]],
    method: str,
    image_path: Path | None,
    csv_path: Path | None,
    status: str = "ok",
    reason: str = "",
) -> dict[str, Any]:
    """Summarize per-sample audit rows into one method-level row."""
    if not rows:
        return {
            "method": method,
            "status": status,
            "passed": False,
            "reason": reason,
            "saliency_source": "",
            "num_samples": 0,
            "all_labels_ok": False,
            "max_pixel_error": "",
            "max_lam_error": "",
            "max_inside_b_error": "",
            "max_outside_a_error": "",
            "image_path": str(image_path) if image_path else "",
            "csv_path": str(csv_path) if csv_path else "",
        }

    inside_values = [
        float(row["inside_b_max_abs_error"])
        for row in rows
        if row.get("inside_b_max_abs_error", "") != ""
    ]
    outside_values = [
        float(row["outside_a_max_abs_error"])
        for row in rows
        if row.get("outside_a_max_abs_error", "") != ""
    ]

    max_pixel_error = max(
        float(row.get("pixel_max_abs_error", 0.0) or 0.0)
        for row in rows
    )
    max_lam_error = max(
        float(row.get("lam_abs_diff", 0.0) or 0.0)
        for row in rows
    )
    all_labels_ok = all(
        bool(row.get("label_b_ok", False))
        for row in rows
    )
    pixel_tolerance = 1e-4
    saliency_sources = sorted(
        {
            str(
                row.get(
                    "saliency_source",
                    "",
                )
            )
            for row in rows
            if row.get(
                "saliency_source",
                "",
            )
            != ""
        }
    )

    return {
        "method": method,
        "status": status,
        "passed": status == "ok"
        and all_labels_ok
        and max_pixel_error <= pixel_tolerance,
        "reason": reason,
        "saliency_source": "|".join(
            saliency_sources,
        ),
        "num_samples": len(rows),
        "all_labels_ok": all_labels_ok,
        "max_pixel_error": max_pixel_error,
        "max_lam_error": max_lam_error,
        "max_inside_b_error": max(inside_values) if inside_values else "",
        "max_outside_a_error": max(outside_values) if outside_values else "",
        "image_path": str(image_path) if image_path else "",
        "csv_path": str(csv_path) if csv_path else "",
    }


def debug_mix_batch(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the configured mixer once and save visual/CSV consistency checks."""
    method = _canonical_method_name(
        args.method,
    )
    reason = _debug_skip_reason(
        method=method,
        args=args,
    )
    if reason is not None:
        print(
            f"Skipping {method}: {reason}"
        )
        return _summarize_rows(
            rows=[],
            method=method,
            image_path=None,
            csv_path=None,
            status="skipped",
            reason=reason,
        )

    metadata = get_metadata(
        args.dataset,
    )
    output_dir = Path(
        args.output_dir,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_np, labels_np, aux_np = _load_visual_batch(
        args,
    )
    images = jnp.asarray(
        images_np,
    )
    labels = jnp.asarray(
        labels_np,
    )
    aux_info = {
        key: jnp.asarray(
            value,
        )
        for key, value in aux_np.items()
    }

    method_args = _force_mix_args_for_visualization(
        method=method,
        args=args,
    )
    aux_info, saliency_source = _ensure_debug_aux_info(
        method=method,
        images=images,
        aux_info=aux_info,
        args=method_args,
    )
    rng = jax.random.PRNGKey(
        method_args.seed,
    )
    mixer = _make_mixer(
        method=method,
        args=method_args,
        num_classes=metadata.num_classes,
    )
    mixer_output = mixer(
        rng=rng,
        images=images,
        labels=labels,
        aux_info=aux_info,
    )
    output = _as_numpy_output(
        mixer_output=mixer_output,
        labels=labels,
    )
    perm = output["perm"]
    lam = _as_flat_lambda(
        output["lam"],
        batch_size=images_np.shape[0],
    )

    num_samples = min(
        args.num_samples,
        images_np.shape[0],
    )
    soft_masks = None
    if method in {
        "guidedmixup",
        "guided_sr",
        "guidedmixup_sr",
    }:
        soft_masks = _guided_soft_masks(
            method=method,
            images=images,
            aux_info=aux_info,
            perm=perm,
            args=method_args,
        )

    rows: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for sample_index in range(
        num_samples,
    ):
        row = _audit_sample_by_method(
            method=method,
            images=images_np,
            images_jax=images,
            mixed_images=output["images"],
            labels=labels_np,
            labels_b=output["labels_b"],
            lam=lam,
            perm=perm,
            aux_info=aux_info,
            args=method_args,
            output=output,
            sample_index=sample_index,
            threshold=args.threshold,
        )
        row = _normalize_legacy_fields(
            row,
        )
        row["saliency_source"] = saliency_source
        rows.append(
            row,
        )
        masks.append(
            _row_mask_for_display(
                row=row,
                source_a=images_np[sample_index],
                mixed=output["images"][sample_index],
                soft_mask=(
                    None
                    if soft_masks is None
                    else soft_masks[sample_index]
                ),
                threshold=args.threshold,
            )
        )

    display_images = _denormalize_images(
        images_np,
        dataset=args.dataset,
    )
    display_mixed = _denormalize_images(
        output["images"],
        dataset=args.dataset,
    )

    output_prefix = getattr(
        args,
        "output_prefix",
        method,
    )
    image_path = output_dir / f"{output_prefix}_debug.png"
    csv_path = output_dir / f"{output_prefix}_debug.csv"
    _save_debug_grid(
        method=method,
        display_images=display_images,
        display_mixed=display_mixed,
        masks=masks,
        rows=rows,
        output_path=image_path,
        dpi=args.dpi,
    )
    _write_csv(
        rows=rows,
        output_path=csv_path,
    )

    summary = _summarize_rows(
        rows=rows,
        method=method,
        image_path=image_path,
        csv_path=csv_path,
    )

    print(
        f"Saved debug image: {image_path}"
    )
    print(
        f"Saved debug CSV: {csv_path}"
    )
    print(
        "Summary: "
        f"label_b_ok={summary['all_labels_ok']}, "
        f"max_lam_abs_diff={float(summary['max_lam_error']):.6f}, "
        f"max_pixel_error={float(summary['max_pixel_error']):.6f}"
    )

    return summary


def _config_to_args(
    base_args: argparse.Namespace,
    config_path: Path,
    method: str | None = None,
) -> argparse.Namespace:
    """Build one debug namespace from a config file and global debug overrides."""
    config = load_yaml_config(config_path)
    values = dict(
        DEFAULTS,
    )
    values.update(
        config,
    )

    if method is not None:
        values["method"] = method

    for key in (
        "output_dir",
        "num_samples",
        "dpi",
        "threshold",
        "saliency_fallback",
        "force_mix",
    ):
        if hasattr(
            base_args,
            key,
        ):
            values[key] = getattr(
                base_args,
                key,
            )

    cli_overrides = set(
        getattr(
            base_args,
            "_cli_overrides",
            [],
        )
    )
    for key in (
        "seed",
        "dataset",
        "data_dir",
        "batch_size",
        "shuffle_buffer_size",
        "basic_aug",
        "aug_recipe",
        "saliency_dir",
        "sal_basic_aug",
        "sal_aug_recipe",
    ):
        if key in cli_overrides and hasattr(
            base_args,
            key,
        ):
            values[key] = getattr(
                base_args,
                key,
            )

    values["config"] = str(
        config_path,
    )
    values["methods"] = [
        _canonical_method_name(
            values["method"],
        )
    ]
    values["output_prefix"] = (
        f"{values.get('dataset', 'dataset')}_"
        f"{values.get('model', 'model')}_"
        f"{config_path.stem}"
    )
    if _canonical_method_name(values["method"]) != _canonical_method_name(
        config.get(
            "method",
            values["method"],
        )
    ):
        values["output_prefix"] = (
            f"{values['output_prefix']}_"
            f"{_canonical_method_name(values['method'])}"
        )

    return argparse.Namespace(
        **values,
    )


def _single_method_args(
    base_args: argparse.Namespace,
    method: str,
) -> argparse.Namespace:
    """Build a method-specific namespace for one loaded config."""
    values = vars(
        base_args,
    ).copy()
    values["method"] = method
    values["methods"] = [
        method,
    ]
    values["output_prefix"] = (
        f"{values.get('dataset', 'dataset')}_"
        f"{values.get('model', 'model')}_"
        f"{method}"
    )

    return argparse.Namespace(
        **values,
    )


def _candidate_configs(
    config_dir: str,
) -> list[Path]:
    """Return sorted YAML configs under a directory."""
    path = Path(
        config_dir,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Config directory not found: {path}"
        )

    configs = sorted(
        list(
            path.glob(
                "*.yaml",
            )
        )
        + list(
            path.glob(
                "*.yml",
            )
        )
    )
    if not configs:
        raise FileNotFoundError(
            f"No YAML configs found in: {path}"
        )

    return configs


def _candidate_configs_recursive(
    config_root: str,
) -> list[Path]:
    """Return sorted YAML configs under a root tree."""
    path = Path(
        config_root,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Config root not found: {path}"
        )

    configs = sorted(
        list(
            path.rglob(
                "*.yaml",
            )
        )
        + list(
            path.rglob(
                "*.yml",
            )
        )
    )
    if not configs:
        raise FileNotFoundError(
            f"No YAML configs found under: {path}"
        )

    return configs


def run_debug_suite(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run one or many mix debug audits and write a suite summary CSV."""
    output_dir = Path(
        args.output_dir,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict[str, Any]] = []

    if args.config_dir or args.config_root:
        config_paths = (
            _candidate_configs_recursive(
                args.config_root,
            )
            if args.config_root
            else _candidate_configs(
                args.config_dir,
            )
        )
        for config_path in config_paths:
            config = load_yaml_config(config_path)
            config_method = _canonical_method_name(
                config.get(
                    "method",
                    "baseline",
                )
            )
            methods = (
                [config_method]
                if args.methods_arg == "config"
                else _parse_methods(
                    methods_arg=args.methods_arg,
                    config_method=config_method,
                )
            )
            for method in methods:
                method_args = _config_to_args(
                    base_args=args,
                    config_path=config_path,
                    method=method,
                )
                try:
                    summary = debug_mix_batch(
                        method_args,
                    )
                except Exception as exc:  # noqa: BLE001 - debug suite should report all failures.
                    summary = _summarize_rows(
                        rows=[],
                        method=method,
                        image_path=None,
                        csv_path=None,
                        status="error",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    print(
                        f"Error in {config_path} ({method}): {summary['reason']}"
                    )
                summary["config"] = str(
                    config_path,
                )
                summaries.append(
                    summary,
                )

    else:
        for method in args.methods:
            method_args = _single_method_args(
                base_args=args,
                method=method,
            )
            try:
                summary = debug_mix_batch(
                    method_args,
                )
            except Exception as exc:  # noqa: BLE001 - debug suite should report all failures.
                summary = _summarize_rows(
                    rows=[],
                    method=method,
                    image_path=None,
                    csv_path=None,
                    status="error",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                print(
                    f"Error in {method}: {summary['reason']}"
                )
            summary["config"] = args.config
            summaries.append(
                summary,
            )

    summary_path = output_dir / "summary.csv"
    _write_csv(
        rows=summaries,
        output_path=summary_path,
    )
    print(
        f"Saved summary: {summary_path}"
    )

    return summaries


def main() -> None:
    """Run the mix debug CLI."""
    args = parse_args()
    run_debug_suite(
        args,
    )


if __name__ == "__main__":
    main()
