from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from allthemix.config import load_optional_yaml_config
from allthemix.data.datasets.tiny_imagenet import is_tiny_imagenet_name
from allthemix.data.pipeline import build_dataset_pipeline
from allthemix.data.preprocessors import cifar, tfds_image, tiny_imagenet
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.data.saliency import get_train_saliency_path
from allthemix.data.salmix_pipeline import build_salmix_dataset_pipeline
from allthemix.methods.guidedmixup import (
    _compute_spectral_residual_saliency_maps,
)
from allthemix.methods.selector import get_mixer
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.utils.cli import str2bool

DEFAULTS: dict[str, Any] = {
    "dataset": "cifar10",
    "data_dir": "./data",
    "method": "baseline",
    "batch_size": 16,
    "shuffle_buffer_size": 10_000,
    "basic_aug": False,
    "aug_recipe": "none",
    "sal_basic_aug": False,
    "sal_aug_recipe": "none",
    "saliency_dir": "./data",
    "validation_split": 0.0,
    "val_source": "train",
    "seed": 0,
    "force_mix": True,
    "mixup_alpha": 1.0,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 1.0,
    "cutmix_no_repeat": False,
    "cutmix_per_sample_lam": False,
    "cutmix_min_lam": 0.0,
    "cutmix_variant": "standard",
    "saliencymix_alpha": 1.0,
    "saliencymix_prob": 1.0,
    "saliencymix_per_sample": False,
    "fmix_alpha": 1.0,
    "fmix_decay": 3.0,
    "fmix_prob": 1.0,
    "fmix_per_sample": False,
    "fmix_no_repeat": False,
    "resizemix_scope_min": 0.1,
    "resizemix_scope_max": 0.8,
    "resizemix_prob": 1.0,
    "resizemix_per_sample": False,
    "guidedmixup_alpha": 1.0,
    "guidedmixup_prob": 1.0,
    "guidedmixup_blur_kernel": 7,
    "guidedmixup_condition": "greedy",
    "catchupmix_alpha": 1.0,
    "catchupmix_cutmix_alpha": 1.0,
    "catchupmix_num_layers": 5,
    "catchupmix_no_repeat": False,
}

VISUALIZABLE_METHODS = (
    "baseline",
    "mixup",
    "cutmix",
    "cutmix_sumix",
    "fmix",
    "resizemix",
    "saliencymix",
    "guided_sr",
)

PRECOMPUTED_SALIENCY_METHODS = {
    "saliencymix",
    "guidedmixup",
}

FEATURE_LEVEL_METHODS = {
    "catchupmix",
    "catchup_mix",
    "catch_up_mix",
}


def _canonical_method_name(
    method: str,
) -> str:
    """Normalize method names to the internal selector spelling."""
    return normalize_method_name(method)


def _parse_methods(
    methods_arg: str,
    config_method: str,
) -> list[str]:
    """Resolve the requested visualization method list."""
    methods_arg = methods_arg.strip()

    if methods_arg == "config":
        return [
            _canonical_method_name(
                config_method,
            )
        ]

    if methods_arg == "all":
        return list(
            VISUALIZABLE_METHODS,
        )

    methods = [
        _canonical_method_name(
            method,
        )
        for method in methods_arg.split(",")
        if method.strip()
    ]

    if not methods:
        raise ValueError(
            "--methods must be config, all, or a comma-separated method list."
        )

    return methods


def _build_parser() -> argparse.ArgumentParser:
    """Build the visualization CLI parser."""
    parser = argparse.ArgumentParser(
        description="Visualize generated mix samples for AllTheMix methods.",
    )

    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--methods", type=str, default="config")
    parser.add_argument("--output_dir", type=str, default="./outputs/visualize")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=160)

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
    parser.add_argument("--sal_basic_aug", type=str2bool, default=None)
    parser.add_argument(
        "--sal_aug_recipe",
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
    parser.add_argument("--saliency_dir", type=str, default=None)
    parser.add_argument("--validation_split", type=float, default=None)
    parser.add_argument(
        "--val_source",
        type=str,
        choices=("train", "test"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force_mix", type=str2bool, default=None)

    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument("--cutmix_alpha", type=float, default=None)
    parser.add_argument("--cutmix_prob", type=float, default=None)
    parser.add_argument("--cutmix_no_repeat", type=str2bool, default=None)
    parser.add_argument("--cutmix_per_sample_lam", type=str2bool, default=None)
    parser.add_argument("--cutmix_min_lam", type=float, default=None)
    parser.add_argument(
        "--cutmix_variant",
        type=str,
        choices=(
            "standard",
            "area_adjusted",
            "torchbearer",
            "fmix_repo",
            "torchbearer_area",
            "fmix_repo_area",
            "torchbearer_inside",
        ),
        default=None,
    )
    parser.add_argument("--saliencymix_alpha", type=float, default=None)
    parser.add_argument("--saliencymix_prob", type=float, default=None)
    parser.add_argument("--saliencymix_per_sample", type=str2bool, default=None)
    parser.add_argument("--fmix_alpha", type=float, default=None)
    parser.add_argument("--fmix_decay", type=float, default=None)
    parser.add_argument("--fmix_prob", type=float, default=None)
    parser.add_argument("--fmix_per_sample", type=str2bool, default=None)
    parser.add_argument("--fmix_no_repeat", type=str2bool, default=None)
    parser.add_argument("--resizemix_scope_min", type=float, default=None)
    parser.add_argument("--resizemix_scope_max", type=float, default=None)
    parser.add_argument("--resizemix_prob", type=float, default=None)
    parser.add_argument("--resizemix_per_sample", type=str2bool, default=None)
    parser.add_argument("--guidedmixup_alpha", type=float, default=None)
    parser.add_argument("--guidedmixup_prob", type=float, default=None)
    parser.add_argument("--guidedmixup_blur_kernel", type=int, default=None)
    parser.add_argument(
        "--guidedmixup_condition",
        type=str,
        choices=("random", "greedy"),
        default=None,
    )
    parser.add_argument("--catchupmix_alpha", type=float, default=None)
    parser.add_argument("--catchupmix_cutmix_alpha", type=float, default=None)
    parser.add_argument("--catchupmix_num_layers", type=int, default=None)
    parser.add_argument("--catchupmix_no_repeat", type=str2bool, default=None)

    return parser


def parse_args() -> argparse.Namespace:
    """Parse visualization args with optional training-config defaults."""
    parser = _build_parser()
    cli_args = parser.parse_args()
    config = load_optional_yaml_config(cli_args.config)

    values = dict(
        DEFAULTS,
    )
    values.update(
        config,
    )

    for key, value in vars(
        cli_args,
    ).items():
        if value is not None and key not in {
            "config",
        }:
            values[key] = value

    values["config"] = cli_args.config
    values["methods"] = _parse_methods(
        methods_arg=values["methods"],
        config_method=values["method"],
    )

    return argparse.Namespace(
        **values,
    )


def _unpack_batch(
    batch: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert a dataset batch into images, labels, and optional aux arrays."""
    if isinstance(
        batch,
        dict,
    ):
        images = batch["images"]
        labels = batch["labels"]
        aux_info = {
            key: value
            for key, value in batch.items()
            if key not in {
                "images",
                "labels",
            }
        }

    elif len(
        batch,
    ) == 2:
        images, labels = batch
        aux_info = {}

    elif len(
        batch,
    ) == 3:
        images, labels, third = batch
        if isinstance(
            third,
            dict,
        ):
            aux_info = third
        else:
            aux_info = {
                "saliency_maps": third,
            }

    else:
        raise ValueError(
            "Unsupported batch format for visualization."
        )

    images = np.asarray(
        images,
    )
    labels = np.asarray(
        labels,
    )
    aux_info = {
        key: np.asarray(
            value,
        )
        for key, value in aux_info.items()
    }

    return images, labels, aux_info


def _needs_precomputed_saliency(
    methods: list[str],
) -> bool:
    """Return whether any requested method needs cached saliency maps."""
    return any(
        method in PRECOMPUTED_SALIENCY_METHODS
        for method in methods
    )


def _saliency_cache_exists(
    dataset: str,
    saliency_dir: str,
) -> bool:
    """Return whether the configured train saliency cache exists."""
    return get_train_saliency_path(
        dataset_name=dataset,
        saliency_dir=saliency_dir,
    ).exists()


def _load_visual_batch(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load one deterministic-ish train batch for visualization."""
    tf.random.set_seed(
        args.seed,
    )

    use_saliency_pipeline = (
        _needs_precomputed_saliency(
            args.methods,
        )
        and _saliency_cache_exists(
            dataset=args.dataset,
            saliency_dir=args.saliency_dir,
        )
    )

    if use_saliency_pipeline:
        train_ds, _ = build_salmix_dataset_pipeline(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            drop_remainder=True,
            use_sal_basic_augmentation=args.sal_basic_aug,
            saliency_dir=args.saliency_dir,
            saliency_augmentation_recipe=args.sal_aug_recipe,
            validation_split=args.validation_split,
            seed=args.seed,
            deterministic_data=True,
            val_source=args.val_source,
        )

    else:
        train_ds, _ = build_dataset_pipeline(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            drop_remainder=True,
            use_basic_augmentation=args.basic_aug,
            augmentation_recipe=args.aug_recipe,
            validation_split=args.validation_split,
            seed=args.seed,
            deterministic_data=True,
            val_source=args.val_source,
        )

    batch = next(
        iter(
            train_ds,
        )
    )

    return _unpack_batch(
        batch,
    )


def _normalization_stats(
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dataset channel mean and std as NumPy arrays."""
    dataset_name = dataset.lower()

    if dataset_name in {
        "cifar10",
        "cifar100",
    }:
        mean, std = cifar.get_normalization_stats(
            dataset_name,
        )
    elif is_tiny_imagenet_name(
        dataset_name,
    ):
        mean, std = tiny_imagenet.get_normalization_stats()
    elif tfds_image.is_supported_tfds_image_dataset(
        dataset_name,
    ):
        mean, std = tfds_image.get_normalization_stats(
            dataset_name,
        )
    else:
        raise ValueError(
            f"Unsupported dataset normalization: {dataset}"
        )

    return (
        np.asarray(
            mean,
            dtype=np.float32,
        ),
        np.asarray(
            std,
            dtype=np.float32,
        ),
    )


def _denormalize_images(
    images: np.ndarray,
    dataset: str,
) -> np.ndarray:
    """Convert normalized images back to displayable [0, 1] RGB."""
    mean, std = _normalization_stats(
        dataset,
    )

    images = images * std.reshape(
        1,
        1,
        1,
        -1,
    ) + mean.reshape(
        1,
        1,
        1,
        -1,
    )

    return np.clip(
        images,
        0.0,
        1.0,
    )


def _as_numpy_output(
    mixer_output: Any,
    labels: jnp.ndarray,
) -> dict[str, np.ndarray]:
    """Normalize different mixer output shapes into a dict."""
    if hasattr(
        mixer_output,
        "images",
    ):
        images = mixer_output.images
        labels_a = mixer_output.labels_a
        labels_b = mixer_output.labels_b
        lam = mixer_output.lam
        perm = getattr(
            mixer_output,
            "perm",
            None,
        )
        layer = getattr(
            mixer_output,
            "layer",
            None,
        )

    elif len(
        mixer_output,
    ) == 2:
        images, labels_a = mixer_output
        labels_b = labels_a
        lam = jnp.ones(
            (
                labels.shape[0],
            ),
            dtype=images.dtype,
        )
        perm = jnp.arange(
            labels.shape[0],
        )
        layer = None

    else:
        images, labels_a, labels_b, lam = mixer_output
        perm = None
        layer = None

    result = {
        "images": np.asarray(
            images,
        ),
        "labels_a": np.asarray(
            labels_a,
        ),
        "labels_b": np.asarray(
            labels_b,
        ),
        "lam": np.asarray(
            lam,
        ),
    }

    if perm is not None:
        result["perm"] = np.asarray(
            perm,
            dtype=np.int32,
        )

    if layer is not None:
        result["layer"] = np.asarray(
            layer,
        )

    return result


def _get_saliency_for_display(
    method: str,
    images: jnp.ndarray,
    aux_info: dict[str, jnp.ndarray],
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Return per-sample saliency maps when a method naturally has them."""
    method = _canonical_method_name(
        method,
    )

    if method in {
        "saliencymix",
        "guidedmixup",
    } and "saliency_maps" in aux_info:
        return np.asarray(
            aux_info["saliency_maps"],
        )

    if method in {
        "guided_sr",
        "guidedmixup_sr",
    }:
        return np.asarray(
            _compute_spectral_residual_saliency_maps(
                images=images,
                blur_kernel=args.guidedmixup_blur_kernel,
            )
        )

    return None


def _normalize_map(
    value_map: np.ndarray,
) -> np.ndarray:
    """Normalize one spatial map to [0, 1] for display."""
    value_map = np.asarray(
        value_map,
        dtype=np.float32,
    )

    if value_map.ndim == 3:
        value_map = value_map[..., 0]

    value_min = float(
        np.min(
            value_map,
        )
    )
    value_max = float(
        np.max(
            value_map,
        )
    )

    if value_max <= value_min:
        return np.zeros_like(
            value_map,
        )

    return (
        value_map
        - value_min
    ) / (
        value_max
        - value_min
    )


def _sample_lambda(
    lam: np.ndarray,
    sample_index: int,
) -> float:
    """Return scalar lambda for a displayed sample."""
    if lam.ndim == 0:
        return float(
            lam,
        )

    return float(
        lam.reshape(
            -1,
        )[sample_index]
    )


def _save_method_grid(
    method: str,
    images: np.ndarray,
    mixed_images: np.ndarray,
    raw_images: np.ndarray,
    raw_mixed_images: np.ndarray,
    labels: np.ndarray,
    labels_b: np.ndarray,
    perm: np.ndarray,
    lam: np.ndarray,
    saliency_maps: np.ndarray | None,
    output_path: Path,
    num_samples: int,
    dpi: int,
    layer: np.ndarray | None = None,
) -> None:
    """Save a source/partner/footprint/mixed grid for one method."""
    method_name = _canonical_method_name(
        method,
    )
    num_samples = min(
        num_samples,
        images.shape[0],
    )

    display_images = images[:num_samples]
    display_mixed = mixed_images[:num_samples]
    delta_source = raw_images[:num_samples]
    delta_mixed = raw_mixed_images[:num_samples]
    partner_images = images[perm[:num_samples]]
    mask_methods = {
        "cutmix",
        "cutmix_sumix",
        "fmix",
        "resizemix",
    }
    delta_maps = np.mean(
        np.abs(
            display_mixed
            - display_images,
        ),
        axis=-1,
    )
    raw_delta_maps = np.mean(
        np.abs(
            delta_mixed
            - delta_source,
        ),
        axis=-1,
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

    layer_suffix = ""
    if layer is not None:
        layer_suffix = f" | layer={int(np.asarray(layer))}"

    figure.suptitle(
        f"{method}{layer_suffix}",
        fontsize=12,
    )

    footprint_title = "change"
    if saliency_maps is not None:
        footprint_title = "saliency"
    elif method_name == "mixup":
        footprint_title = "source B weight"
    elif method_name in mask_methods:
        footprint_title = "source B mask"

    column_titles = [
        "source A",
        "source B",
        footprint_title,
        "mixed",
    ]

    for column_index, title in enumerate(
        column_titles,
    ):
        axes[0, column_index].set_title(
            title,
            fontsize=8,
        )

    for row_index in range(
        num_samples,
    ):
        labels_a_value = int(
            labels[row_index],
        )
        labels_b_value = int(
            labels_b[row_index],
        )
        lam_value = _sample_lambda(
            lam,
            row_index,
        )

        axes[row_index, 0].imshow(
            display_images[row_index],
        )
        axes[row_index, 0].set_ylabel(
            f"y={labels_a_value}",
            fontsize=7,
        )

        axes[row_index, 1].imshow(
            partner_images[row_index],
        )
        axes[row_index, 1].set_ylabel(
            f"y={labels_b_value}",
            fontsize=7,
        )

        if saliency_maps is not None:
            map_to_show = _normalize_map(
                saliency_maps[row_index],
            )
        elif method_name == "mixup":
            map_to_show = np.full_like(
                delta_maps[row_index],
                fill_value=1.0 - lam_value,
                dtype=np.float32,
            )
        elif method_name in mask_methods:
            map_to_show = (
                raw_delta_maps[row_index] > 1e-6
            ).astype(
                np.float32,
            )
        else:
            map_to_show = _normalize_map(
                delta_maps[row_index],
            )

        axes[row_index, 2].imshow(
            map_to_show,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )

        axes[row_index, 3].imshow(
            display_mixed[row_index],
        )
        axes[row_index, 3].set_xlabel(
            f"lam={lam_value:.3f}",
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


def _make_mixer(
    method: str,
    args: argparse.Namespace,
    num_classes: int,
):
    """Create a mixer using the same selector path as training."""
    return get_mixer(
        name=method,
        num_classes=num_classes,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        cutmix_no_repeat=args.cutmix_no_repeat,
        cutmix_variant=args.cutmix_variant,
        cutmix_per_sample_lam=args.cutmix_per_sample_lam,
        cutmix_min_lam=args.cutmix_min_lam,
        saliencymix_alpha=args.saliencymix_alpha,
        saliencymix_prob=args.saliencymix_prob,
        saliencymix_per_sample=args.saliencymix_per_sample,
        fmix_alpha=args.fmix_alpha,
        fmix_decay=args.fmix_decay,
        fmix_prob=args.fmix_prob,
        fmix_per_sample=args.fmix_per_sample,
        fmix_no_repeat=args.fmix_no_repeat,
        resizemix_scope_min=args.resizemix_scope_min,
        resizemix_scope_max=args.resizemix_scope_max,
        resizemix_prob=args.resizemix_prob,
        resizemix_per_sample=args.resizemix_per_sample,
        guidedmixup_alpha=args.guidedmixup_alpha,
        guidedmixup_prob=args.guidedmixup_prob,
        guidedmixup_blur_kernel=args.guidedmixup_blur_kernel,
        guidedmixup_condition=args.guidedmixup_condition,
        catchupmix_alpha=args.catchupmix_alpha,
        catchupmix_cutmix_alpha=args.catchupmix_cutmix_alpha,
        catchupmix_num_layers=args.catchupmix_num_layers,
        catchupmix_no_repeat=args.catchupmix_no_repeat,
    )


def _force_mix_args_for_visualization(
    method: str,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Return visualization args with stochastic no-op probabilities disabled."""
    if not args.force_mix:
        return args

    method = _canonical_method_name(
        method,
    )
    values = vars(
        args,
    ).copy()

    if method in {
        "cutmix",
        "cutmix_sumix",
    }:
        values["cutmix_prob"] = 1.0

    if method == "fmix":
        values["fmix_prob"] = 1.0

    if method == "resizemix":
        values["resizemix_prob"] = 1.0

    if method == "saliencymix":
        values["saliencymix_prob"] = 1.0

    if method in {
        "guidedmixup",
        "guided_sr",
        "guidedmixup_sr",
    }:
        values["guidedmixup_prob"] = 1.0

    return argparse.Namespace(
        **values,
    )


def _skip_reason(
    method: str,
    args: argparse.Namespace,
) -> str | None:
    """Return a skip reason for methods missing required visual inputs."""
    method = _canonical_method_name(
        method,
    )

    if method in PRECOMPUTED_SALIENCY_METHODS and not _saliency_cache_exists(
        dataset=args.dataset,
        saliency_dir=args.saliency_dir,
    ):
        saliency_path = get_train_saliency_path(
            dataset_name=args.dataset,
            saliency_dir=args.saliency_dir,
        )
        return (
            f"missing saliency cache: {saliency_path}. "
            "Generate it with python -m allthemix.data.saliency."
        )

    return None


def visualize_mix_samples(
    args: argparse.Namespace,
) -> list[Path]:
    """Generate visualization grids for requested methods."""
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

    display_images = _denormalize_images(
        images_np,
        dataset=args.dataset,
    )

    saved_paths: list[Path] = []

    for method_index, method in enumerate(
        args.methods,
    ):
        method = _canonical_method_name(
            method,
        )
        reason = _skip_reason(
            method=method,
            args=args,
        )

        if reason is not None:
            print(
                f"Skipping {method}: {reason}"
            )
            continue

        if method in FEATURE_LEVEL_METHODS:
            print(
                f"Visualizing {method}: feature-layer mixes may show unchanged input images."
            )

        rng = jax.random.PRNGKey(
            args.seed
            + method_index,
        )
        method_args = _force_mix_args_for_visualization(
            method=method,
            args=args,
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
        saliency_maps = _get_saliency_for_display(
            method=method,
            images=images,
            aux_info=aux_info,
            args=method_args,
        )
        display_mixed = _denormalize_images(
            output["images"],
            dataset=args.dataset,
        )

        output_path = output_dir / f"{method}_mix_samples.png"
        _save_method_grid(
            method=method,
            images=display_images,
            mixed_images=display_mixed,
            raw_images=images,
            raw_mixed_images=output["images"],
            labels=labels_np,
            labels_b=output["labels_b"],
            perm=perm,
            lam=output["lam"],
            saliency_maps=saliency_maps,
            output_path=output_path,
            num_samples=args.num_samples,
            dpi=args.dpi,
            layer=output.get(
                "layer",
            ),
        )
        saved_paths.append(
            output_path,
        )
        print(
            f"Saved {method}: {output_path}"
        )

    return saved_paths


def main() -> None:
    """Run the visualization CLI."""
    args = parse_args()
    saved_paths = visualize_mix_samples(
        args,
    )

    if not saved_paths:
        raise SystemExit(
            "No visualization files were generated."
        )


if __name__ == "__main__":
    main()
