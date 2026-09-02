from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from allthemix.utils.backend_environment import validate_jax_environment

validate_jax_environment()

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from allthemix.competitors.ifaugnet.models import (
    AugmentationNetwork,
    resolve_architecture,
)
from allthemix.competitors.ifaugnet.steps import create_augment_state
from allthemix.competitors.ifaugnet.transforms import (
    apply_appearance_transform,
    apply_spatial_transform,
)
from allthemix.config import load_yaml_config
from allthemix.data.pipeline import build_raw_augmented_train_pipeline
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.data.utils.normalization import get_normalization_stats
from allthemix.utils.checkpoint import restore_model_state_file
from allthemix.utils.cli import str2bool

SPATIAL_CHANNELS = (
    "A[y<-y] source-y scale",
    "A[y<-x] source-y shear",
    "A[x<-y] source-x shear",
    "A[x<-x] source-x scale",
    "b[y] source-y translation",
    "b[x] source-x translation",
)


@dataclass(frozen=True)
class BasisResult:
    """One positive/negative transform-channel probe."""

    name: str
    negative: np.ndarray
    positive: np.ndarray
    negative_l1: float
    positive_l1: float
    negative_out_of_range: float
    positive_out_of_range: float
    negative_spatial_oob: float = 0.0
    positive_spatial_oob: float = 0.0


def appearance_channel_names(channels: int) -> tuple[str, ...]:
    """Name the dense color matrix and bias channels in decoder order."""
    symbols = ("R", "G", "B") if channels == 3 else tuple(
        f"C{index}" for index in range(channels)
    )
    weights = tuple(
        f"W[{output}<-{source}]"
        for output in symbols
        for source in symbols
    )
    biases = tuple(f"b[{output}]" for output in symbols)
    return (*weights, *biases)


def _to_numpy(values: Any) -> np.ndarray:
    return np.asarray(jax.device_get(values), dtype=np.float32)


def _image_metrics(
    transformed: np.ndarray,
    original: np.ndarray,
) -> tuple[float, float]:
    return (
        float(np.mean(np.abs(transformed - original))),
        float(np.mean((transformed < 0.0) | (transformed > 1.0))),
    )


def _guarded_raw_magnitude(
    target_delta: float,
    transform_scale: float,
) -> float:
    ratio = min(max(target_delta / max(transform_scale, 1.0e-8), 0.0), 0.95)
    return float(np.arctanh(ratio))


def build_spatial_basis_results(
    image: np.ndarray,
    *,
    parameterization: str,
    spatial_scale: float,
    smoothing_kernel: int,
    displacement_fraction: float,
) -> list[BasisResult]:
    """Probe every spatial field channel through the production transform."""
    if image.ndim != 3:
        raise ValueError("image must have shape [height, width, channels].")
    if not 0.0 < displacement_fraction < 0.5:
        raise ValueError("displacement_fraction must be in (0, 0.5).")

    height, width, _ = image.shape
    images = jnp.asarray(image[None], dtype=jnp.float32)
    results = []
    for channel, name in enumerate(SPATIAL_CHANNELS):
        if parameterization == "paper":
            if channel < 4:
                magnitude = displacement_fraction
            else:
                axis_size = height if channel == 4 else width
                magnitude = displacement_fraction * float(max(axis_size - 1, 1))
        elif parameterization == "guarded":
            magnitude = _guarded_raw_magnitude(
                2.0 * displacement_fraction,
                spatial_scale,
            )
        else:
            raise ValueError("parameterization must be 'paper' or 'guarded'.")

        outputs = []
        grids = []
        for sign in (-1.0, 1.0):
            fields = jnp.zeros((1, height, width, 6), dtype=jnp.float32)
            fields = fields.at[..., channel].set(sign * magnitude)
            transformed, grid = apply_spatial_transform(
                images=images,
                spatial_params=fields,
                spatial_scale=spatial_scale,
                smoothing_kernel=smoothing_kernel,
                parameterization=parameterization,
            )
            outputs.append(_to_numpy(transformed[0]))
            grids.append(_to_numpy(grid[0]))

        negative_l1, negative_range = _image_metrics(outputs[0], image)
        positive_l1, positive_range = _image_metrics(outputs[1], image)
        results.append(
            BasisResult(
                name=name,
                negative=outputs[0],
                positive=outputs[1],
                negative_l1=negative_l1,
                positive_l1=positive_l1,
                negative_out_of_range=negative_range,
                positive_out_of_range=positive_range,
                negative_spatial_oob=float(
                    np.mean(np.any(np.abs(grids[0]) > 1.0, axis=-1))
                ),
                positive_spatial_oob=float(
                    np.mean(np.any(np.abs(grids[1]) > 1.0, axis=-1))
                ),
            )
        )
    return results


def build_appearance_basis_results(
    image: np.ndarray,
    *,
    parameterization: str,
    appearance_scale: float,
    smoothing_kernel: int,
    weight_delta: float,
    bias_delta: float,
) -> list[BasisResult]:
    """Probe every appearance field channel through the production transform."""
    if image.ndim != 3:
        raise ValueError("image must have shape [height, width, channels].")
    if weight_delta <= 0.0 or bias_delta <= 0.0:
        raise ValueError("appearance probe magnitudes must be positive.")

    height, width, channels = image.shape
    images = jnp.asarray(image[None], dtype=jnp.float32)
    names = appearance_channel_names(channels)
    weight_channels = channels * channels
    results = []
    for channel, name in enumerate(names):
        target = weight_delta if channel < weight_channels else bias_delta
        if parameterization == "paper":
            magnitude = target
        elif parameterization == "guarded":
            magnitude = _guarded_raw_magnitude(target, appearance_scale)
        else:
            raise ValueError("parameterization must be 'paper' or 'guarded'.")

        outputs = []
        for sign in (-1.0, 1.0):
            fields = jnp.zeros(
                (1, height, width, weight_channels + channels),
                dtype=jnp.float32,
            )
            fields = fields.at[..., channel].set(sign * magnitude)
            transformed, _ = apply_appearance_transform(
                images=images,
                appearance_params=fields,
                appearance_scale=appearance_scale,
                smoothing_kernel=smoothing_kernel,
                parameterization=parameterization,
            )
            outputs.append(_to_numpy(transformed[0]))

        negative_l1, negative_range = _image_metrics(outputs[0], image)
        positive_l1, positive_range = _image_metrics(outputs[1], image)
        results.append(
            BasisResult(
                name=name,
                negative=outputs[0],
                positive=outputs[1],
                negative_l1=negative_l1,
                positive_l1=positive_l1,
                negative_out_of_range=negative_range,
                positive_out_of_range=positive_range,
            )
        )
    return results


def _show_image(axis, image: np.ndarray, *, cmap: str | None = None) -> None:
    axis.imshow(np.clip(image, 0.0, 1.0), cmap=cmap, vmin=0.0, vmax=1.0)
    axis.set_xticks([])
    axis.set_yticks([])


def save_basis_grid(
    original: np.ndarray,
    results: list[BasisResult],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    """Save original, signed probes, and pixel differences for each channel."""
    figure, axes = plt.subplots(
        len(results),
        5,
        figsize=(13.0, 1.75 * len(results)),
        squeeze=False,
    )
    column_titles = ("input", "negative", "positive", "|negative-input|", "|positive-input|")
    for column, column_title in enumerate(column_titles):
        axes[0, column].set_title(column_title, fontsize=10)

    for row, result in enumerate(results):
        negative_difference = np.mean(np.abs(result.negative - original), axis=-1)
        positive_difference = np.mean(np.abs(result.positive - original), axis=-1)
        difference_max = max(
            float(negative_difference.max()),
            float(positive_difference.max()),
            1.0e-6,
        )
        _show_image(axes[row, 0], original)
        _show_image(axes[row, 1], result.negative)
        _show_image(axes[row, 2], result.positive)
        axes[row, 3].imshow(
            negative_difference,
            cmap="magma",
            vmin=0.0,
            vmax=difference_max,
        )
        axes[row, 4].imshow(
            positive_difference,
            cmap="magma",
            vmin=0.0,
            vmax=difference_max,
        )
        for column in (3, 4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(result.name, fontsize=8)
        axes[row, 1].set_xlabel(
            f"L1={result.negative_l1:.3f}\n"
            f"range={result.negative_out_of_range:.1%} "
            f"oob={result.negative_spatial_oob:.1%}",
            fontsize=7,
        )
        axes[row, 2].set_xlabel(
            f"L1={result.positive_l1:.3f}\n"
            f"range={result.positive_out_of_range:.1%} "
            f"oob={result.positive_spatial_oob:.1%}",
            fontsize=7,
        )

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.03, 0.0, 1.0, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_input_views_grid(
    raw_images: np.ndarray,
    base_images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Show the aligned raw and dataset-augmented views used by IF-AugNet."""
    figure, axes = plt.subplots(
        len(raw_images),
        3,
        figsize=(7.5, 2.25 * len(raw_images)),
        squeeze=False,
    )
    for column, title in enumerate(("raw source", "base augmentation", "absolute difference")):
        axes[0, column].set_title(title, fontsize=10)
    for row, (raw_image, base_image, label) in enumerate(
        zip(raw_images, base_images, labels, strict=True)
    ):
        difference = np.mean(np.abs(base_image - raw_image), axis=-1)
        _show_image(axes[row, 0], raw_image)
        _show_image(axes[row, 1], base_image)
        axes[row, 2].imshow(difference, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        axes[row, 0].set_ylabel(f"class {int(label)}", fontsize=8)
        axes[row, 1].set_xlabel(f"L1={np.mean(difference):.3f}", fontsize=7)
    figure.suptitle("IF-AugNet aligned input pipeline", fontsize=14)
    figure.tight_layout(rect=(0.03, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _build_augment_model(config: dict[str, Any], image_size: int, channels: int):
    return AugmentationNetwork(
        image_size=image_size,
        channels=channels,
        tau_dim=int(config.get("ifaugnet_tau_dim", 128)),
        tau_dropout=float(config.get("ifaugnet_tau_dropout", 0.5)),
        spatial_scale=float(config.get("ifaugnet_spatial_scale", 0.20)),
        appearance_scale=float(config.get("ifaugnet_appearance_scale", 0.25)),
        smoothing_kernel=int(config.get("ifaugnet_smoothing_kernel", 4)),
        use_appearance=bool(config.get("ifaugnet_use_appearance", True)),
        encoder_widths=tuple(config.get("ifaugnet_encoder_widths", (16, 32, 64, 128))),
        decoder_widths=tuple(config.get("ifaugnet_decoder_widths", (64, 32, 16))),
        decoder_base_width=int(config.get("ifaugnet_decoder_base_width", 128)),
        parameterization=str(config.get("ifaugnet_transform_parameterization", "guarded")),
        composition=str(config.get("ifaugnet_composition", "serial")),
        architecture=resolve_architecture(
            str(config.get("ifaugnet_architecture", "auto")),
            image_size,
        ),
    )


def apply_learned_policy(
    base_images: np.ndarray,
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    seed: int,
    training: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Restore an influence policy and execute the exact AugmentationNetwork."""
    image_size = int(base_images.shape[1])
    channels = int(base_images.shape[-1])
    model = _build_augment_model(config, image_size, channels)
    state = create_augment_state(
        rng=jax.random.PRNGKey(seed),
        model=model,
        input_shape=(1, image_size, image_size, channels),
        learning_rate=float(config.get("ifaugnet_learning_rate", 1.0e-4)),
        beta1=float(config.get("ifaugnet_beta1", 0.9)),
        beta2=float(config.get("ifaugnet_beta2", 0.99)),
        gradient_clip_norm=float(config.get("ifaugnet_gradient_clip_norm", 1.0)),
        zero_nonfinite_grads=bool(config.get("ifaugnet_zero_nonfinite_grads", True)),
    )
    state, loaded = restore_model_state_file(
        state=state,
        checkpoint_path=checkpoint_path,
    )
    augmented, aux = state.apply_fn(
        {"params": state.params},
        jnp.asarray(base_images, dtype=jnp.float32),
        training=training,
        return_aux=True,
        rngs={"dropout": jax.random.PRNGKey(seed + 1)},
    )
    return (
        _to_numpy(augmented),
        {key: _to_numpy(value) for key, value in aux.items()},
        loaded,
    )


def save_learned_policy_grid(
    raw_images: np.ndarray,
    base_images: np.ndarray,
    augmented: np.ndarray,
    aux: dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: Path,
    *,
    dpi: int,
) -> list[dict[str, float]]:
    """Show every intermediate that composes the classifier input."""
    spatial = aux["spatial_images"]
    appearance = aux.get("appearance_images", base_images)
    grid = aux["sample_grid"]
    figure, axes = plt.subplots(
        len(base_images),
        6,
        figsize=(14.0, 2.25 * len(base_images)),
        squeeze=False,
    )
    titles = ("raw", "base aug", "spatial", "appearance", "final", "|final-base|")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)

    metrics = []
    for row in range(len(base_images)):
        difference = np.mean(np.abs(augmented[row] - base_images[row]), axis=-1)
        l1, out_of_range = _image_metrics(augmented[row], base_images[row])
        spatial_oob = float(np.mean(np.any(np.abs(grid[row]) > 1.0, axis=-1)))
        metrics.append(
            {
                "label": float(labels[row]),
                "augmented_l1": l1,
                "augmented_out_of_range_fraction": out_of_range,
                "spatial_oob_fraction": spatial_oob,
            }
        )
        for column, image in enumerate(
            (raw_images[row], base_images[row], spatial[row], appearance[row], augmented[row])
        ):
            _show_image(axes[row, column], image)
        axes[row, 5].imshow(difference, cmap="magma", vmin=0.0, vmax=max(float(difference.max()), 1.0e-6))
        axes[row, 5].set_xticks([])
        axes[row, 5].set_yticks([])
        axes[row, 0].set_ylabel(f"class {int(labels[row])}", fontsize=8)
        axes[row, 4].set_xlabel(
            f"L1={l1:.3f} range={out_of_range:.1%}\noob={spatial_oob:.1%}",
            fontsize=7,
        )

    figure.suptitle(
        "IF-AugNet learned policy: forced transform output before probability mask",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return metrics


def _load_input_views(
    config: dict[str, Any],
    *,
    data_dir: str,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = str(config["dataset"])
    pipeline = build_raw_augmented_train_pipeline(
        name=dataset,
        data_dir=data_dir,
        batch_size=num_samples,
        shuffle_buffer_size=int(config.get("shuffle_buffer_size", 10_000)),
        seed=seed,
        drop_remainder=False,
        use_basic_augmentation=bool(config.get("basic_aug", False)),
        augmentation_recipe=config.get("aug_recipe"),
        validation_split=float(config.get("validation_split", 0.0)),
        tiny_imagenet_normalization=str(
            config.get("tiny_imagenet_normalization", "imagenet")
        ),
        deterministic_data=True,
        train_subset_fraction=float(
            config.get("train_subset_fraction", 1.0)
        ),
        val_source=str(config.get("val_source", "train")),
    )
    batch = next(iter(pipeline))
    mean, std = get_normalization_stats(
        dataset=dataset,
        tiny_imagenet_normalization=str(
            config.get("tiny_imagenet_normalization", "imagenet")
        ),
    )
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, -1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, -1)
    raw_images = np.clip(np.asarray(batch["raw_images"]) * std_array + mean_array, 0.0, 1.0)
    base_images = np.clip(np.asarray(batch["images"]) * std_array + mean_array, 0.0, 1.0)
    labels = np.asarray(batch["labels"])
    return raw_images, base_images, labels


def _resolve_checkpoint(path: str) -> Path | None:
    if not path:
        return None
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "ifaugnet_influence_final.msgpack"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"IF-AugNet policy checkpoint does not exist: {checkpoint}")
    return checkpoint


def visualize_ifaugnet_options(args: argparse.Namespace) -> dict[str, Any]:
    """Generate transform-basis, input-pipeline, and learned-policy audits."""
    config = load_yaml_config(args.config)
    if config.get("method") != "ifaugnet":
        raise ValueError("The visualization config must use method: ifaugnet.")
    if args.data_dir:
        config["data_dir"] = args.data_dir
    data_dir = str(config.get("data_dir", "./data"))
    raw_images, base_images, labels = _load_input_views(
        config,
        data_dir=data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    metadata = get_metadata(str(config["dataset"]))
    if base_images.shape[1:3] != (metadata.image_size, metadata.image_size):
        raise ValueError("Visualization input size does not match dataset metadata.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    input_path = output_dir / "ifaugnet_input_pipeline.png"
    save_input_views_grid(raw_images, base_images, labels, input_path, dpi=args.dpi)
    saved["input_pipeline"] = str(input_path)

    sample_index = min(max(args.basis_sample_index, 0), len(base_images) - 1)
    parameterization = str(config.get("ifaugnet_transform_parameterization", "guarded"))
    spatial_results = build_spatial_basis_results(
        base_images[sample_index],
        parameterization=parameterization,
        spatial_scale=float(config.get("ifaugnet_spatial_scale", 0.20)),
        smoothing_kernel=int(config.get("ifaugnet_smoothing_kernel", 4)),
        displacement_fraction=args.spatial_displacement_fraction,
    )
    spatial_path = output_dir / "ifaugnet_spatial_basis.png"
    save_basis_grid(
        base_images[sample_index],
        spatial_results,
        spatial_path,
        title=f"IF-AugNet spatial parameter basis ({parameterization})",
        dpi=args.dpi,
    )
    saved["spatial_basis"] = str(spatial_path)

    appearance_results = []
    if bool(config.get("ifaugnet_use_appearance", True)) and metadata.channels > 1:
        appearance_results = build_appearance_basis_results(
            base_images[sample_index],
            parameterization=parameterization,
            appearance_scale=float(config.get("ifaugnet_appearance_scale", 0.25)),
            smoothing_kernel=int(config.get("ifaugnet_smoothing_kernel", 4)),
            weight_delta=args.appearance_weight_delta,
            bias_delta=args.appearance_bias_delta,
        )
        appearance_path = output_dir / "ifaugnet_appearance_basis.png"
        save_basis_grid(
            base_images[sample_index],
            appearance_results,
            appearance_path,
            title=f"IF-AugNet appearance parameter basis ({parameterization})",
            dpi=args.dpi,
        )
        saved["appearance_basis"] = str(appearance_path)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    learned_metrics = []
    loaded_fields = []
    if checkpoint is not None:
        augmented, aux, loaded_fields = apply_learned_policy(
            base_images,
            config=config,
            checkpoint_path=checkpoint,
            seed=args.seed,
            training=args.learned_training_mode,
        )
        learned_path = output_dir / "ifaugnet_learned_policy.png"
        learned_metrics = save_learned_policy_grid(
            raw_images,
            base_images,
            augmented,
            aux,
            labels,
            learned_path,
            dpi=args.dpi,
        )
        saved["learned_policy"] = str(learned_path)

    summary = {
        "dataset": config["dataset"],
        "parameterization": parameterization,
        "composition": config.get("ifaugnet_composition", "serial"),
        "basic_aug": bool(config.get("basic_aug", False)),
        "aug_recipe": config.get("aug_recipe"),
        "num_samples": len(base_images),
        "basis_sample_index": sample_index,
        "spatial_displacement_fraction": args.spatial_displacement_fraction,
        "appearance_weight_delta": args.appearance_weight_delta,
        "appearance_bias_delta": args.appearance_bias_delta,
        "spatial_channels": [result.name for result in spatial_results],
        "appearance_channels": [result.name for result in appearance_results],
        "spatial_metrics": [
            {key: value for key, value in asdict(result).items() if key not in {"negative", "positive"}}
            for result in spatial_results
        ],
        "appearance_metrics": [
            {key: value for key, value in asdict(result).items() if key not in {"negative", "positive"}}
            for result in appearance_results
        ],
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "checkpoint_loaded_fields": loaded_fields,
        "learned_training_mode": args.learned_training_mode,
        # Fallback must match the CLI default (args.py), or summaries of
        # configs that omit the key misreport the deployed probability.
        "learned_aug_probability": float(
            config.get("ifaugnet_learned_aug_probability", 0.1)
        ),
        "learned_sample_metrics": learned_metrics,
        "saved": saved,
    }
    summary_path = output_dir / "ifaugnet_visualization_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every IF-AugNet transform channel and an optional learned policy.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="outputs/visualize/ifaugnet")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--basis-sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--spatial-displacement-fraction", type=float, default=0.08)
    parser.add_argument("--appearance-weight-delta", type=float, default=0.25)
    parser.add_argument("--appearance-bias-delta", type=float, default=0.15)
    parser.add_argument("--learned-training-mode", type=str2bool, default=True)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def main() -> None:
    summary = visualize_ifaugnet_options(_parse_args())
    print(json.dumps(summary["saved"], indent=2, sort_keys=True))
    print(f"summary={summary['summary']}")


if __name__ == "__main__":
    main()
