from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from allthemix.methods.selector import get_mixer
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.loop import train_one_epoch
from allthemix.training.engine.single.train import create_train_state, train_step
from allthemix.training.losses.loss_selector import compute_train_loss_and_targets
from allthemix.training.utils.mix_metrics import compute_mix_debug_metrics

IMAGE_SIZE = 32
NUM_CLASSES = 2
BATCH_SIZE = 2


def _build_parser() -> argparse.ArgumentParser:
    """Build the binary CutMix epoch diagnostic parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run real training epochs on a two-sample binary CutMix batch."
        ),
    )
    parser.add_argument("--model", type=str, default="preact_resnet18")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--output_dir", type=str, default="outputs/debug")
    parser.add_argument(
        "--formula_tolerance",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--execution_tolerance",
        type=float,
        default=5e-4,
        help=(
            "Tolerance between an isolated forward pass and the fused JIT "
            "forward/backward pass on accelerators."
        ),
    )
    return parser


def _build_binary_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create one all-zero image and one all-one image with opposite labels."""
    zeros = jnp.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        dtype=jnp.float32,
    )
    ones = jnp.ones_like(
        zeros,
    )
    images = jnp.stack(
        [
            zeros,
            ones,
        ],
        axis=0,
    )
    labels = jnp.asarray(
        [
            0,
            1,
        ],
        dtype=jnp.int32,
    )

    return images, labels


def _tree_max_abs_delta(
    first: Any,
    second: Any,
) -> float:
    """Return the largest absolute difference across two matching pytrees."""
    deltas = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda left, right: jnp.max(
                jnp.abs(
                    left - right,
                )
            ),
            first,
            second,
        )
    )

    if not deltas:
        return 0.0

    return float(
        jnp.max(
            jnp.stack(
                deltas,
            )
        )
    )


def _tree_is_finite(
    tree: Any,
) -> bool:
    """Return whether every array in a pytree contains finite values."""
    leaves = jax.tree_util.tree_leaves(
        tree,
    )

    return all(
        bool(
            jnp.all(
                jnp.isfinite(
                    leaf,
                )
            )
        )
        for leaf in leaves
    )


def _manual_soft_label_loss(
    logits: jnp.ndarray,
    labels_a: jnp.ndarray,
    labels_b: jnp.ndarray,
    lam: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the CutMix soft-label loss without repository loss helpers."""
    log_probabilities = jax.nn.log_softmax(
        logits,
        axis=-1,
    )
    row_indices = jnp.arange(
        logits.shape[0],
    )
    loss_a = -log_probabilities[
        row_indices,
        labels_a,
    ]
    loss_b = -log_probabilities[
        row_indices,
        labels_b,
    ]

    return jnp.mean(
        lam * loss_a
        + (
            1.0 - lam
        )
        * loss_b
    )


def _save_visualization(
    output_path: Path,
    images: np.ndarray,
    paired_images: np.ndarray,
    mixed_images: np.ndarray,
    lam: float,
) -> None:
    """Save the exact binary source, paired source, mask, and mixed batch."""
    mask = np.abs(
        mixed_images - images,
    ).max(
        axis=-1,
    )
    figure, axes = plt.subplots(
        BATCH_SIZE,
        4,
        figsize=(
            10,
            5,
        ),
    )

    for row in range(BATCH_SIZE):
        panels = (
            images[row, :, :, 0],
            paired_images[row, :, :, 0],
            mask[row],
            mixed_images[row, :, :, 0],
        )
        titles = (
            f"source A[{row}]",
            f"source B[{row}]",
            "changed mask",
            f"mixed, lam={lam:.6f}",
        )

        for column, (panel, title) in enumerate(
            zip(
                panels,
                titles,
            )
        ):
            axis = axes[
                row,
                column,
            ]
            axis.imshow(
                panel,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axis.set_title(
                title,
            )
            axis.set_xticks([])
            axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(
        figure,
    )


def run_probe(
    model_name: str,
    seed: int,
    epochs: int,
    alpha: float,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    output_dir: Path,
    formula_tolerance: float,
    execution_tolerance: float,
) -> dict[str, Any]:
    """Run and validate actual CutMix training epochs."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if formula_tolerance <= 0.0:
        raise ValueError("formula_tolerance must be positive.")
    if execution_tolerance <= 0.0:
        raise ValueError("execution_tolerance must be positive.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    images, labels = _build_binary_batch()
    model = build_model(
        name=model_name,
        num_classes=NUM_CLASSES,
    )
    mixer_fn = get_mixer(
        name="cutmix",
        num_classes=NUM_CLASSES,
        cutmix_alpha=alpha,
        cutmix_prob=1.0,
        cutmix_no_repeat=True,
        cutmix_variant="standard",
    )
    root_rng = jax.random.PRNGKey(
        seed,
    )
    root_rng, init_rng = jax.random.split(
        root_rng,
    )
    initial_state = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        input_shape=(
            BATCH_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
    )

    next_epoch_rng, step_rng = jax.random.split(
        root_rng,
    )
    mix_rng, dropout_rng = jax.random.split(
        step_rng,
        2,
    )
    mixer_output = mixer_fn(
        rng=mix_rng,
        images=images,
        labels=labels,
        aux_info={},
    )
    mixed_images = mixer_output.images
    paired_images = images[
        mixer_output.perm
    ]
    variables = {
        "params": initial_state.params,
        "batch_stats": initial_state.batch_stats,
    }
    logits, _ = initial_state.apply_fn(
        variables,
        mixed_images,
        training=True,
        mutable=[
            "batch_stats",
        ],
        rngs={
            "dropout": dropout_rng,
        },
    )
    expected_loss, target_labels = compute_train_loss_and_targets(
        method="cutmix",
        logits=logits,
        mixer_output=mixer_output,
        num_classes=NUM_CLASSES,
    )
    manual_loss = _manual_soft_label_loss(
        logits=logits,
        labels_a=mixer_output.labels_a,
        labels_b=mixer_output.labels_b,
        lam=mixer_output.lam,
    )
    expected_accuracy = jnp.mean(
        jnp.argmax(
            logits,
            axis=-1,
        )
        == target_labels
    )
    expected_mix_metrics = compute_mix_debug_metrics(
        images=images,
        mixed_images=mixed_images,
        labels_a=mixer_output.labels_a,
        labels_b=mixer_output.labels_b,
        lam=mixer_output.lam,
        perm=mixer_output.perm,
    )

    (
        direct_state,
        direct_loss,
        direct_accuracy,
        direct_mix_metrics,
    ) = train_step(
        state=initial_state,
        rng=step_rng,
        images=images,
        labels=labels,
        mixer_fn=mixer_fn,
        method="cutmix",
        num_classes=NUM_CLASSES,
        aux_info={},
        return_mix_metrics=True,
    )

    (
        trained_state,
        returned_rng,
        epoch_loss,
        epoch_accuracy,
        epoch_mix_metrics,
    ) = train_one_epoch(
        state=initial_state,
        rng=root_rng,
        train_ds=[
            (
                np.asarray(
                    images,
                ),
                np.asarray(
                    labels,
                ),
            ),
        ],
        mixer_fn=mixer_fn,
        method="cutmix",
        num_classes=NUM_CLASSES,
        max_train_steps=-1,
        return_mix_metrics=True,
    )

    lam = float(
        mixer_output.lam,
    )
    changed_per_sample = np.mean(
        np.abs(
            np.asarray(
                mixed_images,
            )
            - np.asarray(
                images,
            )
        )
        > 1e-6,
        axis=(
            1,
            2,
            3,
        ),
    )
    effective_lam = 1.0 - changed_per_sample
    epoch_outputs = [
        (
            mixer_output,
            paired_images,
        ),
    ]
    epoch_history = [
        {
            "epoch": 1,
            "lambda": lam,
            "changed_ratio_per_sample": changed_per_sample.tolist(),
            "train_step_loss": float(
                direct_loss,
            ),
            "epoch_loss": epoch_loss,
            "train_step_accuracy": float(
                direct_accuracy,
            ),
            "epoch_accuracy": epoch_accuracy,
            "parameter_update_max_abs_delta": _tree_max_abs_delta(
                initial_state.params,
                trained_state.params,
            ),
            "train_step_epoch_parameter_max_abs_delta": _tree_max_abs_delta(
                direct_state.params,
                trained_state.params,
            ),
            "train_step_epoch_batch_stats_max_abs_delta": _tree_max_abs_delta(
                direct_state.batch_stats,
                trained_state.batch_stats,
            ),
            "train_step_epoch_metric_max_abs_delta": max(
                abs(
                    epoch_mix_metrics[key]
                    - float(
                        direct_mix_metrics[key],
                    )
                )
                for key in direct_mix_metrics
            ),
            "rng_matches": bool(
                jnp.array_equal(
                    returned_rng,
                    next_epoch_rng,
                )
            ),
        },
    ]

    first_trained_state = trained_state
    first_returned_rng = returned_rng
    current_state = trained_state
    current_rng = returned_rng

    for epoch_index in range(
        2,
        epochs + 1,
    ):
        next_rng, next_step_rng = jax.random.split(
            current_rng,
        )
        next_mix_rng, _ = jax.random.split(
            next_step_rng,
            2,
        )
        next_mixer_output = mixer_fn(
            rng=next_mix_rng,
            images=images,
            labels=labels,
            aux_info={},
        )
        next_paired_images = images[
            next_mixer_output.perm
        ]
        (
            next_direct_state,
            next_direct_loss,
            next_direct_accuracy,
            next_direct_metrics,
        ) = train_step(
            state=current_state,
            rng=next_step_rng,
            images=images,
            labels=labels,
            mixer_fn=mixer_fn,
            method="cutmix",
            num_classes=NUM_CLASSES,
            aux_info={},
            return_mix_metrics=True,
        )
        (
            next_epoch_state,
            next_returned_rng,
            next_epoch_loss,
            next_epoch_accuracy,
            next_epoch_metrics,
        ) = train_one_epoch(
            state=current_state,
            rng=current_rng,
            train_ds=[
                (
                    np.asarray(
                        images,
                    ),
                    np.asarray(
                        labels,
                    ),
                ),
            ],
            mixer_fn=mixer_fn,
            method="cutmix",
            num_classes=NUM_CLASSES,
            max_train_steps=-1,
            return_mix_metrics=True,
        )
        next_changed_ratio = np.mean(
            np.abs(
                np.asarray(
                    next_mixer_output.images,
                )
                - np.asarray(
                    images,
                )
            )
            > 1e-6,
            axis=(
                1,
                2,
                3,
            ),
        )
        metric_max_abs_delta = max(
            abs(
                next_epoch_metrics[key]
                - float(
                    next_direct_metrics[key],
                )
            )
            for key in next_direct_metrics
        )
        epoch_history.append(
            {
                "epoch": epoch_index,
                "lambda": float(
                    next_mixer_output.lam,
                ),
                "changed_ratio_per_sample": next_changed_ratio.tolist(),
                "train_step_loss": float(
                    next_direct_loss,
                ),
                "epoch_loss": next_epoch_loss,
                "train_step_accuracy": float(
                    next_direct_accuracy,
                ),
                "epoch_accuracy": next_epoch_accuracy,
                "parameter_update_max_abs_delta": _tree_max_abs_delta(
                    current_state.params,
                    next_epoch_state.params,
                ),
                "train_step_epoch_parameter_max_abs_delta": (
                    _tree_max_abs_delta(
                        next_direct_state.params,
                        next_epoch_state.params,
                    )
                ),
                "train_step_epoch_batch_stats_max_abs_delta": (
                    _tree_max_abs_delta(
                        next_direct_state.batch_stats,
                        next_epoch_state.batch_stats,
                    )
                ),
                "train_step_epoch_metric_max_abs_delta": metric_max_abs_delta,
                "rng_matches": bool(
                    jnp.array_equal(
                        next_returned_rng,
                        next_rng,
                    )
                ),
            }
        )
        epoch_outputs.append(
            (
                next_mixer_output,
                next_paired_images,
            )
        )
        current_state = next_epoch_state
        current_rng = next_returned_rng

    trained_state = current_state
    returned_rng = current_rng
    parameter_delta = _tree_max_abs_delta(
        initial_state.params,
        trained_state.params,
    )
    batch_stats_delta = _tree_max_abs_delta(
        initial_state.batch_stats,
        trained_state.batch_stats,
    )
    epoch_parameter_delta = _tree_max_abs_delta(
        direct_state.params,
        first_trained_state.params,
    )
    epoch_batch_stats_delta = _tree_max_abs_delta(
        direct_state.batch_stats,
        first_trained_state.batch_stats,
    )
    checks = {
        "binary_output": bool(
            np.all(
                np.isin(
                    np.asarray(
                        mixed_images,
                    ),
                    [
                        0.0,
                        1.0,
                    ],
                )
            )
        ),
        "no_identity_pairs": bool(
            jnp.all(
                mixer_output.perm
                != jnp.arange(
                    BATCH_SIZE,
                )
            )
        ),
        "paired_labels_follow_permutation": bool(
            jnp.array_equal(
                mixer_output.labels_b,
                labels[
                    mixer_output.perm
                ],
            )
        ),
        "lambda_matches_changed_area": bool(
            np.allclose(
                effective_lam,
                lam,
                atol=formula_tolerance,
            )
        ),
        "manual_loss_matches_repository_loss": bool(
            np.isclose(
                float(
                    manual_loss,
                ),
                float(
                    expected_loss,
                ),
                atol=formula_tolerance,
            )
        ),
        "compiled_train_loss_matches_reference": bool(
            np.isclose(
                float(
                    direct_loss,
                ),
                float(
                    expected_loss,
                ),
                atol=execution_tolerance,
            )
        ),
        "compiled_train_accuracy_matches_reference": bool(
            np.isclose(
                float(
                    direct_accuracy,
                ),
                float(
                    expected_accuracy,
                ),
                atol=formula_tolerance,
            )
        ),
        "compiled_mix_metrics_match_reference": all(
            np.isclose(
                float(
                    direct_mix_metrics[key],
                ),
                float(
                    value,
                ),
                atol=formula_tolerance,
            )
            for key, value in expected_mix_metrics.items()
        ),
        "epoch_loss_matches_train_step": bool(
            np.isclose(
                epoch_loss,
                float(
                    direct_loss,
                ),
                atol=formula_tolerance,
            )
        ),
        "epoch_accuracy_matches_train_step": bool(
            np.isclose(
                epoch_accuracy,
                float(
                    direct_accuracy,
                ),
                atol=formula_tolerance,
            )
        ),
        "epoch_mix_metrics_match": all(
            np.isclose(
                epoch_mix_metrics[key],
                float(
                    direct_mix_metrics[key],
                ),
                atol=formula_tolerance,
            )
            for key in expected_mix_metrics
        ),
        "epoch_parameters_match_train_step": (
            epoch_parameter_delta <= formula_tolerance
        ),
        "epoch_batch_stats_match_train_step": (
            epoch_batch_stats_delta <= formula_tolerance
        ),
        "first_epoch_rng_advanced": bool(
            jnp.array_equal(
                first_returned_rng,
                next_epoch_rng,
            )
        ),
        "optimizer_step_matches_epoch_count": int(
            trained_state.step
        )
        == int(
            initial_state.step
        )
        + epochs,
        "all_epoch_losses_match_train_step": all(
            np.isclose(
                record["epoch_loss"],
                record["train_step_loss"],
                atol=formula_tolerance,
            )
            for record in epoch_history
        ),
        "all_epoch_accuracies_match_train_step": all(
            np.isclose(
                record["epoch_accuracy"],
                record["train_step_accuracy"],
                atol=formula_tolerance,
            )
            for record in epoch_history
        ),
        "all_epoch_states_match_train_step": all(
            record["train_step_epoch_parameter_max_abs_delta"]
            <= formula_tolerance
            and record["train_step_epoch_batch_stats_max_abs_delta"]
            <= formula_tolerance
            for record in epoch_history
        ),
        "all_epoch_metrics_match_train_step": all(
            record["train_step_epoch_metric_max_abs_delta"]
            <= formula_tolerance
            for record in epoch_history
        ),
        "all_epoch_rngs_match": all(
            record["rng_matches"]
            for record in epoch_history
        ),
        "all_epoch_parameters_updated": all(
            record["parameter_update_max_abs_delta"] > 0.0
            for record in epoch_history
        ),
        "all_epoch_lambdas_match_changed_area": all(
            np.allclose(
                1.0
                - np.asarray(
                    record["changed_ratio_per_sample"],
                ),
                record["lambda"],
                atol=formula_tolerance,
            )
            for record in epoch_history
        ),
        "parameters_updated": parameter_delta > 0.0,
        "batch_stats_updated": batch_stats_delta > 0.0,
        "trained_state_is_finite": _tree_is_finite(
            trained_state,
        ),
    }
    passed = all(
        checks.values()
    )
    report = {
        "passed": passed,
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "image_shape": [
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ],
        "labels_a": np.asarray(
            mixer_output.labels_a,
        ).tolist(),
        "labels_b": np.asarray(
            mixer_output.labels_b,
        ).tolist(),
        "permutation": np.asarray(
            mixer_output.perm,
        ).tolist(),
        "lambda": lam,
        "changed_ratio_per_sample": changed_per_sample.tolist(),
        "effective_lambda_per_sample": effective_lam.tolist(),
        "manual_loss": float(
            manual_loss,
        ),
        "repository_loss": float(
            expected_loss,
        ),
        "compiled_train_step_loss": float(
            direct_loss,
        ),
        "epoch_loss": epoch_loss,
        "expected_accuracy": float(
            expected_accuracy,
        ),
        "epoch_accuracy": epoch_accuracy,
        "compiled_train_step_accuracy": float(
            direct_accuracy,
        ),
        "optimizer_step_before": int(
            initial_state.step,
        ),
        "optimizer_step_after": int(
            trained_state.step,
        ),
        "parameter_max_abs_delta": parameter_delta,
        "batch_stats_max_abs_delta": batch_stats_delta,
        "epoch_parameter_max_abs_delta_from_train_step": epoch_parameter_delta,
        "epoch_batch_stats_max_abs_delta_from_train_step": (
            epoch_batch_stats_delta
        ),
        "mix_metrics": epoch_mix_metrics,
        "epoch_history": epoch_history,
        "checks": checks,
    }

    for epoch_index, (
        saved_mixer_output,
        saved_paired_images,
    ) in enumerate(
        epoch_outputs,
        start=1,
    ):
        saved_mixed_images = np.asarray(
            saved_mixer_output.images,
        )

        for sample_index in range(
            BATCH_SIZE,
        ):
            matrix = np.asarray(
                saved_mixed_images[
                    sample_index,
                    :,
                    :,
                    0,
                ],
                dtype=np.int32,
            )
            np.savetxt(
                output_dir
                / (
                    f"cutmix_binary_epoch_{epoch_index:03d}_"
                    f"sample{sample_index}.txt"
                ),
                matrix,
                fmt="%d",
            )

        _save_visualization(
            output_path=(
                output_dir
                / f"cutmix_binary_epoch_{epoch_index:03d}.png"
            ),
            images=np.asarray(
                images,
            ),
            paired_images=np.asarray(
                saved_paired_images,
            ),
            mixed_images=saved_mixed_images,
            lam=float(
                saved_mixer_output.lam,
            ),
        )

    _save_visualization(
        output_path=output_dir / "cutmix_binary_epoch.png",
        images=np.asarray(
            images,
        ),
        paired_images=np.asarray(
            epoch_outputs[0][1],
        ),
        mixed_images=np.asarray(
            epoch_outputs[0][0].images,
        ),
        lam=float(
            epoch_outputs[0][0].lam,
        ),
    )
    with (
        output_dir / "cutmix_binary_epoch.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            sort_keys=True,
        )

    return report


def main() -> None:
    """Run the CLI diagnostic and fail loudly on any inconsistency."""
    args = _build_parser().parse_args()
    report = run_probe(
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        output_dir=Path(
            args.output_dir,
        ),
        formula_tolerance=args.formula_tolerance,
        execution_tolerance=args.execution_tolerance,
    )
    status = (
        "PASS"
        if report["passed"]
        else "FAIL"
    )
    print(f"Binary CutMix epoch probe: {status}")
    print(
        "  permutation: "
        f"{report['permutation']} | labels_a: {report['labels_a']} | "
        f"labels_b: {report['labels_b']}"
    )
    print(
        "  lambda: "
        f"{report['lambda']:.9f} | changed ratios: "
        f"{report['changed_ratio_per_sample']}"
    )
    print(
        "  loss manual/repository/train_step/epoch: "
        f"{report['manual_loss']:.9f} / "
        f"{report['repository_loss']:.9f} / "
        f"{report['compiled_train_step_loss']:.9f} / "
        f"{report['epoch_loss']:.9f}"
    )
    print(
        "  accuracy expected/train_step/epoch: "
        f"{report['expected_accuracy']:.6f} / "
        f"{report['compiled_train_step_accuracy']:.6f} / "
        f"{report['epoch_accuracy']:.6f}"
    )
    print(
        "  optimizer step: "
        f"{report['optimizer_step_before']} -> "
        f"{report['optimizer_step_after']} | parameter delta: "
        f"{report['parameter_max_abs_delta']:.9g}"
    )
    print("  per-epoch results:")

    for record in report["epoch_history"]:
        print(
            f"    epoch {record['epoch']}: "
            f"lam={record['lambda']:.9f}, "
            f"loss={record['epoch_loss']:.9f}, "
            f"acc={record['epoch_accuracy']:.6f}, "
            "parameter_delta="
            f"{record['parameter_update_max_abs_delta']:.9g}"
        )

    for name, passed in report["checks"].items():
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {name}"
        )

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
