from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from allthemix.networks.batch_norm import batch_norm
from allthemix.networks.builder import build_model

BATCH_NORM_MOMENTUM = 0.9
BATCH_NORM_EPSILON = 1e-5


class BatchNormProbe(nn.Module):
    """Expose the repository BatchNorm helper as a minimal test module."""

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        training: bool,
        sync_batch_stats: bool,
    ) -> jnp.ndarray:
        """Normalize one synthetic image batch."""
        return batch_norm(
            images,
            training=training,
            sync_batch_stats=sync_batch_stats,
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the SyncBN diagnostic command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate synchronized BatchNorm statistics across JAX PMAP devices."
        ),
    )
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--formula_tolerance",
        type=float,
        default=2e-3,
        help="Tolerance for TPU/XLA floating-point formula comparisons.",
    )
    parser.add_argument(
        "--replica_tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for cross-replica running-stat equality.",
    )
    parser.add_argument("--model_image_size", type=int, default=64)
    parser.add_argument("--model_per_device_batch_size", type=int, default=2)
    parser.add_argument("--skip_model_probe", action="store_true")
    parser.add_argument("--allow_single_device", action="store_true")
    parser.add_argument("--json_output", type=str, default="")
    return parser


def _validate_positive_args(
    args: argparse.Namespace,
) -> None:
    """Reject invalid dimensions before compiling a PMAP probe."""
    integer_values = {
        "per_device_batch_size": args.per_device_batch_size,
        "height": args.height,
        "width": args.width,
        "channels": args.channels,
        "steps": args.steps,
        "model_image_size": args.model_image_size,
        "model_per_device_batch_size": args.model_per_device_batch_size,
    }
    invalid = [
        name
        for name, value in integer_values.items()
        if value <= 0
    ]
    if invalid:
        raise ValueError(
            "Expected positive integer arguments: " + ", ".join(invalid),
        )
    if args.formula_tolerance <= 0.0:
        raise ValueError("formula_tolerance must be positive.")
    if args.replica_tolerance <= 0.0:
        raise ValueError("replica_tolerance must be positive.")


def make_probe_batch(
    num_devices: int,
    per_device_batch_size: int,
    height: int,
    width: int,
    channels: int,
    step: int = 0,
) -> np.ndarray:
    """Create device shards with deliberately different local statistics."""
    scalar_count = per_device_batch_size * height * width
    within_device = np.linspace(
        -1.0,
        1.0,
        scalar_count,
        dtype=np.float32,
    ).reshape(
        1,
        per_device_batch_size,
        height,
        width,
        1,
    )
    device_offsets = (
        np.arange(num_devices, dtype=np.float32).reshape(
            num_devices,
            1,
            1,
            1,
            1,
        )
        * 4.0
    )
    channel_offsets = (
        np.arange(channels, dtype=np.float32).reshape(
            1,
            1,
            1,
            1,
            channels,
        )
        * 0.25
    )

    # Shift every step so the running-stat recurrence is also testable.
    step_offset = np.float32(step * 0.75)
    return within_device + device_offsets + channel_offsets + step_offset


def compute_batch_statistics(
    batch: np.ndarray,
    synchronized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected stats using Flax's default fast-variance formula."""
    local_reduction_axes = (
        1,
        2,
        3,
    )
    local_mean = np.mean(
        batch,
        axis=local_reduction_axes,
        dtype=np.float32,
    )
    local_mean_square = np.mean(
        np.square(
            batch,
            dtype=np.float32,
        ),
        axis=local_reduction_axes,
        dtype=np.float32,
    )

    if synchronized:
        # PMAP SyncBN averages each replica's equally sized local moments.
        mean = np.mean(
            local_mean,
            axis=0,
            dtype=np.float32,
        )
        mean_square = np.mean(
            local_mean_square,
            axis=0,
            dtype=np.float32,
        )
    else:
        mean = local_mean
        mean_square = local_mean_square

    # Flax BatchNorm defaults to var = max(0, E[x^2] - E[x]^2).
    variance = np.maximum(
        np.float32(0.0),
        mean_square
        - np.square(
            mean,
            dtype=np.float32,
        ),
    )
    return mean, variance


def expected_normalized_batch(
    batch: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    synchronized: bool,
) -> np.ndarray:
    """Apply the expected BatchNorm training formula to a sharded batch."""
    if synchronized:
        stat_shape = (
            1,
            1,
            1,
            1,
            batch.shape[-1],
        )
    else:
        stat_shape = (
            batch.shape[0],
            1,
            1,
            1,
            batch.shape[-1],
        )

    # BN(x) = (x - mean) / sqrt(population_variance + epsilon).
    return (
        batch - mean.reshape(stat_shape)
    ) / np.sqrt(
        variance.reshape(stat_shape) + BATCH_NORM_EPSILON,
    )


def update_expected_running_statistics(
    running_mean: np.ndarray,
    running_variance: np.ndarray,
    batch_mean: np.ndarray,
    batch_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Flax exponential moving-average BatchNorm update."""
    # running = momentum * running + (1 - momentum) * current_batch.
    new_mean = (
        BATCH_NORM_MOMENTUM * running_mean
        + (1.0 - BATCH_NORM_MOMENTUM) * batch_mean
    )
    new_variance = (
        BATCH_NORM_MOMENTUM * running_variance
        + (1.0 - BATCH_NORM_MOMENTUM) * batch_variance
    )
    return new_mean, new_variance


def _flatten_tree(
    tree: Any,
) -> dict[str, Any]:
    """Flatten a Flax variable tree into slash-delimited paths."""
    return flatten_dict(
        unfreeze(tree),
        sep="/",
    )


def _extract_single_batch_norm_stats(
    batch_stats: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the only running mean and variance from the minimal probe."""
    flat_stats = _flatten_tree(
        batch_stats,
    )
    means = [
        np.asarray(value)
        for path, value in flat_stats.items()
        if path.endswith("/mean") or path == "mean"
    ]
    variances = [
        np.asarray(value)
        for path, value in flat_stats.items()
        if path.endswith("/var") or path == "var"
    ]
    if len(means) != 1 or len(variances) != 1:
        raise RuntimeError(
            "Expected exactly one BatchNorm mean and variance in the probe, "
            f"found {len(means)} means and {len(variances)} variances.",
        )
    return means[0], variances[0]


def _max_abs_error(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    """Return the largest absolute difference between two arrays."""
    return float(
        np.max(
            np.abs(
                actual - expected,
            ),
        ),
    )


def _max_replica_spread(
    tree: Any,
) -> tuple[float, bool, int]:
    """Measure the maximum BatchNorm-state difference from replica zero."""
    max_spread = 0.0
    all_finite = True
    leaf_count = 0
    for path, value in _flatten_tree(tree).items():
        if not (
            path.endswith("/mean")
            or path.endswith("/var")
            or path in {"mean", "var"}
        ):
            continue

        array = np.asarray(value)
        leaf_count += 1
        all_finite = all_finite and bool(
            np.all(
                np.isfinite(array),
            ),
        )
        if array.shape[0] > 1:
            max_spread = max(
                max_spread,
                float(
                    np.max(
                        np.abs(
                            array - array[0],
                        ),
                    ),
                ),
            )

    return max_spread, all_finite, leaf_count


def _make_training_apply_fn(
    model: nn.Module,
    synchronized: bool,
):
    """Create a PMAP training forward pass for one synchronization mode."""

    @partial(
        jax.pmap,
        axis_name="batch",
    )
    def apply_fn(
        params,
        batch_stats,
        images,
    ):
        outputs, mutable_state = model.apply(
            {
                "params": params,
                "batch_stats": batch_stats,
            },
            images,
            training=True,
            sync_batch_stats=synchronized,
            mutable=["batch_stats"],
        )
        return outputs, mutable_state["batch_stats"]

    return apply_fn


def run_exact_probe(
    num_devices: int,
    per_device_batch_size: int,
    height: int,
    width: int,
    channels: int,
    steps: int,
    formula_tolerance: float,
    replica_tolerance: float,
) -> dict[str, Any]:
    """Validate exact synchronized and local BatchNorm calculations."""
    model = BatchNormProbe()
    dummy_images = jnp.zeros(
        (
            per_device_batch_size,
            height,
            width,
            channels,
        ),
        dtype=jnp.float32,
    )
    variables = model.init(
        jax.random.PRNGKey(0),
        dummy_images,
        training=True,
        sync_batch_stats=False,
    )

    sync_params = jax_utils.replicate(
        variables["params"],
    )
    sync_batch_stats = jax_utils.replicate(
        variables["batch_stats"],
    )
    local_params = jax_utils.replicate(
        variables["params"],
    )
    local_batch_stats = jax_utils.replicate(
        variables["batch_stats"],
    )

    sync_apply = _make_training_apply_fn(
        model=model,
        synchronized=True,
    )
    local_apply = _make_training_apply_fn(
        model=model,
        synchronized=False,
    )

    expected_sync_mean = np.zeros(
        (channels,),
        dtype=np.float32,
    )
    expected_sync_variance = np.ones(
        (channels,),
        dtype=np.float32,
    )
    expected_local_mean = np.zeros(
        (
            num_devices,
            channels,
        ),
        dtype=np.float32,
    )
    expected_local_variance = np.ones(
        (
            num_devices,
            channels,
        ),
        dtype=np.float32,
    )

    sync_output_error = 0.0
    local_output_error = 0.0
    sync_output_mean_abs = 0.0
    sync_output_variance_error = 0.0

    for step in range(steps):
        batch = make_probe_batch(
            num_devices=num_devices,
            per_device_batch_size=per_device_batch_size,
            height=height,
            width=width,
            channels=channels,
            step=step,
        )
        sync_mean, sync_variance = compute_batch_statistics(
            batch=batch,
            synchronized=True,
        )
        local_mean, local_variance = compute_batch_statistics(
            batch=batch,
            synchronized=False,
        )

        sync_outputs, sync_batch_stats = sync_apply(
            sync_params,
            sync_batch_stats,
            jnp.asarray(batch),
        )
        local_outputs, local_batch_stats = local_apply(
            local_params,
            local_batch_stats,
            jnp.asarray(batch),
        )
        sync_outputs_np = np.asarray(
            jax.device_get(sync_outputs),
        )
        local_outputs_np = np.asarray(
            jax.device_get(local_outputs),
        )

        expected_sync_outputs = expected_normalized_batch(
            batch=batch,
            mean=sync_mean,
            variance=sync_variance,
            synchronized=True,
        )
        expected_local_outputs = expected_normalized_batch(
            batch=batch,
            mean=local_mean,
            variance=local_variance,
            synchronized=False,
        )
        sync_output_error = max(
            sync_output_error,
            _max_abs_error(
                sync_outputs_np,
                expected_sync_outputs,
            ),
        )
        local_output_error = max(
            local_output_error,
            _max_abs_error(
                local_outputs_np,
                expected_local_outputs,
            ),
        )

        actual_sync_output_mean = np.mean(
            sync_outputs_np,
            axis=(
                0,
                1,
                2,
                3,
            ),
        )
        actual_sync_output_variance = np.var(
            sync_outputs_np,
            axis=(
                0,
                1,
                2,
                3,
            ),
        )
        expected_normalized_variance = sync_variance / (
            sync_variance + BATCH_NORM_EPSILON
        )
        sync_output_mean_abs = max(
            sync_output_mean_abs,
            float(
                np.max(
                    np.abs(actual_sync_output_mean),
                ),
            ),
        )
        sync_output_variance_error = max(
            sync_output_variance_error,
            _max_abs_error(
                actual_sync_output_variance,
                expected_normalized_variance,
            ),
        )

        expected_sync_mean, expected_sync_variance = (
            update_expected_running_statistics(
                running_mean=expected_sync_mean,
                running_variance=expected_sync_variance,
                batch_mean=sync_mean,
                batch_variance=sync_variance,
            )
        )
        expected_local_mean, expected_local_variance = (
            update_expected_running_statistics(
                running_mean=expected_local_mean,
                running_variance=expected_local_variance,
                batch_mean=local_mean,
                batch_variance=local_variance,
            )
        )

    actual_sync_mean, actual_sync_variance = _extract_single_batch_norm_stats(
        jax.device_get(sync_batch_stats),
    )
    actual_local_mean, actual_local_variance = _extract_single_batch_norm_stats(
        jax.device_get(local_batch_stats),
    )
    sync_mean_error = _max_abs_error(
        actual_sync_mean,
        np.broadcast_to(
            expected_sync_mean,
            actual_sync_mean.shape,
        ),
    )
    sync_variance_error = _max_abs_error(
        actual_sync_variance,
        np.broadcast_to(
            expected_sync_variance,
            actual_sync_variance.shape,
        ),
    )
    local_mean_error = _max_abs_error(
        actual_local_mean,
        expected_local_mean,
    )
    local_variance_error = _max_abs_error(
        actual_local_variance,
        expected_local_variance,
    )
    sync_replica_spread = _max_abs_error(
        actual_sync_mean,
        np.broadcast_to(
            actual_sync_mean[0],
            actual_sync_mean.shape,
        ),
    )
    sync_replica_spread = max(
        sync_replica_spread,
        _max_abs_error(
            actual_sync_variance,
            np.broadcast_to(
                actual_sync_variance[0],
                actual_sync_variance.shape,
            ),
        ),
    )
    local_replica_spread = _max_abs_error(
        actual_local_mean,
        np.broadcast_to(
            actual_local_mean[0],
            actual_local_mean.shape,
        ),
    )

    checks = {
        "sync_output_formula": sync_output_error <= formula_tolerance,
        "local_output_formula": local_output_error <= formula_tolerance,
        "sync_output_zero_mean": sync_output_mean_abs <= formula_tolerance,
        "sync_output_variance": (
            sync_output_variance_error <= formula_tolerance
        ),
        "sync_running_mean": sync_mean_error <= formula_tolerance,
        "sync_running_variance": sync_variance_error <= formula_tolerance,
        "local_running_mean": local_mean_error <= formula_tolerance,
        "local_running_variance": local_variance_error <= formula_tolerance,
        "sync_replicas_identical": (
            sync_replica_spread <= replica_tolerance
        ),
    }
    if num_devices > 1:
        checks["probe_detects_local_replica_difference"] = (
            local_replica_spread > replica_tolerance
        )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "sync_output_max_abs_error": sync_output_error,
            "local_output_max_abs_error": local_output_error,
            "sync_output_mean_abs_max": sync_output_mean_abs,
            "sync_output_variance_max_abs_error": sync_output_variance_error,
            "sync_running_mean_max_abs_error": sync_mean_error,
            "sync_running_variance_max_abs_error": sync_variance_error,
            "local_running_mean_max_abs_error": local_mean_error,
            "local_running_variance_max_abs_error": local_variance_error,
            "sync_replica_max_spread": sync_replica_spread,
            "local_replica_max_spread": local_replica_spread,
        },
    }


def _make_model_probe_batch(
    num_devices: int,
    per_device_batch_size: int,
    image_size: int,
) -> np.ndarray:
    """Create RGB model inputs with distinct device-local distributions."""
    base = make_probe_batch(
        num_devices=num_devices,
        per_device_batch_size=per_device_batch_size,
        height=image_size,
        width=image_size,
        channels=3,
    )
    return base / np.float32(max(num_devices * 4.0, 1.0))


def run_model_probe(
    num_devices: int,
    per_device_batch_size: int,
    image_size: int,
    replica_tolerance: float,
) -> dict[str, Any]:
    """Check every PreActResNet BatchNorm leaf across PMAP replicas."""
    model = build_model(
        name="preact_resnet18",
        num_classes=200,
        resnet_stem_type="imagenet",
    )
    dummy_images = jnp.zeros(
        (
            per_device_batch_size,
            image_size,
            image_size,
            3,
        ),
        dtype=jnp.float32,
    )
    variables = model.init(
        jax.random.PRNGKey(1),
        dummy_images,
        training=True,
        sync_batch_stats=False,
    )
    params = jax_utils.replicate(
        variables["params"],
    )
    initial_batch_stats = jax_utils.replicate(
        variables["batch_stats"],
    )
    images = jnp.asarray(
        _make_model_probe_batch(
            num_devices=num_devices,
            per_device_batch_size=per_device_batch_size,
            image_size=image_size,
        ),
    )

    sync_apply = _make_training_apply_fn(
        model=model,
        synchronized=True,
    )
    local_apply = _make_training_apply_fn(
        model=model,
        synchronized=False,
    )
    sync_logits, sync_batch_stats = sync_apply(
        params,
        initial_batch_stats,
        images,
    )
    _, local_batch_stats = local_apply(
        params,
        initial_batch_stats,
        images,
    )
    sync_logits_np = np.asarray(
        jax.device_get(sync_logits),
    )
    sync_spread, sync_finite, sync_leaf_count = _max_replica_spread(
        jax.device_get(sync_batch_stats),
    )
    local_spread, local_finite, local_leaf_count = _max_replica_spread(
        jax.device_get(local_batch_stats),
    )

    checks = {
        "sync_model_stats_finite": sync_finite,
        "local_model_stats_finite": local_finite,
        "sync_model_logits_finite": bool(np.all(np.isfinite(sync_logits_np))),
        "model_bn_leaf_counts_match": sync_leaf_count == local_leaf_count,
        "sync_model_replicas_identical": sync_spread <= replica_tolerance,
    }
    if num_devices > 1:
        checks["model_probe_detects_local_replica_difference"] = (
            local_spread > replica_tolerance
        )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "sync_model_replica_max_spread": sync_spread,
            "local_model_replica_max_spread": local_spread,
            "sync_model_batch_stat_leaf_count": sync_leaf_count,
            "local_model_batch_stat_leaf_count": local_leaf_count,
        },
    }


def _print_section(
    name: str,
    result: dict[str, Any],
) -> None:
    """Print one diagnostic section in a paste-friendly format."""
    print(f"\n{name}: {'PASS' if result['passed'] else 'FAIL'}")
    for check_name, passed in result["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check_name}")
    for metric_name, value in result["metrics"].items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.8g}")
        else:
            print(f"  {metric_name}: {value}")


def main() -> None:
    """Run exact and model-level SyncBN diagnostics on local JAX devices."""
    args = _build_parser().parse_args()
    _validate_positive_args(
        args,
    )

    local_devices = jax.local_devices()
    num_devices = len(local_devices)
    print("SyncBatchNorm diagnostic")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX process count: {jax.process_count()}")
    print(f"JAX global device count: {jax.device_count()}")
    print(f"JAX local device count: {num_devices}")
    print("Local devices:")
    for device in local_devices:
        print(f"  - {device}")

    if num_devices < 2 and not args.allow_single_device:
        raise RuntimeError(
            "SyncBN requires at least two local devices for a meaningful "
            "diagnostic. Run this command on the TPU, or pass "
            "--allow_single_device for a formula-only local smoke test.",
        )

    exact_result = run_exact_probe(
        num_devices=num_devices,
        per_device_batch_size=args.per_device_batch_size,
        height=args.height,
        width=args.width,
        channels=args.channels,
        steps=args.steps,
        formula_tolerance=args.formula_tolerance,
        replica_tolerance=args.replica_tolerance,
    )
    _print_section(
        "Exact BatchNorm probe",
        exact_result,
    )

    model_result = None
    if not args.skip_model_probe:
        model_result = run_model_probe(
            num_devices=num_devices,
            per_device_batch_size=args.model_per_device_batch_size,
            image_size=args.model_image_size,
            replica_tolerance=args.replica_tolerance,
        )
        _print_section(
            "PreActResNet-18 integration probe",
            model_result,
        )

    passed = exact_result["passed"] and (
        model_result is None or model_result["passed"]
    )
    report = {
        "passed": passed,
        "environment": {
            "backend": jax.default_backend(),
            "process_count": jax.process_count(),
            "global_device_count": jax.device_count(),
            "local_device_count": num_devices,
            "devices": [str(device) for device in local_devices],
        },
        "exact_probe": exact_result,
        "model_probe": model_result,
    }

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to: {output_path}")

    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
