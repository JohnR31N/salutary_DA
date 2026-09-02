from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from allthemix.training.engine.parallel import parallel_loop


def _replicated_scalar(value, dtype) -> jax.Array:
    """Return one scalar per local device with a controlled exact dtype."""

    return jnp.full((jax.local_device_count(),), value, dtype=dtype)


def test_packed_metric_accumulator_preserves_dynamic_reducers_and_counts() -> None:
    """Pack dynamic metrics without changing per-key epoch aggregation."""

    accumulator = parallel_loop._PackedMetricAccumulator()
    accumulator.accumulate(
        {
            "metric_mean": _replicated_scalar(1.0, jnp.float32),
            "metric_min": _replicated_scalar(-0.0, jnp.float32),
            "metric_max": _replicated_scalar(2.0, jnp.float32),
        }
    )
    accumulator.accumulate(
        {
            "metric_max": _replicated_scalar(5.0, jnp.float32),
            "metric_mean": _replicated_scalar(3.0, jnp.float32),
            "integer_mean": _replicated_scalar(4, jnp.int32),
        }
    )
    accumulator.accumulate(
        {
            "integer_mean": _replicated_scalar(8, jnp.int32),
            "metric_min": _replicated_scalar(-2.0, jnp.float32),
        }
    )

    assert accumulator.keys == (
        "metric_mean",
        "metric_min",
        "metric_max",
        "integer_mean",
    )
    assert accumulator.counts == {
        "metric_mean": 2,
        "metric_min": 2,
        "metric_max": 2,
        "integer_mean": 2,
    }
    assert len(set(accumulator.dtype_tokens.values())) == 2
    observed = accumulator.finalize(
        jax.device_get(accumulator.device_values())
    )
    assert list(observed) == list(accumulator.keys)
    assert observed == {
        "metric_mean": 2.0,
        "metric_min": -2.0,
        "metric_max": 5.0,
        "integer_mean": 6.0,
    }


def test_packed_metric_accumulator_copies_first_nan_and_signed_zero() -> None:
    """Do not introduce neutral elements when a metric is first observed."""

    accumulator = parallel_loop._PackedMetricAccumulator()
    accumulator.accumulate(
        {
            "nan_mean": _replicated_scalar(np.nan, jnp.float32),
            "signed_zero_min": _replicated_scalar(-0.0, jnp.float32),
        }
    )
    observed = accumulator.finalize(
        jax.device_get(accumulator.device_values())
    )
    assert np.isnan(observed["nan_mean"])
    assert observed["signed_zero_min"] == 0.0
    assert np.signbit(observed["signed_zero_min"])


def test_packed_metric_accumulator_matches_scalar_device_chains_exactly() -> None:
    """Match the replaced scalar JAX reducers without numerical tolerance."""

    keys = tuple(
        f"metric_{index}_{'min' if index % 7 == 0 else 'max' if index % 7 == 1 else 'mean'}"
        for index in range(21)
    )
    accumulator = parallel_loop._PackedMetricAccumulator()
    scalar_values: dict[str, jax.Array] = {}
    counts: dict[str, int] = {}
    for step in range(9):
        metrics = {
            key: _replicated_scalar(
                np.float32((index - 10) * (step + 1) / 37.0),
                jnp.float32,
            )
            for index, key in enumerate(keys)
        }
        accumulator.accumulate(metrics)
        for key, value in metrics.items():
            scalar = value[0]
            if key not in scalar_values:
                scalar_values[key] = scalar
                counts[key] = 1
            elif key.endswith("_min"):
                scalar_values[key] = jnp.minimum(scalar_values[key], scalar)
                counts[key] += 1
            elif key.endswith("_max"):
                scalar_values[key] = jnp.maximum(scalar_values[key], scalar)
                counts[key] += 1
            else:
                scalar_values[key] = scalar_values[key] + scalar
                counts[key] += 1

    assert len(accumulator.device_values()) == 1
    assert next(iter(accumulator.device_values().values())).shape == (21,)
    packed = accumulator.finalize(jax.device_get(accumulator.device_values()))
    scalar_host = jax.device_get(scalar_values)
    expected = {
        key: float(
            value
            if key.endswith(("_min", "_max"))
            else value / counts[key]
        )
        for key, value in scalar_host.items()
    }
    assert packed.keys() == expected.keys()
    for key in keys:
        np.testing.assert_array_equal(packed[key], expected[key])


@pytest.mark.parametrize(
    "invalid_metrics, error_type, message",
    [
        (
            lambda: {
                "valid_new": _replicated_scalar(3.0, jnp.float32),
                "bad_shape": jnp.ones(
                    (jax.local_device_count(), 1),
                    dtype=jnp.float32,
                ),
            },
            ValueError,
            "exact shape",
        ),
        (
            lambda: {"stable_mean": _replicated_scalar(3, jnp.int32)},
            TypeError,
            "changed dtype",
        ),
    ],
)
def test_packed_metric_accumulator_rejects_atomically(
    invalid_metrics,
    error_type,
    message,
) -> None:
    """Reject invalid leaves before changing device packs or host metadata."""

    accumulator = parallel_loop._PackedMetricAccumulator()
    accumulator.accumulate(
        {"stable_mean": _replicated_scalar(1.0, jnp.float32)}
    )
    keys_before = accumulator.keys
    counts_before = accumulator.counts
    tokens_before = accumulator.dtype_tokens
    packs_before = jax.device_get(accumulator.device_values())

    with pytest.raises(error_type, match=message):
        accumulator.accumulate(invalid_metrics())

    assert accumulator.keys == keys_before
    assert accumulator.counts == counts_before
    assert accumulator.dtype_tokens == tokens_before
    packs_after = jax.device_get(accumulator.device_values())
    assert set(packs_after) == set(packs_before)
    for token in packs_before:
        np.testing.assert_array_equal(packs_after[token], packs_before[token])


def _make_rngs() -> jax.Array:
    """Create one RNG key per local device."""
    return jax.random.split(
        jax.random.PRNGKey(
            0,
        ),
        jax.local_device_count(),
    )


def _make_train_ds() -> list[tuple[np.ndarray, np.ndarray]]:
    """Create one global batch divisible by local device count."""
    num_devices = jax.local_device_count()
    batch_size = num_devices * 2

    images = np.ones(
        (
            batch_size,
            4,
            4,
            1,
        ),
        dtype=np.float32,
    )

    labels = np.zeros(
        (
            batch_size,
        ),
        dtype=np.int32,
    )

    return [
        (
            images,
            labels,
        )
    ]


def test_parallel_train_one_epoch_passes_false_sumix_metrics_static_arg(
    monkeypatch,
) -> None:
    """Verify non-debug PMAP training passes every static arg positionally."""
    calls = []

    def fake_parallel_train_step(
        *args,
    ):
        """Record PMAP call arguments without running a real training step."""
        calls.append(
            args,
        )

        return args[0], jnp.asarray([1.0]), jnp.asarray([0.5])

    monkeypatch.setattr(
        parallel_loop,
        "parallel_train_step",
        fake_parallel_train_step,
    )

    parallel_loop.parallel_train_one_epoch(
        state="state",
        rngs=_make_rngs(),
        train_ds=_make_train_ds(),
        mixer_fn=lambda **_: None,
        method="baseline",
        num_classes=10,
        max_train_steps=-1,
        sumix_gamma=0.5,
        sumix_semantic_scale=-1.0,
        return_sumix_metrics=False,
    )

    assert calls
    assert len(
        calls[0],
    ) == 15
    assert calls[0][10] is False
    assert calls[0][11] is False
    assert calls[0][12] is False
    assert calls[0][13] is False
    assert calls[0][14] is False


def test_parallel_train_one_epoch_passes_true_sumix_metrics_static_arg(
    monkeypatch,
) -> None:
    """Verify debug PMAP training passes the metrics flag positionally."""
    calls = []

    def fake_parallel_train_step(
        *args,
    ):
        """Record PMAP call arguments without running a real training step."""
        calls.append(
            args,
        )

        return (
            args[0],
            jnp.asarray([1.0]),
            jnp.asarray([0.5]),
            {
                "lam_a_mean": jnp.full(
                    (jax.local_device_count(),),
                    0.5,
                ),
            },
        )

    monkeypatch.setattr(
        parallel_loop,
        "parallel_train_step",
        fake_parallel_train_step,
    )

    _, _, _, _, metrics, batch_count = parallel_loop.parallel_train_one_epoch(
        state="state",
        rngs=_make_rngs(),
        train_ds=_make_train_ds(),
        mixer_fn=lambda **_: None,
        method="cutmix_sumix",
        num_classes=10,
        max_train_steps=-1,
        sumix_gamma=0.5,
        sumix_semantic_scale=10.0,
        return_sumix_metrics=True,
        return_batch_count=True,
    )

    assert calls
    assert len(
        calls[0],
    ) == 15
    assert calls[0][10] is True
    assert calls[0][11] is False
    assert calls[0][12] is False
    assert calls[0][13] is False
    assert calls[0][14] is False
    assert metrics["lam_a_mean"] == 0.5
    assert batch_count == 1


def test_parallel_train_one_epoch_passes_cross_device_shuffle_static_arg(
    monkeypatch,
) -> None:
    """Verify PMAP training forwards the cross-device shuffle flag."""
    calls = []

    def fake_parallel_train_step(
        *args,
    ):
        """Record PMAP call arguments without running a real training step."""
        calls.append(
            args,
        )

        return args[0], jnp.asarray([1.0]), jnp.asarray([0.5])

    monkeypatch.setattr(
        parallel_loop,
        "parallel_train_step",
        fake_parallel_train_step,
    )

    parallel_loop.parallel_train_one_epoch(
        state="state",
        rngs=_make_rngs(),
        train_ds=_make_train_ds(),
        mixer_fn=lambda **_: None,
        method="fmix",
        num_classes=10,
        max_train_steps=-1,
        cross_device_shuffle=True,
        cross_device_no_repeat=True,
    )

    assert calls
    assert calls[0][11] is True
    assert calls[0][12] is True
    assert calls[0][13] is False
    assert calls[0][14] is False


def test_parallel_train_one_epoch_passes_sync_batch_stats_static_arg(
    monkeypatch,
) -> None:
    """Verify PMAP training forwards the synchronized BatchNorm flag."""
    calls = []

    def fake_parallel_train_step(
        *args,
    ):
        """Record PMAP call arguments without running a real training step."""
        calls.append(
            args,
        )

        return args[0], jnp.asarray([1.0]), jnp.asarray([0.5])

    monkeypatch.setattr(
        parallel_loop,
        "parallel_train_step",
        fake_parallel_train_step,
    )

    parallel_loop.parallel_train_one_epoch(
        state="state",
        rngs=_make_rngs(),
        train_ds=_make_train_ds(),
        mixer_fn=lambda **_: None,
        method="cutmix",
        num_classes=10,
        max_train_steps=-1,
        sync_batch_stats=True,
    )

    assert calls
    assert calls[0][13] is True
    assert calls[0][14] is False


def test_parallel_train_one_epoch_passes_mix_metrics_static_arg(
    monkeypatch,
) -> None:
    """Verify PMAP training forwards the generic mix metrics flag."""
    calls = []

    def fake_parallel_train_step(
        *args,
    ):
        """Record PMAP call arguments without running a real training step."""
        calls.append(
            args,
        )

        return (
            args[0],
            jnp.asarray([1.0]),
            jnp.asarray([0.5]),
            {
                "mix_lam_mean": jnp.full(
                    (jax.local_device_count(),),
                    0.5,
                ),
            },
        )

    monkeypatch.setattr(
        parallel_loop,
        "parallel_train_step",
        fake_parallel_train_step,
    )

    _, _, _, _, metrics = parallel_loop.parallel_train_one_epoch(
        state="state",
        rngs=_make_rngs(),
        train_ds=_make_train_ds(),
        mixer_fn=lambda **_: None,
        method="fmix",
        num_classes=10,
        max_train_steps=-1,
        return_mix_metrics=True,
    )

    assert calls
    assert calls[0][10] is False
    assert calls[0][14] is True
    assert metrics["mix_lam_mean"] == 0.5


def test_parallel_evaluate_pads_tail_batch_without_counting_padding(
    monkeypatch,
) -> None:
    """Verify PMAP eval pads tail batches but masks padded examples out."""
    calls = []

    monkeypatch.setattr(
        parallel_loop.jax,
        "local_device_count",
        lambda: 4,
    )

    def fake_parallel_eval_step(
        state,
        images,
        labels,
        valid_mask,
        num_classes,
    ):
        """Return deterministic sample sums from the sharded eval batch."""
        calls.append(
            (
                images,
                labels,
                valid_mask,
                num_classes,
            )
        )
        valid_count = jnp.sum(valid_mask)
        loss_sum = valid_count * 2.0
        top1_correct_sum = jnp.sum(
            (labels == 1).astype(jnp.float32) * valid_mask
        )
        top5_correct_sum = valid_count

        return (
            jnp.repeat(loss_sum[None], images.shape[0]),
            jnp.repeat(top1_correct_sum[None], images.shape[0]),
            jnp.repeat(top5_correct_sum[None], images.shape[0]),
            jnp.repeat(valid_count[None], images.shape[0]),
        )

    monkeypatch.setattr(
        parallel_loop,
        "parallel_eval_step",
        fake_parallel_eval_step,
    )

    images = np.ones(
        (
            5,
            4,
            4,
            1,
        ),
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            1,
            0,
            1,
            1,
            0,
        ],
        dtype=np.int32,
    )

    (
        loss,
        top1_accuracy,
        top5_accuracy,
        top1_error,
        top5_error,
        batch_count,
        example_count,
    ) = parallel_loop.parallel_evaluate(
        state="state",
        test_ds=[
            (
                images,
                labels,
            )
        ],
        num_classes=10,
        max_eval_steps=-1,
        return_counts=True,
    )

    assert calls
    _, _, valid_mask, _ = calls[0]
    assert valid_mask.shape == (
        4,
        2,
    )
    assert float(
        jnp.sum(
            valid_mask,
        )
    ) == 5.0
    np.testing.assert_array_equal(
        np.asarray(valid_mask).reshape(-1)[5:],
        np.zeros(
            3,
            dtype=np.float32,
        ),
    )
    assert loss == 2.0
    assert top1_accuracy == 0.6
    assert top5_accuracy == 1.0
    assert top1_error == 0.4
    assert top5_error == 0.0
    assert batch_count == 1
    assert example_count == 5
