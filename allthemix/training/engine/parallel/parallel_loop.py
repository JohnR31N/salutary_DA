from __future__ import annotations

import time
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.training.engine.batch_utils import unpack_batch
from allthemix.training.engine.parallel.parallel_eval import parallel_eval_step
from allthemix.training.engine.parallel.parallel_train import parallel_train_step
from allthemix.training.engine.parallel.parallel_utils import (
    pad_to_device_multiple,
    shard_aux_info,
)
from allthemix.utils.parallel import shard_array

_METRIC_REDUCER_SUM = 0
_METRIC_REDUCER_MINIMUM = 1
_METRIC_REDUCER_MAXIMUM = 2


def _metric_reducer(key: str) -> int:
    """Return the repository reducer selected by an exact metric suffix."""

    if key.endswith("_min"):
        return _METRIC_REDUCER_MINIMUM
    if key.endswith("_max"):
        return _METRIC_REDUCER_MAXIMUM
    return _METRIC_REDUCER_SUM


@cache
def _make_initial_metric_pack(incoming_keys: tuple[str, ...]):
    """Compile one first-replica gather for a newly observed exact dtype."""

    if not incoming_keys:
        raise ValueError("an initial metric pack requires at least one key")

    @jax.jit
    def pack(incoming_values):
        """Stack the first device replica of every incoming metric."""

        return jnp.stack(tuple(value[0] for value in incoming_values))

    return pack


@cache
def _make_metric_pack_update(
    previous_keys: tuple[str, ...],
    incoming_keys: tuple[str, ...],
    result_keys: tuple[str, ...],
):
    """Compile one vector update for a fixed exact-key/dtype layout."""

    previous_positions = {
        key: index for index, key in enumerate(previous_keys)
    }
    incoming_positions = {
        key: index for index, key in enumerate(incoming_keys)
    }
    reducers = tuple(_metric_reducer(key) for key in result_keys)

    @jax.jit
    def update(previous_values, incoming_values):
        """Apply exact per-key reducers through one packed vector dispatch."""

        incoming_scalars = tuple(value[0] for value in incoming_values)
        result = []
        for key, reducer in zip(result_keys, reducers, strict=True):
            previous_position = previous_positions.get(key)
            incoming_position = incoming_positions.get(key)
            if previous_position is None:
                # A first value is copied directly, preserving NaN and signed zero.
                result.append(incoming_scalars[incoming_position])
                continue
            previous = previous_values[previous_position]
            if incoming_position is None:
                result.append(previous)
                continue
            incoming = incoming_scalars[incoming_position]
            if reducer == _METRIC_REDUCER_MINIMUM:
                result.append(jnp.minimum(previous, incoming))
            elif reducer == _METRIC_REDUCER_MAXIMUM:
                result.append(jnp.maximum(previous, incoming))
            else:
                result.append(previous + incoming)
        return jnp.stack(tuple(result))

    return update


class _PackedMetricAccumulator:
    """Accumulate replicated scalar metrics with one device chain per dtype."""

    def __init__(self, *, device_count: int | None = None) -> None:
        self.device_count = (
            jax.local_device_count() if device_count is None else int(device_count)
        )
        if self.device_count <= 0:
            raise ValueError("device_count must be positive")
        self._keys: list[str] = []
        self._counts: dict[str, int] = {}
        self._key_dtypes: dict[str, object] = {}
        self._key_pack_tokens: dict[str, str] = {}
        self._pack_keys: dict[str, list[str]] = {}
        self._pack_dtypes: dict[str, object] = {}
        self._pack_values: dict[str, jax.Array] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        """Return metric keys in global first-seen order."""

        return tuple(self._keys)

    @property
    def counts(self) -> dict[str, int]:
        """Return a copy of exact per-key presence counts."""

        return dict(self._counts)

    @property
    def dtype_tokens(self) -> dict[str, str]:
        """Return each key's exact observed dtype-pack token."""

        return dict(self._key_pack_tokens)

    def accumulate(self, metrics: dict[str, jnp.ndarray]) -> None:
        """Validate, then atomically enqueue one packed update per exact dtype."""

        validated: list[tuple[str, jax.Array, object, str]] = []
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError("metric keys must be exact strings")
            shape = tuple(value.shape)
            expected_shape = (self.device_count,)
            if shape != expected_shape:
                raise ValueError(
                    f"metric {key!r} must have exact shape {expected_shape}, "
                    f"received {shape}"
                )
            dtype = jnp.dtype(value.dtype)
            token = str(dtype)
            if key in self._key_dtypes and self._key_dtypes[key] != dtype:
                raise TypeError(
                    f"metric {key!r} changed dtype from "
                    f"{self._key_dtypes[key]} to {dtype}"
                )
            if token in self._pack_dtypes and self._pack_dtypes[token] != dtype:
                raise TypeError(
                    f"dtype token {token!r} does not identify one exact dtype"
                )
            validated.append((key, value, dtype, token))

        if not validated:
            return

        incoming_by_token: dict[str, list[tuple[str, jax.Array, object]]] = {}
        for key, value, dtype, token in validated:
            incoming_by_token.setdefault(token, []).append((key, value, dtype))

        proposed_pack_values: dict[str, jax.Array] = {}
        proposed_pack_keys: dict[str, list[str]] = {}
        proposed_pack_dtypes: dict[str, object] = {}
        for token, entries in incoming_by_token.items():
            incoming_keys = tuple(key for key, _value, _dtype in entries)
            incoming_values = tuple(value for _key, value, _dtype in entries)
            previous_keys = tuple(self._pack_keys.get(token, ()))
            new_keys = tuple(key for key in incoming_keys if key not in self._counts)
            result_keys = previous_keys + new_keys
            if not previous_keys:
                packed = _make_initial_metric_pack(incoming_keys)(incoming_values)
            else:
                packed = _make_metric_pack_update(
                    previous_keys,
                    incoming_keys,
                    result_keys,
                )(self._pack_values[token], incoming_values)
            proposed_pack_values[token] = packed
            proposed_pack_keys[token] = list(result_keys)
            proposed_pack_dtypes[token] = entries[0][2]

        # Commit host metadata only after every incoming leaf has been validated
        # and every dtype-specific packed update has been constructed.
        self._pack_values.update(proposed_pack_values)
        self._pack_keys.update(proposed_pack_keys)
        self._pack_dtypes.update(proposed_pack_dtypes)
        for key, _value, dtype, token in validated:
            if key not in self._counts:
                self._keys.append(key)
                self._counts[key] = 1
                self._key_dtypes[key] = dtype
                self._key_pack_tokens[key] = token
            else:
                self._counts[key] += 1

    def device_values(self) -> dict[str, jax.Array]:
        """Return packed device vectors for the epoch's single host sync."""

        return dict(self._pack_values)

    def finalize(self, host_packs: dict[str, np.ndarray]) -> dict[str, float]:
        """Restore the public metric mapping from host-materialized packs."""

        observed_tokens = set(host_packs)
        expected_tokens = set(self._pack_values)
        if observed_tokens != expected_tokens:
            raise ValueError("host metric packs do not match accumulator dtypes")
        result: dict[str, float] = {}
        pack_positions = {
            token: {key: index for index, key in enumerate(keys)}
            for token, keys in self._pack_keys.items()
        }
        for key in self._keys:
            token = self._key_pack_tokens[key]
            value = np.asarray(host_packs[token])[pack_positions[token][key]]
            if key.endswith(("_min", "_max")):
                result[key] = float(value)
            else:
                result[key] = float(value / self._counts[key])
        return result


def _accumulate_step_metrics(
    accumulator: _PackedMetricAccumulator,
    metrics: dict[str, jnp.ndarray],
) -> None:
    """Accumulate replicated metrics through dtype-specific packed vectors."""

    accumulator.accumulate(metrics)


def parallel_train_one_epoch(
    state,
    rngs: jax.Array,
    train_ds,
    mixer_fn,
    method: str,
    num_classes: int,
    max_train_steps: int,
    sumix_gamma: float = 0.5,
    sumix_semantic_scale: float = -1.0,
    return_sumix_metrics: bool = False,
    cross_device_shuffle: bool = False,
    cross_device_no_repeat: bool = False,
    sync_batch_stats: bool = False,
    return_mix_metrics: bool = False,
    validation_aware_strategy=None,
    batch_training_strategy=None,
    component_timing: dict[str, float] | None = None,
    return_batch_count: bool = False,
):
    """Run one PMAP training epoch across local devices.

    When ``return_batch_count`` is enabled, append the number of updates
    actually executed.  The default return shape remains unchanged for
    existing callers.
    """
    loss_sum = None
    accuracy_sum = None
    extra_metric_accumulator = _PackedMetricAccumulator()
    strategy_pair_sums = None
    strategy_pair_counts = None

    if (
        validation_aware_strategy is not None
        and batch_training_strategy is not None
    ):
        raise ValueError(
            "Only one custom training strategy may own a batch update."
        )

    iterator = iter(train_ds)
    step = 0
    while True:
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if component_timing is not None:
            component_timing["data"] = component_timing.get("data", 0.0) + (
                time.perf_counter() - data_started
            )
        if max_train_steps > 0 and step >= max_train_steps:
            break

        images, labels, aux_info = unpack_batch(
            batch,
        )

        images = jnp.asarray(
            shard_array(
                images,
            )
        )

        labels = jnp.asarray(
            shard_array(
                labels,
            )
        )

        aux_info = shard_aux_info(
            aux_info,
        )

        split_rngs = jax.vmap(
            lambda key: jax.random.split(
                key,
                2,
            )
        )(
            rngs,
        )

        rngs = split_rngs[:, 0]
        step_rngs = split_rngs[:, 1]

        if batch_training_strategy is not None:
            state, loss, accuracy, step_extra_metrics = (
                batch_training_strategy.train_step(
                    task_state=state,
                    images=images,
                    labels=labels,
                    rng=step_rngs,
                )
            )
            _accumulate_step_metrics(
                accumulator=extra_metric_accumulator,
                metrics=step_extra_metrics,
            )

        elif validation_aware_strategy is not None:
            meta_batch = validation_aware_strategy.next_meta_batch()
            meta_images, meta_labels, _ = unpack_batch(
                meta_batch,
            )
            meta_images = jnp.asarray(
                shard_array(
                    meta_images,
                )
            )
            meta_labels = jnp.asarray(
                shard_array(
                    meta_labels,
                )
            )
            (
                state,
                loss,
                accuracy,
                step_extra_metrics,
                step_pair_sums,
                step_pair_counts,
            ) = validation_aware_strategy.train_step(
                task_state=state,
                images=images,
                labels=labels,
                meta_images=meta_images,
                meta_labels=meta_labels,
                rng=step_rngs,
            )
            _accumulate_step_metrics(
                accumulator=extra_metric_accumulator,
                metrics=step_extra_metrics,
            )
            strategy_pair_sums = (
                step_pair_sums[0]
                if strategy_pair_sums is None
                else strategy_pair_sums + step_pair_sums[0]
            )
            strategy_pair_counts = (
                step_pair_counts[0]
                if strategy_pair_counts is None
                else strategy_pair_counts + step_pair_counts[0]
            )

        elif return_sumix_metrics or return_mix_metrics:
            state, loss, accuracy, step_extra_metrics = parallel_train_step(
                state,
                step_rngs,
                images,
                labels,
                aux_info,
                mixer_fn,
                method,
                num_classes,
                sumix_gamma,
                sumix_semantic_scale,
                return_sumix_metrics,
                cross_device_shuffle,
                cross_device_no_repeat,
                sync_batch_stats,
                return_mix_metrics,
            )

            _accumulate_step_metrics(
                accumulator=extra_metric_accumulator,
                metrics=step_extra_metrics,
            )

        else:
            state, loss, accuracy = parallel_train_step(
                state,
                step_rngs,
                images,
                labels,
                aux_info,
                mixer_fn,
                method,
                num_classes,
                sumix_gamma,
                sumix_semantic_scale,
                False,
                cross_device_shuffle,
                cross_device_no_repeat,
                sync_batch_stats,
                False,
            )

        loss_sum = loss[0] if loss_sum is None else loss_sum + loss[0]
        accuracy_sum = (
            accuracy[0]
            if accuracy_sum is None
            else accuracy_sum + accuracy[0]
        )
        step += 1

    if loss_sum is None or accuracy_sum is None or step == 0:
        raise ValueError("Training dataset produced no batches.")

    device_epoch_values = {
        "loss_sum": loss_sum,
        "accuracy_sum": accuracy_sum,
        "extra_metric_packs": extra_metric_accumulator.device_values(),
    }
    if validation_aware_strategy is not None:
        if strategy_pair_sums is None or strategy_pair_counts is None:
            raise ValueError(
                "Validation-aware training dataset produced no batches."
            )
        device_epoch_values["strategy_pair_sums"] = strategy_pair_sums
        device_epoch_values["strategy_pair_counts"] = strategy_pair_counts
    metric_sync_started = time.perf_counter()
    host_epoch_values = jax.device_get(device_epoch_values)
    if component_timing is not None:
        component_timing["metric_sync"] = component_timing.get(
            "metric_sync",
            0.0,
        ) + (time.perf_counter() - metric_sync_started)

    mean_loss = float(host_epoch_values["loss_sum"] / step)
    mean_accuracy = float(host_epoch_values["accuracy_sum"] / step)
    mean_extra_metrics = extra_metric_accumulator.finalize(
        host_epoch_values["extra_metric_packs"]
    )

    if validation_aware_strategy is not None:
        mean_extra_metrics.update(
            validation_aware_strategy.finish_epoch(
                pair_sums=np.asarray(host_epoch_values["strategy_pair_sums"]),
                pair_counts=np.asarray(host_epoch_values["strategy_pair_counts"]),
            )
        )

    result = (state, rngs, mean_loss, mean_accuracy, mean_extra_metrics)
    if return_batch_count:
        return (*result, step)
    return result


def parallel_evaluate(
    state,
    test_ds,
    num_classes: int,
    max_eval_steps: int,
    return_counts: bool = False,
):
    """Evaluate a PMAP model across local devices.

    When ``return_counts`` is enabled, append the number of processed batches
    and real (unpadded) examples.  Existing callers retain the five-value
    metric tuple by default.
    """
    total_loss_device = None
    total_top1_device = None
    total_top5_device = None
    total_count_device = None
    processed_batches = 0

    for step, batch in enumerate(
        test_ds,
    ):
        if max_eval_steps > 0 and step >= max_eval_steps:
            break

        images, labels, _ = unpack_batch(
            batch,
        )

        images, valid_mask = pad_to_device_multiple(
            images,
            fill_value=0.0,
        )

        labels, _ = pad_to_device_multiple(
            labels,
            fill_value=0,
        )

        images = jnp.asarray(
            shard_array(
                images,
            )
        )

        labels = jnp.asarray(
            shard_array(
                labels,
            )
        )

        valid_mask = jnp.asarray(
            shard_array(
                valid_mask,
            )
        )

        (
            loss_sum,
            top1_correct_sum,
            top5_correct_sum,
            valid_count,
        ) = parallel_eval_step(
            state,
            images,
            labels,
            valid_mask,
            num_classes,
        )

        total_loss_device = (
            loss_sum[0]
            if total_loss_device is None
            else total_loss_device + loss_sum[0]
        )
        total_top1_device = (
            top1_correct_sum[0]
            if total_top1_device is None
            else total_top1_device + top1_correct_sum[0]
        )
        total_top5_device = (
            top5_correct_sum[0]
            if total_top5_device is None
            else total_top5_device + top5_correct_sum[0]
        )
        total_count_device = (
            valid_count[0]
            if total_count_device is None
            else total_count_device + valid_count[0]
        )
        processed_batches += 1

    if total_count_device is None:
        raise ValueError("Evaluation dataset produced no samples.")

    host_totals = jax.device_get(
        {
            "loss": total_loss_device,
            "top1": total_top1_device,
            "top5": total_top5_device,
            "count": total_count_device,
        }
    )
    total_loss = float(host_totals["loss"])
    total_top1_correct = float(host_totals["top1"])
    total_top5_correct = float(host_totals["top5"])
    total_count = float(host_totals["count"])

    if total_count == 0.0:
        raise ValueError(
            "Evaluation dataset produced no samples."
        )

    mean_loss = float(  # Average eval loss over real samples.
        total_loss / total_count
    )

    mean_top1_accuracy = float(  # Average top-1 accuracy over real samples.
        total_top1_correct / total_count
    )

    mean_top5_accuracy = float(  # Average top-5 accuracy over real samples.
        total_top5_correct / total_count
    )

    mean_top1_error = float(  # Error is one minus top-1 accuracy.
        1.0 - mean_top1_accuracy
    )

    mean_top5_error = float(  # Error is one minus top-5 accuracy.
        1.0 - mean_top5_accuracy
    )

    result = (
        mean_loss,
        mean_top1_accuracy,
        mean_top5_accuracy,
        mean_top1_error,
        mean_top5_error,
    )
    if return_counts:
        return (*result, processed_batches, int(total_count))
    return result
