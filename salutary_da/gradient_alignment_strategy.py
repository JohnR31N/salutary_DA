"""Repository-aligned every-batch gradient-alignment training strategy."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.training.engine.parallel.parallel_train import parallel_train_step
from allthemix.training.losses.cross_entropy import (
    cross_entropy,
    soft_cross_entropy_per_sample,
)
from allthemix.training.losses.mixup_loss import mixup_loss
from salutary_da.policies.per_row_continuous import (
    PerRowContinuousPolicyConfig,
    PerRowContinuousPolicyDecision,
    decide_and_summarize_per_row_continuous_device,
    decide_per_row_continuous_device,
    summarize_per_row_continuous_decision_in_pmap,
)
from salutary_da.scorers.gradient_alignment import (
    ClassifierHeadGradientAlignmentScorer,
    FullParameterGradientAlignmentScorer,
    prepare_stratified_validation_batch_cycle,
    prepare_validation_batch,
)


def make_parallel_mixup_probe(mixer_fn, num_classes: int):
    """Return the standard MixUp recipe produced from the step RNGs."""

    if num_classes < 2:
        raise ValueError("MixUp probe requires at least two classes")

    @partial(jax.pmap, axis_name="batch")
    def extract(rng, images, labels):
        mix_rng, dropout_rng = jax.random.split(rng, 2)
        mixed_images, labels_a, labels_b, lam = mixer_fn(
            rng=mix_rng,
            images=images,
            labels=labels,
            aux_info={},
        )[:4]
        one_hot_a = jax.nn.one_hot(labels_a, num_classes)
        one_hot_b = jax.nn.one_hot(labels_b, num_classes)
        target_lam = lam if lam.ndim == 0 else lam.reshape((-1, 1))
        soft_targets = target_lam * one_hot_a + (1.0 - target_lam) * one_hot_b
        return (
            mixed_images,
            labels_a,
            labels_b,
            lam,
            soft_targets,
            dropout_rng,
        )

    return extract


def make_parallel_origin_probe(num_classes: int):
    """Build a baseline probe with the ordinary RNG split and one-hot targets."""

    if num_classes < 2:
        raise ValueError("origin probe requires at least two classes")

    @partial(jax.pmap, axis_name="batch")
    def extract(rng, images, labels):
        _unused_method_rng, dropout_rng = jax.random.split(rng, 2)
        targets = jax.nn.one_hot(labels, num_classes, dtype=jnp.float32)
        ones = jnp.ones((labels.shape[0],), dtype=jnp.float32)
        return images, labels, labels, ones, targets, dropout_rng

    return extract


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(3, 4),
)
def _shuffle_continuous_decision(
    decision: PerRowContinuousPolicyDecision,
    original_targets: jax.Array,
    state_step: jax.Array,
    mode: str,
    seed: int,
) -> PerRowContinuousPolicyDecision:
    """Shuffle equal-dose actions across the complete global batch."""

    if mode not in {"soft_label", "reweight"}:
        raise ValueError("shuffled control requires soft_label or reweight mode")
    local_rows = original_targets.shape[0]
    gathered_weights = jax.lax.all_gather(decision.weights, "batch").reshape(-1)
    gathered_labels = jax.lax.all_gather(
        decision.selected_labels,
        "batch",
    ).reshape(-1)
    gathered_doses = jax.lax.all_gather(decision.doses, "batch").reshape(-1)
    shared_step = jax.lax.pmin(state_step, "batch").astype(jnp.uint32)
    key = jax.random.fold_in(jax.random.PRNGKey(seed), shared_step)
    permutation = jax.random.permutation(key, gathered_weights.size)
    start = jax.lax.axis_index("batch") * local_rows

    def local(value):
        return jax.lax.dynamic_slice_in_dim(value, start, local_rows, axis=0)

    if mode == "reweight":
        weights = local(gathered_weights[permutation])
        return decision._replace(
            weights=weights,
            applied_rows=jnp.abs(weights - 1.0) > 1e-7,
        )
    labels = local(gathered_labels[permutation])
    doses = local(gathered_doses[permutation])
    vertices = jax.nn.one_hot(
        labels,
        original_targets.shape[-1],
        dtype=original_targets.dtype,
    )
    targets_after = (1.0 - doses[:, None]) * original_targets + doses[
        :, None
    ] * vertices
    return decision._replace(
        targets_after=targets_after,
        selected_labels=labels,
        applied_rows=doses > 0.0,
        doses=doses,
    )


def make_parallel_continuous_target_step(
    apply_fn,
    *,
    sync_batch_stats: bool,
):
    """Build a weighted soft-target update for PMAP training."""

    if not sync_batch_stats:
        raise ValueError("continuous SalDA requires synchronized BatchNorm")

    @partial(jax.pmap, axis_name="batch")
    def train_step(
        state,
        images,
        soft_targets,
        weights,
        accuracy_labels,
        dropout_rng,
    ):
        def loss_fn(params):
            logits, updates = apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images,
                training=True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
                sync_batch_stats=True,
            )
            per_example = soft_cross_entropy_per_sample(logits, soft_targets)
            loss = jnp.mean(weights * per_example)
            return loss, (updates["batch_stats"], logits)

        (local_loss, (batch_stats, logits)), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        gradients = jax.lax.pmean(gradients, axis_name="batch")
        new_state = state.apply_gradients(
            grads=gradients,
            batch_stats=batch_stats,
        )
        local_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == accuracy_labels)
        return (
            new_state,
            jax.lax.pmean(local_loss, axis_name="batch"),
            jax.lax.pmean(local_accuracy, axis_name="batch"),
        )

    return train_step


def make_parallel_precomputed_base_step(
    apply_fn,
    *,
    num_classes: int,
    base_method: str,
):
    """Build the ordinary baseline/MixUp update from an already mixed batch."""

    if base_method not in {"baseline", "mixup"}:
        raise ValueError("precomputed GA base method must be baseline or mixup")

    @partial(jax.pmap, axis_name="batch")
    def train_step(
        state,
        mixed_images,
        labels_a,
        labels_b,
        lam,
        dropout_rng,
    ):
        def loss_fn(params):
            logits, updates = apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                mixed_images,
                training=True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
                sync_batch_stats=True,
            )
            if base_method == "baseline":
                loss = cross_entropy(logits, labels_a, num_classes)
            else:
                loss = mixup_loss(
                    logits,
                    labels_a,
                    labels_b,
                    num_classes,
                    lam,
                )
            return loss, (updates["batch_stats"], logits)

        (local_loss, (batch_stats, logits)), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        gradients = jax.lax.pmean(gradients, axis_name="batch")
        new_state = state.apply_gradients(
            grads=gradients,
            batch_stats=batch_stats,
        )
        local_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels_a)
        return (
            new_state,
            jax.lax.pmean(local_loss, axis_name="batch"),
            jax.lax.pmean(local_accuracy, axis_name="batch"),
        )

    return train_step


@partial(jax.pmap, axis_name="batch")
def _summarize_decision(
    decision: PerRowContinuousPolicyDecision,
) -> dict[str, jax.Array]:
    return summarize_per_row_continuous_decision_in_pmap(decision)


def _replicated_optimizer_step(step) -> int:
    """Read one optimizer step and require all device replicas to agree."""

    values = np.asarray(jax.device_get(step))
    if values.size == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("optimizer step must contain replicated integers")
    flattened = values.reshape(-1)
    if np.any(flattened != flattened[0]):
        raise ValueError("optimizer-step replicas disagree")
    return int(flattened[0])


def _mean_replicated_directions(directions):
    """Average equal-size validation gradients without changing their PyTree."""

    if not directions:
        raise ValueError("at least one validation direction is required")
    structure = jax.tree_util.tree_structure(directions[0])
    if any(jax.tree_util.tree_structure(value) != structure for value in directions):
        raise ValueError("validation direction structures disagree")
    return jax.tree_util.tree_map(
        lambda *leaves: (
            sum(leaves[1:], leaves[0])
            / jnp.asarray(len(directions), dtype=leaves[0].dtype)
        ),
        *directions,
    )


def _replace_direction_in_mean(mean, old, new, *, count: int):
    """Update an equal-weight cached mean after replacing one component."""

    if count <= 0:
        raise ValueError("validation direction count must be positive")
    return jax.tree_util.tree_map(
        lambda mean_leaf, old_leaf, new_leaf: (
            mean_leaf
            + (new_leaf - old_leaf) / jnp.asarray(count, dtype=mean_leaf.dtype)
        ),
        mean,
        old,
        new,
    )


def _replicated_direction_geometry(estimated, reference):
    """Compare replica-zero PyTree directions without transferring to host."""

    if jax.tree_util.tree_structure(estimated) != jax.tree_util.tree_structure(
        reference
    ):
        raise ValueError("validation direction structures disagree")
    estimated_leaves = jax.tree_util.tree_leaves(estimated)
    reference_leaves = jax.tree_util.tree_leaves(reference)
    if not estimated_leaves:
        raise ValueError("validation direction must contain array leaves")
    dot = jnp.asarray(0.0, dtype=jnp.float32)
    estimated_square = jnp.asarray(0.0, dtype=jnp.float32)
    reference_square = jnp.asarray(0.0, dtype=jnp.float32)
    difference_square = jnp.asarray(0.0, dtype=jnp.float32)
    for estimated_leaf, reference_leaf in zip(
        estimated_leaves,
        reference_leaves,
        strict=True,
    ):
        if estimated_leaf.ndim < 1 or estimated_leaf.shape != reference_leaf.shape:
            raise ValueError("replicated validation direction shapes disagree")
        estimated_value = estimated_leaf[0].astype(jnp.float32)
        reference_value = reference_leaf[0].astype(jnp.float32)
        dot = dot + jnp.sum(estimated_value * reference_value)
        estimated_square = estimated_square + jnp.sum(estimated_value**2)
        reference_square = reference_square + jnp.sum(reference_value**2)
        difference_square = difference_square + jnp.sum(
            (estimated_value - reference_value) ** 2
        )
    denominator = jnp.sqrt(estimated_square * reference_square)
    cosine = jnp.where(
        denominator > 0.0,
        dot / denominator,
        jnp.where(difference_square == 0.0, 1.0, 0.0),
    )
    relative_l2 = jnp.sqrt(
        difference_square / jnp.maximum(reference_square, jnp.finfo(jnp.float32).tiny)
    )
    return jnp.clip(cosine, -1.0, 1.0), relative_l2


class GradientAlignmentBatchStrategy:
    """Apply exact full-pool or cached-aggregate GA to origin or MixUp."""

    def __init__(
        self,
        *,
        apply_fn,
        template_params,
        mixer_fn,
        num_classes: int,
        validation_images: np.ndarray,
        validation_labels: np.ndarray,
        validation_direction_mode: str = "full",
        validation_batch_size: int = 500,
        validation_batch_seed: int = 0,
        validation_reanchor_interval: int = 50,
        initial_optimizer_step: int = 0,
        learning_rate_fn: Callable[[jax.Array], jax.Array],
        policy: PerRowContinuousPolicyConfig,
        parameter_scope: str = "classifier_head",
        sync_batch_stats: bool = True,
        action_enabled: bool = True,
        expected_validation_examples: int = 5_000,
        audit_mode: bool = False,
        profile_components: bool = False,
        base_method: str = "mixup",
        shuffled_control: bool = False,
        control_seed: int = 0,
        score_start_optimizer_step: int = 0,
        score_stop_optimizer_step: int | None = None,
        action_start_optimizer_step: int = 0,
        action_stop_optimizer_step: int | None = None,
    ) -> None:
        if not sync_batch_stats:
            raise ValueError("GradientAlignmentBatchStrategy requires SyncBN")
        if parameter_scope not in {"classifier_head", "full"}:
            raise ValueError("parameter_scope must be classifier_head or full")
        if expected_validation_examples <= 0:
            raise ValueError("expected_validation_examples must be positive")
        if (
            expected_validation_examples in {4_000, 5_000}
            and jax.local_device_count() != 4
        ):
            raise ValueError(
                "the registered SalDA Vdev batch requires exactly four local devices"
            )
        validation_images = np.asarray(validation_images, dtype=np.float32)
        validation_labels = np.asarray(validation_labels, dtype=np.int32)
        if validation_labels.ndim != 1:
            raise ValueError("validation labels must be one-dimensional")
        if validation_images.shape[0] != validation_labels.shape[0]:
            raise ValueError("validation images and labels must align")
        if validation_labels.shape[0] != expected_validation_examples:
            raise ValueError(
                "SalDA validation direction must use exactly "
                f"{expected_validation_examples} examples"
            )
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if validation_direction_mode not in {"full", "batch_aggregate"}:
            raise ValueError(
                "validation_direction_mode must be full or batch_aggregate"
            )
        if isinstance(validation_reanchor_interval, bool) or not isinstance(
            validation_reanchor_interval,
            (int, np.integer),
        ):
            raise TypeError("validation_reanchor_interval must be an integer")
        if validation_reanchor_interval <= 0:
            raise ValueError("validation_reanchor_interval must be positive")
        if isinstance(initial_optimizer_step, bool) or not isinstance(
            initial_optimizer_step,
            (int, np.integer),
        ):
            raise TypeError("initial_optimizer_step must be an integer")
        if initial_optimizer_step < 0:
            raise ValueError("initial_optimizer_step must be non-negative")
        phase_steps = {
            "score_start_optimizer_step": score_start_optimizer_step,
            "action_start_optimizer_step": action_start_optimizer_step,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < initial_optimizer_step
            for value in phase_steps.values()
        ):
            raise ValueError(
                "score and action start steps must be integer optimizer steps "
                "at or after the initial step"
            )
        if score_start_optimizer_step > action_start_optimizer_step:
            raise ValueError("score start step must not exceed action start step")
        if score_stop_optimizer_step is not None and (
            isinstance(score_stop_optimizer_step, bool)
            or not isinstance(score_stop_optimizer_step, (int, np.integer))
            or score_stop_optimizer_step <= score_start_optimizer_step
        ):
            raise ValueError("score stop step must be after score start step")
        if action_stop_optimizer_step is not None and (
            isinstance(action_stop_optimizer_step, bool)
            or not isinstance(action_stop_optimizer_step, (int, np.integer))
            or action_stop_optimizer_step <= action_start_optimizer_step
        ):
            raise ValueError("action stop step must be after action start step")
        if (
            score_stop_optimizer_step is not None
            and action_stop_optimizer_step is not None
            and action_stop_optimizer_step > score_stop_optimizer_step
        ):
            raise ValueError("action stop step must not exceed score stop step")
        if (
            score_stop_optimizer_step is not None
            and policy.mode != "score_only"
            and action_start_optimizer_step >= score_stop_optimizer_step
        ):
            raise ValueError("action start step must precede score stop step")
        if (
            validation_direction_mode == "batch_aggregate"
            and parameter_scope != "classifier_head"
        ):
            raise ValueError(
                "batch-aggregate validation directions require classifier_head"
            )
        if not callable(learning_rate_fn):
            raise TypeError("learning_rate_fn must be callable")
        if base_method not in {"baseline", "mixup"}:
            raise ValueError("GA base method must be baseline or mixup")
        if shuffled_control and policy.mode not in {"soft_label", "reweight"}:
            raise ValueError("shuffled GA control requires a continuous action mode")

        self.num_classes = int(num_classes)
        self.learning_rate_fn = learning_rate_fn
        self.policy = policy
        self.parameter_scope = parameter_scope
        self.validation_direction_mode = validation_direction_mode
        self.action_enabled = bool(action_enabled)
        self.audit_mode = bool(audit_mode)
        self.profile_components = bool(profile_components)
        self.base_method = base_method
        self.shuffled_control = bool(shuffled_control)
        self.control_seed = int(control_seed)
        self.validation_reanchor_interval = int(validation_reanchor_interval)
        self._initial_optimizer_step = int(initial_optimizer_step)
        self._score_start_optimizer_step = int(score_start_optimizer_step)
        self._score_stop_optimizer_step = (
            None
            if score_stop_optimizer_step is None
            else int(score_stop_optimizer_step)
        )
        self._action_start_optimizer_step = int(action_start_optimizer_step)
        self._action_stop_optimizer_step = (
            None
            if action_stop_optimizer_step is None
            else int(action_stop_optimizer_step)
        )
        self._timing_seconds: defaultdict[str, float] = defaultdict(float)
        self._timing_calls: defaultdict[str, int] = defaultdict(int)
        self._train_steps = 0
        self._scored_steps = 0
        self._action_active_steps = 0
        self._direction_refreshes = 0
        self._validation_gradient_evaluations = 0
        self._direction_validation_example_visits = 0
        self._validation_exact_reanchors = 0
        self._validation_direction_cache = None
        self._validation_aggregate_direction = None
        self._validation_anchor_drift_comparisons = 0
        self._validation_anchor_cosine_sum = None
        self._validation_anchor_cosine_min = None
        self._validation_anchor_relative_l2_sum = None
        self._validation_anchor_relative_l2_max = None
        self._validation_pool_examples = int(expected_validation_examples)
        self._extract_batch = (
            make_parallel_mixup_probe(mixer_fn, num_classes)
            if base_method == "mixup"
            else make_parallel_origin_probe(num_classes)
        )
        self._continuous_step = make_parallel_continuous_target_step(
            apply_fn,
            sync_batch_stats=True,
        )
        self._precomputed_base_step = make_parallel_precomputed_base_step(
            apply_fn,
            num_classes=num_classes,
            base_method=base_method,
        )
        self._mixer_fn = mixer_fn
        if validation_direction_mode == "full":
            self._validation_batch = prepare_validation_batch(
                validation_images,
                validation_labels,
                num_devices=jax.local_device_count(),
            )
            self._validation_batch_cycle = None
            self._validation_examples_per_gradient_evaluation = int(
                expected_validation_examples
            )
            self._validation_direction_cycle_length = 1
            self._validation_batch_seed = None
            full_order = np.arange(
                expected_validation_examples,
                dtype="<i8",
            )
            self._validation_batch_schedule_sha256 = hashlib.sha256(
                full_order.tobytes()
            ).hexdigest()
        else:
            self._validation_batch = None
            self._validation_batch_cycle = prepare_stratified_validation_batch_cycle(
                validation_images,
                validation_labels,
                num_classes=num_classes,
                global_batch_size=validation_batch_size,
                seed=validation_batch_seed,
                num_devices=jax.local_device_count(),
            )
            self._validation_examples_per_gradient_evaluation = int(
                self._validation_batch_cycle.batch_size
            )
            self._validation_direction_cycle_length = int(
                self._validation_batch_cycle.cycle_length
            )
            if (
                self.validation_reanchor_interval
                % self._validation_direction_cycle_length
            ):
                raise ValueError(
                    "validation reanchor interval must contain complete cycles"
                )
            self._validation_batch_seed = int(self._validation_batch_cycle.seed)
            self._validation_batch_schedule_sha256 = (
                self._validation_batch_cycle.schedule_sha256
            )
        scorer_type = (
            ClassifierHeadGradientAlignmentScorer
            if parameter_scope == "classifier_head"
            else FullParameterGradientAlignmentScorer
        )
        self._scorer = scorer_type(
            apply_fn=apply_fn,
            template_params=template_params,
        )

    def _record_timing(self, name: str, started_at: float, value) -> None:
        if not self.profile_components:
            return
        leaves = jax.tree_util.tree_leaves(value)
        if leaves:
            leaves[0].block_until_ready()
        self._timing_seconds[name] += time.perf_counter() - started_at
        self._timing_calls[name] += 1

    def timing_summary(self) -> dict[str, dict[str, float | int]]:
        """Return synchronized component timings collected in profile mode."""

        return {
            name: {
                "seconds": self._timing_seconds[name],
                "calls": self._timing_calls[name],
            }
            for name in sorted(self._timing_seconds)
        }

    def timing_totals(self) -> dict[str, float]:
        """Return cumulative synchronized seconds by component."""

        return dict(self._timing_seconds)

    def execution_summary(self) -> dict[str, int | float | bool | str | None]:
        """Return exact action-path workload counters."""

        drift = self._validation_anchor_drift_summary()
        return {
            "action_enabled": self.action_enabled,
            "parameter_scope": self.parameter_scope,
            "base_method": self.base_method,
            "shuffled_control": self.shuffled_control,
            "score_start_optimizer_step": self._score_start_optimizer_step,
            "score_stop_optimizer_step": self._score_stop_optimizer_step,
            "action_start_optimizer_step": self._action_start_optimizer_step,
            "action_stop_optimizer_step": self._action_stop_optimizer_step,
            "validation_direction_mode": self.validation_direction_mode,
            "validation_pool_examples": self._validation_pool_examples,
            "validation_examples_per_gradient_evaluation": (
                self._validation_examples_per_gradient_evaluation
            ),
            "validation_direction_cycle_length": (
                self._validation_direction_cycle_length
            ),
            "validation_reanchor_interval": (
                None
                if self.validation_direction_mode == "full"
                else self.validation_reanchor_interval
            ),
            "validation_batch_seed": self._validation_batch_seed,
            "validation_initial_optimizer_step": (
                None
                if self.validation_direction_mode == "full"
                else self._initial_optimizer_step
            ),
            "validation_batch_schedule_sha256": (
                self._validation_batch_schedule_sha256
            ),
            "train_steps": self._train_steps,
            "scored_steps": self._scored_steps,
            "action_active_steps": self._action_active_steps,
            "direction_refreshes": self._direction_refreshes,
            "validation_gradient_evaluations": (self._validation_gradient_evaluations),
            "validation_exact_reanchors": self._validation_exact_reanchors,
            **drift,
            "direction_validation_example_visits": (
                self._direction_validation_example_visits
            ),
        }

    def _validation_anchor_drift_summary(self) -> dict[str, int | float | None]:
        """Synchronize aggregate drift statistics only when a summary is read."""

        count = self._validation_anchor_drift_comparisons
        if count == 0:
            return {
                "validation_anchor_drift_comparisons": 0,
                "validation_anchor_stale_to_exact_cosine_mean": None,
                "validation_anchor_stale_to_exact_cosine_min": None,
                "validation_anchor_stale_to_exact_relative_l2_mean": None,
                "validation_anchor_stale_to_exact_relative_l2_max": None,
            }

        def scalar(value) -> float:
            """Read one deferred device scalar for the final artifact."""

            return float(np.asarray(jax.device_get(value)))

        return {
            "validation_anchor_drift_comparisons": count,
            "validation_anchor_stale_to_exact_cosine_mean": scalar(
                self._validation_anchor_cosine_sum / count
            ),
            "validation_anchor_stale_to_exact_cosine_min": scalar(
                self._validation_anchor_cosine_min
            ),
            "validation_anchor_stale_to_exact_relative_l2_mean": scalar(
                self._validation_anchor_relative_l2_sum / count
            ),
            "validation_anchor_stale_to_exact_relative_l2_max": scalar(
                self._validation_anchor_relative_l2_max
            ),
        }

    # #### GA VALIDATION DIRECTION SCHEDULE: START ####
    def _compute_validation_direction(self, state, batch):
        """Compute and account for one equal-weight validation-batch gradient."""

        direction = self._scorer.distributed_validation_direction_replicated(
            state,
            batch,
            verify_count_on_host=self.audit_mode,
        )
        self._validation_gradient_evaluations += 1
        self._direction_validation_example_visits += batch.example_count
        return direction

    def _reanchor_validation_aggregate(self, state):
        """Recompute every balanced component at one common model state."""

        if self._validation_batch_cycle is None:
            raise AssertionError("validation batch cycle was not prepared")
        stale_direction = self._validation_aggregate_direction
        directions = [
            self._compute_validation_direction(state, batch)
            for batch in self._validation_batch_cycle.batches
        ]
        self._validation_direction_cache = directions
        self._validation_aggregate_direction = _mean_replicated_directions(directions)
        if stale_direction is not None:
            cosine, relative_l2 = _replicated_direction_geometry(
                stale_direction,
                self._validation_aggregate_direction,
            )
            if self._validation_anchor_drift_comparisons == 0:
                self._validation_anchor_cosine_sum = cosine
                self._validation_anchor_cosine_min = cosine
                self._validation_anchor_relative_l2_sum = relative_l2
                self._validation_anchor_relative_l2_max = relative_l2
            else:
                self._validation_anchor_cosine_sum = (
                    self._validation_anchor_cosine_sum + cosine
                )
                self._validation_anchor_cosine_min = jnp.minimum(
                    self._validation_anchor_cosine_min,
                    cosine,
                )
                self._validation_anchor_relative_l2_sum = (
                    self._validation_anchor_relative_l2_sum + relative_l2
                )
                self._validation_anchor_relative_l2_max = jnp.maximum(
                    self._validation_anchor_relative_l2_max,
                    relative_l2,
                )
            self._validation_anchor_drift_comparisons += 1
        self._validation_exact_reanchors += 1
        return self._validation_aggregate_direction

    def _validation_direction_for_state(self, state):
        """Return the exact full direction or a cyclic aggregate estimator."""

        if self.validation_direction_mode == "full":
            if self._validation_batch is None:
                raise AssertionError("full validation batch was not prepared")
            return self._compute_validation_direction(
                state,
                self._validation_batch,
            )
        if self._validation_batch_cycle is None:
            raise AssertionError("validation batch cycle was not prepared")
        expected_step = self._initial_optimizer_step + self._train_steps
        if self.audit_mode:
            observed_step = _replicated_optimizer_step(state.step)
            if observed_step != expected_step:
                raise ValueError(
                    "optimizer step is discontinuous for validation batching: "
                    f"observed {observed_step}, expected {expected_step}"
                )
        if (
            self._validation_direction_cache is None
            or expected_step % self.validation_reanchor_interval == 0
        ):
            return self._reanchor_validation_aggregate(state)

        batch_index = expected_step % self._validation_direction_cycle_length
        new_direction = self._compute_validation_direction(
            state,
            self._validation_batch_cycle.batches[batch_index],
        )
        old_direction = self._validation_direction_cache[batch_index]
        self._validation_aggregate_direction = _replace_direction_in_mean(
            self._validation_aggregate_direction,
            old_direction,
            new_direction,
            count=self._validation_direction_cycle_length,
        )
        self._validation_direction_cache[batch_index] = new_direction
        return self._validation_aggregate_direction
    # #### GA VALIDATION DIRECTION SCHEDULE: END ####

    def _learning_rate(self, state) -> jax.Array:
        value = jnp.asarray(self.learning_rate_fn(state.step), dtype=jnp.float32)
        if value.ndim == 0:
            value = jnp.full((jax.local_device_count(),), value, dtype=jnp.float32)
        if value.shape != (jax.local_device_count(),):
            raise ValueError(
                "learning_rate_fn must return one replicated scalar per device"
            )
        return value

    def _standard_mixup_step(self, task_state, images, labels, rng):
        return parallel_train_step(
            task_state,
            rng,
            images,
            labels,
            {},
            self._mixer_fn,
            self.base_method,
            self.num_classes,
            0.5,
            -1.0,
            False,
            False,
            False,
            True,
            False,
        )

    def train_step(self, task_state, images, labels, rng):
        """Score and update one standard repository origin or MixUp batch."""

        optimizer_step = self._initial_optimizer_step + self._train_steps
        if not self.action_enabled:
            started_at = time.perf_counter()
            state, loss, accuracy = self._standard_mixup_step(
                task_state,
                images,
                labels,
                rng,
            )
            self._record_timing("update", started_at, loss)
            self._train_steps += 1
            return state, loss, accuracy, {}
        if (
            optimizer_step < self._score_start_optimizer_step
            or (
                self._score_stop_optimizer_step is not None
                and optimizer_step >= self._score_stop_optimizer_step
            )
        ):
            started_at = time.perf_counter()
            state, loss, accuracy = self._standard_mixup_step(
                task_state,
                images,
                labels,
                rng,
            )
            self._record_timing("update", started_at, loss)
            self._train_steps += 1
            return state, loss, accuracy, {}

        action_phase_active = (
            optimizer_step >= self._action_start_optimizer_step
            and (
                self._action_stop_optimizer_step is None
                or optimizer_step < self._action_stop_optimizer_step
            )
            and self.policy.mode != "score_only"
        )

        started_at = time.perf_counter()
        (
            mixed_images,
            labels_a,
            _labels_b,
            lam,
            soft_targets,
            dropout_rng,
        ) = self._extract_batch(rng, images, labels)
        self._record_timing("augmentation_mix", started_at, mixed_images)

        # #### GA STEP DIRECTION REFRESH: START ####
        started_at = time.perf_counter()
        direction = self._validation_direction_for_state(task_state)
        self._direction_refreshes += 1
        self._record_timing("vdev_direction", started_at, direction)
        # #### GA STEP DIRECTION REFRESH: END ####

        batch = SimpleNamespace(
            images=mixed_images,
            soft_targets=soft_targets,
            dropout_keys=dropout_rng,
        )
        # #### GA STEP SCORE DISPATCH: START ####
        # Dispatch J_theta_S logits(x; theta) @ u_B to the selected parameter
        # scope S; the scorer projects it into gains and optional utility.
        started_at = time.perf_counter()
        sample_weight_scores = None
        if self.policy.mode == "reweight":
            (
                raw_gains,
                sample_weight_scores,
            ) = self._scorer.score_labels_and_sample_weights_device_replicated(
                task_state.params,
                task_state.batch_stats,
                batch,
                direction,
            )
        else:
            raw_gains = self._scorer.score_hard_labels_device_replicated(
                task_state.params,
                task_state.batch_stats,
                batch,
                direction,
            )
        self._record_timing("jvp", started_at, raw_gains)
        # #### GA STEP SCORE DISPATCH: END ####

        # #### GA STEP POLICY DECISION: START ####
        started_at = time.perf_counter()
        learning_rate = self._learning_rate(task_state)
        metrics = None
        if self.shuffled_control:
            decision = decide_per_row_continuous_device(
                raw_gains,
                soft_targets,
                learning_rate,
                self.policy,
                sample_weight_scores=sample_weight_scores,
            )
        else:
            (
                decision,
                metrics,
            ) = decide_and_summarize_per_row_continuous_device(
                raw_gains,
                soft_targets,
                learning_rate,
                self.policy,
                sample_weight_scores=sample_weight_scores,
            )
        if self.shuffled_control and action_phase_active:
            decision = _shuffle_continuous_decision(
                decision,
                soft_targets,
                task_state.step,
                self.policy.mode,
                self.control_seed,
            )
        if not action_phase_active and self.policy.mode != "score_only":
            decision = decision._replace(
                targets_after=soft_targets,
                weights=jnp.ones_like(decision.weights),
                applied_rows=jnp.zeros_like(decision.applied_rows),
                doses=jnp.zeros_like(decision.doses),
                fallback_applied=jnp.zeros_like(decision.fallback_applied),
            )
            metrics = None
        self._record_timing("policy", started_at, decision.targets_after)
        # #### GA STEP POLICY DECISION: END ####

        # #### GA STEP ACTION UPDATE: START ####
        # Score-only/inactive phases keep the ordinary target. Active phases
        # feed the selected target and mean-one weight into the shared update.
        started_at = time.perf_counter()
        if self.policy.mode == "score_only" or not action_phase_active:
            state, loss, accuracy = self._precomputed_base_step(
                task_state,
                mixed_images,
                labels_a,
                _labels_b,
                lam,
                dropout_rng,
            )
        else:
            state, loss, accuracy = self._continuous_step(
                task_state,
                mixed_images,
                decision.targets_after,
                decision.weights,
                labels_a,
                dropout_rng,
            )
        self._record_timing("update", started_at, loss)
        # #### GA STEP ACTION UPDATE: END ####

        if metrics is None:
            started_at = time.perf_counter()
            metrics = _summarize_decision(decision)
            self._record_timing("policy", started_at, metrics)
        self._train_steps += 1
        self._scored_steps += 1
        if action_phase_active:
            self._action_active_steps += 1
        return state, loss, accuracy, metrics
