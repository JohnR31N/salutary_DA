"""Device-resident continuous actions for every-batch gradient alignment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DECISION_SELECTED = 0
DECISION_GAIN_BELOW_THRESHOLD = 1
DECISION_MARGIN_BELOW_THRESHOLD = 2
DECISION_INVALID_SCORE = 3
DECISION_BUDGET_EXCLUDED = 4


@dataclass(frozen=True)
class PerRowContinuousPolicyConfig:
    """Configuration for soft-label, reweight, or score-only row decisions."""

    mode: str
    maximum_rows: int | None = None
    soft_label_dose: float = 0.01
    maximum_weight_deviation: float = 0.05
    weight_temperature: float = 1.0
    minimum_relative_ess: float = 0.9
    minimum_gain: float = 0.0
    minimum_label_margin: float = 0.0
    minimum_relative_label_margin: float = 0.0
    fallback_enabled: bool = False
    fallback_soft_label_dose: float = 0.01

    def __post_init__(self) -> None:
        if self.mode not in {"score_only", "soft_label", "reweight"}:
            raise ValueError(
                "continuous policy mode must be score_only, soft_label, or reweight"
            )
        if self.maximum_rows is not None and (
            isinstance(self.maximum_rows, bool)
            or not isinstance(self.maximum_rows, int)
            or self.maximum_rows <= 0
        ):
            raise ValueError("maximum_rows must be a positive integer or None")
        if not np.isfinite(self.soft_label_dose) or not (
            0.0 < self.soft_label_dose <= 1.0
        ):
            raise ValueError("soft_label_dose must be finite and within (0, 1]")
        if not np.isfinite(self.fallback_soft_label_dose) or not (
            0.0 < self.fallback_soft_label_dose <= 1.0
        ):
            raise ValueError(
                "fallback_soft_label_dose must be finite and within (0, 1]"
            )
        if not np.isfinite(self.maximum_weight_deviation) or not (
            0.0 < self.maximum_weight_deviation < 1.0
        ):
            raise ValueError(
                "maximum_weight_deviation must be finite and within (0, 1)"
            )
        if not np.isfinite(self.weight_temperature) or self.weight_temperature <= 0:
            raise ValueError("weight_temperature must be finite and positive")
        if not np.isfinite(self.minimum_relative_ess) or not (
            0.0 < self.minimum_relative_ess <= 1.0
        ):
            raise ValueError("minimum_relative_ess must be within (0, 1]")
        thresholds = (
            self.minimum_gain,
            self.minimum_label_margin,
            self.minimum_relative_label_margin,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in thresholds):
            raise ValueError(
                "continuous policy thresholds must be finite and nonnegative"
            )
        if self.fallback_enabled and self.mode != "soft_label":
            raise ValueError("the every-batch fallback is defined only for soft_label")


class PerRowContinuousPolicyDecision(NamedTuple):
    """Arrays produced by one continuous-policy decision."""

    targets_after: jax.Array
    weights: jax.Array
    selected_labels: jax.Array
    eligible_rows: jax.Array
    applied_rows: jax.Array
    decision_codes: jax.Array
    scaled_best_gains: jax.Array
    scaled_weight_scores: jax.Array
    label_margins: jax.Array
    doses: jax.Array
    relative_ess: jax.Array
    eligible_count: jax.Array
    applied_count: jax.Array
    fallback_applied: jax.Array
    scores_valid: jax.Array


def _relative_ess(weights: jax.Array) -> jax.Array:
    count = jnp.asarray(weights.size, dtype=weights.dtype)
    return jnp.square(jnp.sum(weights)) / (
        count * jnp.sum(jnp.square(weights))
    )


def _bounded_mean_one_weights(
    scores: jax.Array,
    *,
    maximum_deviation: float,
    temperature: float,
    minimum_relative_ess: float,
) -> jax.Array:
    """Map scores monotonically to bounded mean-one weights on device."""

    median = jnp.median(scores)
    mad_scale = 1.4826 * jnp.median(jnp.abs(scores - median))
    std_scale = jnp.std(scores)
    scale = jnp.where(mad_scale > 1e-12, mad_scale, std_scale)
    standardized = jnp.where(
        scale > 1e-12,
        jnp.clip((scores - median) / scale, -3.0, 3.0),
        jnp.zeros_like(scores),
    )
    logits = standardized / jnp.asarray(temperature, dtype=scores.dtype)
    minimum = jnp.asarray(1.0 - maximum_deviation, dtype=scores.dtype)
    maximum = jnp.asarray(1.0 + maximum_deviation, dtype=scores.dtype)

    def offset_body(_index, bounds):
        lower, upper = bounds
        offset = 0.5 * (lower + upper)
        sigmoid = jax.nn.sigmoid(logits + offset)
        mean_weight = jnp.mean(minimum + (maximum - minimum) * sigmoid)
        return (
            jnp.where(mean_weight < 1.0, offset, lower),
            jnp.where(mean_weight < 1.0, upper, offset),
        )

    lower, upper = jax.lax.fori_loop(
        0,
        48,
        offset_body,
        (
            jnp.asarray(-64.0, dtype=scores.dtype),
            jnp.asarray(64.0, dtype=scores.dtype),
        ),
    )
    offset = 0.5 * (lower + upper)
    weights = minimum + (maximum - minimum) * jax.nn.sigmoid(logits + offset)

    def ess_body(_index, bounds):
        accepted, rejected = bounds
        amount = 0.5 * (accepted + rejected)
        trial = 1.0 + amount * (weights - 1.0)
        passes = _relative_ess(trial) >= minimum_relative_ess
        return (
            jnp.where(passes, amount, accepted),
            jnp.where(passes, rejected, amount),
        )

    accepted, _ = jax.lax.fori_loop(
        0,
        40,
        ess_body,
        (
            jnp.asarray(0.0, dtype=scores.dtype),
            jnp.asarray(1.0, dtype=scores.dtype),
        ),
    )
    shrink = jnp.where(
        _relative_ess(weights) >= minimum_relative_ess,
        jnp.asarray(1.0, dtype=scores.dtype),
        accepted,
    )
    return 1.0 + shrink * (weights - 1.0)


def bounded_mean_one_weights(
    scores: jax.Array,
    *,
    maximum_deviation: float,
    temperature: float,
    minimum_relative_ess: float,
) -> jax.Array:
    """Public checked map from one score vector to bounded mean-one weights."""

    values = jnp.asarray(scores)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weight scores must be a non-empty vector")
    scalar_values = (maximum_deviation, temperature, minimum_relative_ess)
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("weight policy scalars must be finite")
    if not 0.0 < maximum_deviation < 1.0:
        raise ValueError("maximum weight deviation must be within (0,1)")
    if temperature <= 0.0:
        raise ValueError("weight temperature must be positive")
    if not 0.0 < minimum_relative_ess <= 1.0:
        raise ValueError("minimum relative ESS must be within (0,1]")
    return _bounded_mean_one_weights(
        values,
        maximum_deviation=maximum_deviation,
        temperature=temperature,
        minimum_relative_ess=minimum_relative_ess,
    )


def _decide_global_arrays(
    raw_gains: jax.Array,
    soft_targets: jax.Array,
    sample_weight_scores: jax.Array,
    learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
) -> PerRowContinuousPolicyDecision:
    if raw_gains.ndim != 2 or soft_targets.shape != raw_gains.shape:
        raise ValueError("raw gains and soft targets must share shape [rows, classes]")
    if raw_gains.shape[1] < 2:
        raise ValueError("continuous policy requires at least two classes")
    if sample_weight_scores.shape != (raw_gains.shape[0],):
        raise ValueError("sample_weight_scores must have shape [rows]")

    # #### GA POLICY ROW SELECTION: START ####
    # Rank classes independently within each row, then apply the batch-wide
    # finite-value gate plus per-row gain and margin gates. Only an enabled
    # soft-label row budget globally ranks the eligible rows against each other.
    row_count, class_count = raw_gains.shape
    scaled = raw_gains * (
        jnp.asarray(learning_rate, dtype=raw_gains.dtype) / row_count
    )
    scaled_weight_scores = sample_weight_scores * (
        jnp.asarray(learning_rate, dtype=raw_gains.dtype) / row_count
    )
    scores_valid = jnp.all(jnp.isfinite(scaled)) & jnp.all(
        jnp.isfinite(soft_targets)
    )
    if config.mode == "reweight":
        scores_valid = scores_valid & jnp.all(jnp.isfinite(scaled_weight_scores))
    mutable = soft_targets < jnp.asarray(1.0 - 1e-7, dtype=soft_targets.dtype)
    ranked = jnp.where(mutable, scaled, -jnp.inf)
    top_values, top_labels = jax.lax.top_k(ranked, 2)
    best_gains = top_values[:, 0]
    label_margins = top_values[:, 0] - top_values[:, 1]
    selected_labels = top_labels[:, 0].astype(jnp.int32)
    required_margin = jnp.maximum(
        jnp.asarray(config.minimum_label_margin, dtype=scaled.dtype),
        jnp.asarray(
            config.minimum_relative_label_margin,
            dtype=scaled.dtype,
        )
        * best_gains,
    )
    gain_pass = best_gains > config.minimum_gain
    margin_pass = label_margins > required_margin
    eligible = scores_valid & gain_pass & margin_pass
    decision_codes = jnp.where(
        ~scores_valid,
        DECISION_INVALID_SCORE,
        jnp.where(
            ~gain_pass,
            DECISION_GAIN_BELOW_THRESHOLD,
            jnp.where(
                ~margin_pass,
                DECISION_MARGIN_BELOW_THRESHOLD,
                DECISION_SELECTED,
            ),
        ),
    ).astype(jnp.int32)
    if config.mode == "reweight":
        eligible = jnp.full((row_count,), scores_valid, dtype=jnp.bool_)
        decision_codes = jnp.where(
            scores_valid,
            jnp.asarray(DECISION_SELECTED, dtype=jnp.int32),
            jnp.asarray(DECISION_INVALID_SCORE, dtype=jnp.int32),
        ) * jnp.ones((row_count,), dtype=jnp.int32)

    selected_for_action = eligible
    if config.mode == "soft_label" and config.maximum_rows is not None:
        action_budget = min(config.maximum_rows, row_count)
        ranked_rows = jnp.argsort(
            -jnp.where(eligible, best_gains, -jnp.inf),
            stable=True,
        )
        selected_for_action = jnp.zeros((row_count,), dtype=jnp.bool_).at[
            ranked_rows[:action_budget]
        ].set(True) & eligible
        decision_codes = jnp.where(
            eligible & ~selected_for_action,
            jnp.asarray(DECISION_BUDGET_EXCLUDED, dtype=jnp.int32),
            decision_codes,
        )
    # #### GA POLICY ROW SELECTION: END ####

    # #### GA POLICY ACTION MATERIALIZATION: START ####
    # Soft-label actions use (1-beta)t + beta*e_c. Reweight actions preserve
    # mean one and satisfy the configured relative effective-sample-size floor.
    fallback = jnp.zeros((row_count,), dtype=jnp.bool_)
    if config.fallback_enabled:
        fallback_index = jnp.argmax(best_gains)
        fallback = (
            scores_valid
            & ~jnp.any(eligible)
            & (jnp.arange(row_count) == fallback_index)
        )

    targets_after = soft_targets
    weights = jnp.ones((row_count,), dtype=soft_targets.dtype)
    doses = jnp.zeros((row_count,), dtype=soft_targets.dtype)
    applied = jnp.zeros((row_count,), dtype=jnp.bool_)
    if config.mode == "soft_label":
        applied = selected_for_action | fallback
        doses = jnp.where(
            selected_for_action,
            jnp.asarray(config.soft_label_dose, dtype=soft_targets.dtype),
            jnp.where(
                fallback,
                jnp.asarray(
                    config.fallback_soft_label_dose,
                    dtype=soft_targets.dtype,
                ),
                jnp.asarray(0.0, dtype=soft_targets.dtype),
            ),
        )
        vertices = jax.nn.one_hot(
            selected_labels,
            class_count,
            dtype=soft_targets.dtype,
        )
        targets_after = (
            (1.0 - doses[:, None]) * soft_targets
            + doses[:, None] * vertices
        )
    elif config.mode == "reweight":
        proposed = _bounded_mean_one_weights(
            scaled_weight_scores,
            maximum_deviation=config.maximum_weight_deviation,
            temperature=config.weight_temperature,
            minimum_relative_ess=config.minimum_relative_ess,
        )
        weights = jnp.where(scores_valid, proposed, jnp.ones_like(proposed))
        applied = scores_valid & (jnp.abs(weights - 1.0) > 1e-7)
    # #### GA POLICY ACTION MATERIALIZATION: END ####

    selected_labels = jnp.where(
        scores_valid,
        selected_labels,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    return PerRowContinuousPolicyDecision(
        targets_after=targets_after,
        weights=weights,
        selected_labels=selected_labels,
        eligible_rows=eligible,
        applied_rows=applied,
        decision_codes=decision_codes,
        scaled_best_gains=best_gains,
        scaled_weight_scores=scaled_weight_scores,
        label_margins=label_margins,
        doses=doses,
        relative_ess=_relative_ess(weights),
        eligible_count=jnp.sum(eligible, dtype=jnp.int32),
        applied_count=jnp.sum(applied, dtype=jnp.int32),
        fallback_applied=jnp.any(fallback),
        scores_valid=scores_valid,
    )


def _decide_sharded_arrays_in_pmap(
    local_raw_gains: jax.Array,
    local_soft_targets: jax.Array,
    local_sample_weight_scores: jax.Array,
    local_learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
) -> PerRowContinuousPolicyDecision:
    """Gather scores and return a local decision inside the named PMAP axis."""

    gathered_gains = jax.lax.all_gather(local_raw_gains, "batch")
    gathered_targets = jax.lax.all_gather(local_soft_targets, "batch")
    gathered_weight_scores = jax.lax.all_gather(
        local_sample_weight_scores,
        "batch",
    )
    global_gains = gathered_gains.reshape((-1, gathered_gains.shape[-1]))
    global_targets = gathered_targets.reshape((-1, gathered_targets.shape[-1]))
    global_weight_scores = gathered_weight_scores.reshape(-1)
    learning_rate = jax.lax.pmean(local_learning_rate, "batch")
    global_decision = _decide_global_arrays(
        global_gains,
        global_targets,
        global_weight_scores,
        learning_rate,
        config,
    )
    local_batch = local_raw_gains.shape[0]
    start = jax.lax.axis_index("batch") * local_batch

    def local_rows(value):
        if value.ndim == 0:
            return value
        return jax.lax.dynamic_slice_in_dim(value, start, local_batch, axis=0)

    return jax.tree_util.tree_map(local_rows, global_decision)


def summarize_per_row_continuous_decision_in_pmap(
    decision: PerRowContinuousPolicyDecision,
) -> dict[str, jax.Array]:
    """Summarize one local decision with the established collective order."""

    local_rows = jnp.asarray(decision.applied_rows.size, dtype=jnp.float32)
    global_rows = jax.lax.psum(local_rows, "batch")
    scores_valid = decision.scores_valid.astype(jnp.float32)
    valid_best = jnp.where(
        decision.scores_valid,
        decision.scaled_best_gains,
        jnp.zeros_like(decision.scaled_best_gains),
    )
    valid_margin = jnp.where(
        decision.scores_valid,
        decision.label_margins,
        jnp.zeros_like(decision.label_margins),
    )
    valid_weight_scores = jnp.where(
        decision.scores_valid,
        decision.scaled_weight_scores,
        jnp.zeros_like(decision.scaled_weight_scores),
    )
    eligible_count = jax.lax.psum(
        jnp.sum(decision.eligible_rows, dtype=jnp.float32),
        "batch",
    )
    applied_count = jax.lax.psum(
        jnp.sum(decision.applied_rows, dtype=jnp.float32),
        "batch",
    )
    reason_fractions = {
        "salda_gain_abstention_fraction": DECISION_GAIN_BELOW_THRESHOLD,
        "salda_margin_abstention_fraction": DECISION_MARGIN_BELOW_THRESHOLD,
        "salda_budget_excluded_fraction": DECISION_BUDGET_EXCLUDED,
        "salda_invalid_fraction": DECISION_INVALID_SCORE,
    }
    return {
        "salda_scored_fraction": jnp.asarray(1.0, dtype=jnp.float32),
        "salda_scores_valid": jax.lax.pmin(scores_valid, "batch"),
        "salda_eligible_fraction": eligible_count / global_rows,
        "salda_applied_fraction": applied_count / global_rows,
        "salda_batch_action_coverage": (applied_count > 0).astype(jnp.float32),
        "salda_fallback_fraction": decision.fallback_applied.astype(jnp.float32),
        "salda_score_mean": jax.lax.psum(jnp.sum(valid_best), "batch")
        / global_rows,
        "salda_score_min": jax.lax.pmin(jnp.min(valid_best), "batch"),
        "salda_score_max": jax.lax.pmax(jnp.max(valid_best), "batch"),
        "salda_label_margin_mean": jax.lax.psum(jnp.sum(valid_margin), "batch")
        / global_rows,
        "salda_dose_mean": jax.lax.psum(jnp.sum(decision.doses), "batch")
        / global_rows,
        "salda_weight_mean": jax.lax.psum(jnp.sum(decision.weights), "batch")
        / global_rows,
        "salda_weight_score_mean": jax.lax.psum(
            jnp.sum(valid_weight_scores),
            "batch",
        )
        / global_rows,
        "salda_weight_score_min": jax.lax.pmin(
            jnp.min(valid_weight_scores),
            "batch",
        ),
        "salda_weight_score_max": jax.lax.pmax(
            jnp.max(valid_weight_scores),
            "batch",
        ),
        "salda_weight_min": jax.lax.pmin(jnp.min(decision.weights), "batch"),
        "salda_weight_max": jax.lax.pmax(jnp.max(decision.weights), "batch"),
        "salda_weight_relative_ess": decision.relative_ess,
        **{
            name: jax.lax.psum(
                jnp.sum(decision.decision_codes == code, dtype=jnp.float32),
                "batch",
            )
            / global_rows
            for name, code in reason_fractions.items()
        },
    }


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def _decide_sharded_arrays(
    local_raw_gains: jax.Array,
    local_soft_targets: jax.Array,
    local_sample_weight_scores: jax.Array,
    local_learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
) -> PerRowContinuousPolicyDecision:
    """Gather one small score matrix and return each device's local decision."""

    return _decide_sharded_arrays_in_pmap(
        local_raw_gains,
        local_soft_targets,
        local_sample_weight_scores,
        local_learning_rate,
        config,
    )


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def _decide_and_summarize_sharded_arrays(
    local_raw_gains: jax.Array,
    local_soft_targets: jax.Array,
    local_sample_weight_scores: jax.Array,
    local_learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
) -> tuple[PerRowContinuousPolicyDecision, dict[str, jax.Array]]:
    """Return the unchanged local decision and summary from one PMAP."""

    decision = _decide_sharded_arrays_in_pmap(
        local_raw_gains,
        local_soft_targets,
        local_sample_weight_scores,
        local_learning_rate,
        config,
    )
    decision_for_summary = decision
    if config.mode == "reweight":
        # Preserve the f32 decision boundary before a symmetric score sum.
        gathered_weight_scores = jax.lax.all_gather(
            decision.scaled_weight_scores,
            "batch",
        )
        local_weight_scores = jax.lax.dynamic_index_in_dim(
            gathered_weight_scores,
            jax.lax.axis_index("batch"),
            axis=0,
            keepdims=False,
        )
        decision_for_summary = decision._replace(
            scaled_weight_scores=local_weight_scores
        )
    return decision, summarize_per_row_continuous_decision_in_pmap(
        decision_for_summary
    )


def _validate_device_policy_inputs(
    raw_gains: jax.Array,
    soft_targets: jax.Array,
    learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
    sample_weight_scores: jax.Array | None,
) -> jax.Array:
    """Validate device-policy arrays and materialize the optional score input."""

    if raw_gains.ndim != 3 or soft_targets.shape != raw_gains.shape:
        raise ValueError(
            "sharded raw gains and soft targets must share [devices, rows, classes]"
        )
    if raw_gains.shape[0] != jax.local_device_count():
        raise ValueError("leading score dimension must equal local device count")
    if learning_rate.shape != (jax.local_device_count(),):
        raise ValueError("learning_rate must contain one replicated scalar per device")
    if sample_weight_scores is None:
        if config.mode == "reweight":
            raise ValueError("reweight mode requires current-target sample scores")
        sample_weight_scores = jnp.zeros(raw_gains.shape[:2], dtype=raw_gains.dtype)
    if sample_weight_scores.shape != raw_gains.shape[:2]:
        raise ValueError("sharded sample weight scores must have shape [devices, rows]")
    return sample_weight_scores


def decide_per_row_continuous_device(
    raw_gains: jax.Array,
    soft_targets: jax.Array,
    learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
    sample_weight_scores: jax.Array | None = None,
) -> PerRowContinuousPolicyDecision:
    """Apply the continuous policy to PMAP-sharded score and target arrays."""

    sample_weight_scores = _validate_device_policy_inputs(
        raw_gains,
        soft_targets,
        learning_rate,
        config,
        sample_weight_scores,
    )
    return _decide_sharded_arrays(
        raw_gains,
        soft_targets,
        sample_weight_scores,
        learning_rate,
        config,
    )


def decide_and_summarize_per_row_continuous_device(
    raw_gains: jax.Array,
    soft_targets: jax.Array,
    learning_rate: jax.Array,
    config: PerRowContinuousPolicyConfig,
    sample_weight_scores: jax.Array | None = None,
) -> tuple[PerRowContinuousPolicyDecision, dict[str, jax.Array]]:
    """Apply and summarize the device policy with one PMAP dispatch."""

    sample_weight_scores = _validate_device_policy_inputs(
        raw_gains,
        soft_targets,
        learning_rate,
        config,
        sample_weight_scores,
    )
    return _decide_and_summarize_sharded_arrays(
        raw_gains,
        soft_targets,
        sample_weight_scores,
        learning_rate,
        config,
    )


def decide_per_row_continuous_reference(
    raw_gains: np.ndarray,
    soft_targets: np.ndarray,
    *,
    learning_rate: float,
    config: PerRowContinuousPolicyConfig,
    sample_weight_scores: np.ndarray | None = None,
) -> PerRowContinuousPolicyDecision:
    """Return a host-materialized reference decision for audits and tests."""

    if sample_weight_scores is None:
        if config.mode == "reweight":
            raise ValueError("reweight mode requires current-target sample scores")
        sample_weight_scores = np.zeros(np.asarray(raw_gains).shape[0], np.float32)
    decision = _decide_global_arrays(
        jnp.asarray(raw_gains),
        jnp.asarray(soft_targets),
        jnp.asarray(sample_weight_scores),
        jnp.asarray(learning_rate, dtype=jnp.float32),
        config,
    )
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), decision)
