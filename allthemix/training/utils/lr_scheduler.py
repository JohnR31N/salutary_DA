from __future__ import annotations

from collections.abc import Sequence

import optax


def build_step_lr_schedule(
    base_learning_rate: float,
    steps_per_epoch: int,
    decay_epochs: Sequence[int],
    decay_rate: float,
) -> optax.Schedule:
    """Build a step-decay learning-rate schedule."""
    boundaries_and_scales = {}

    for epoch in decay_epochs:
        step = epoch * steps_per_epoch  # Convert epoch boundary to update step.
        boundaries_and_scales[step] = decay_rate

    schedule = optax.piecewise_constant_schedule(
        init_value=base_learning_rate,
        boundaries_and_scales=boundaries_and_scales,
    )

    return schedule


def build_cosine_lr_schedule(
    base_learning_rate: float,
    min_learning_rate: float,
    total_steps: int,
) -> optax.Schedule:
    """Build a cosine-decay learning-rate schedule."""
    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive for cosine learning-rate schedule."
        )

    if base_learning_rate <= 0:
        return optax.constant_schedule(
            value=base_learning_rate,
        )

    alpha = min_learning_rate / base_learning_rate  # Optax cosine floor as LR ratio.

    return optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=total_steps,
        alpha=alpha,
    )


def build_warmup_cosine_lr_schedule(
    base_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """Build a linear-warmup plus cosine-decay learning-rate schedule."""
    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps must be non-negative for warmup cosine schedule."
        )

    if warmup_steps == 0:
        return build_cosine_lr_schedule(
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            total_steps=total_steps,
        )

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive for warmup cosine schedule."
        )

    if warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must be smaller than total_steps for warmup "
            "cosine schedule."
        )

    if base_learning_rate <= 0:
        return optax.constant_schedule(
            value=base_learning_rate,
        )

    alpha = min_learning_rate / base_learning_rate  # Optax cosine floor as LR ratio.
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_steps,
    )
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=total_steps - warmup_steps,
        alpha=alpha,
    )

    return optax.join_schedules(
        schedules=[
            warmup_schedule,
            cosine_schedule,
        ],
        boundaries=[
            warmup_steps,
        ],
    )


def build_step_cosine_lr_schedule(
    base_learning_rate: float,
    min_learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
    decay_epochs: Sequence[int],
) -> optax.Schedule:
    """Build a plateau schedule followed by cosine decay."""
    if not decay_epochs:
        raise ValueError(
            "step_cosine lr schedule requires at least one lr_decay_epoch."
        )

    total_steps = steps_per_epoch * epochs
    plateau_steps = decay_epochs[0] * steps_per_epoch

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive for step cosine schedule."
        )

    if plateau_steps < 0 or plateau_steps >= total_steps:
        raise ValueError(
            "The first lr_decay_epoch must be in [0, epochs) for "
            "step_cosine schedule."
        )

    if base_learning_rate <= 0:
        return optax.constant_schedule(
            value=base_learning_rate,
        )

    constant_schedule = optax.constant_schedule(
        value=base_learning_rate,
    )
    cosine_schedule = build_cosine_lr_schedule(
        base_learning_rate=base_learning_rate,
        min_learning_rate=min_learning_rate,
        total_steps=total_steps - plateau_steps,
    )

    return optax.join_schedules(
        schedules=[
            constant_schedule,
            cosine_schedule,
        ],
        boundaries=[
            plateau_steps,
        ],
    )


def build_lr_schedule(
    schedule_name: str,
    base_learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
    decay_epochs: Sequence[int],
    decay_rate: float,
    min_learning_rate: float = 0.0,
    warmup_epochs: int = 0,
) -> optax.Schedule:
    """Build the requested learning-rate schedule."""
    schedule_name = schedule_name.lower()

    if schedule_name == "step":
        return build_step_lr_schedule(
            base_learning_rate=base_learning_rate,
            steps_per_epoch=steps_per_epoch,
            decay_epochs=decay_epochs,
            decay_rate=decay_rate,
        )

    if schedule_name == "cosine":
        return build_cosine_lr_schedule(
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            total_steps=steps_per_epoch * epochs,
        )

    if schedule_name in (
        "warmup_cosine",
        "linear_warmup_cosine",
    ):
        return build_warmup_cosine_lr_schedule(
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_epochs * steps_per_epoch,
            total_steps=steps_per_epoch * epochs,
        )

    if schedule_name in (
        "step_cosine",
        "plateau_cosine",
    ):
        return build_step_cosine_lr_schedule(
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            decay_epochs=decay_epochs,
        )

    raise ValueError(
        "Unsupported lr_schedule: "
        f"{schedule_name}. Expected step, cosine, warmup_cosine, "
        "or step_cosine."
    )
