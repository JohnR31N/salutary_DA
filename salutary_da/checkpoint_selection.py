"""Shared checkpoint-selection contract for instantaneous-GA runs."""

from __future__ import annotations

import math

STRICT_VDEV_TOP1_ERROR_RULE = (
    "strictly_lower_full_Vdev_top1_error_first_epoch_wins_ties"
)


def should_replace_best_vdev_top1_error(
    observed_top1_error: float,
    best_top1_error: float,
) -> bool:
    """Return whether the current epoch strictly improves full-Vdev error."""

    observed = float(observed_top1_error)
    best = float(best_top1_error)
    if not math.isfinite(observed):
        raise ValueError("observed Vdev top-1 error must be finite")
    if not (math.isfinite(best) or (math.isinf(best) and best > 0.0)):
        raise ValueError("best Vdev top-1 error must be finite or positive infinity")
    return observed < best
