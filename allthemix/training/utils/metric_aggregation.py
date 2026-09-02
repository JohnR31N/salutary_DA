from __future__ import annotations

import numpy as np


def aggregate_epoch_metric_lists(
    metric_lists: dict[str, list[float]],
) -> dict[str, float]:
    """Aggregate step metrics according to their declared statistic."""
    aggregated = {}

    for key, values in metric_lists.items():
        if not values:
            continue

        if key.endswith(
            "_min",
        ):
            value = np.min(
                values,
            )

        elif key.endswith(
            "_max",
        ):
            value = np.max(
                values,
            )

        else:
            value = np.mean(
                values,
            )

        aggregated[key] = float(
            value,
        )

    return aggregated
