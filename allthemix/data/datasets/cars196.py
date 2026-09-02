from __future__ import annotations

from allthemix.data.datasets.registry import CARS196, is_dataset

CARS196_NAME = CARS196
CARS196_DIR_CANDIDATES = (
    "cars196",
    "stanford_cars",
    "stanford-car-dataset",
    "car_data/car_data",
)


def is_cars196_name(
    name: str,
) -> bool:
    """Return whether a dataset name is the canonical Stanford Cars name."""
    return is_dataset(name, CARS196)
