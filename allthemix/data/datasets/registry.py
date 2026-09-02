"""Single source of dataset identity.

Every place that needs to know "which dataset is this name" goes through
``canonicalize``/``is_dataset`` instead of carrying its own lowercase
comparisons. Aliases, when a dataset ever grows one, are declared here and
nowhere else; per-dataset modules keep only their own constants (URLs,
directory candidates, class tables).
"""

from __future__ import annotations

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"
CARS196 = "cars196"
TINY_IMAGENET = "tiny_imagenet"
IMAGENET100 = "imagenet100"

# name -> canonical id; extend ONLY here when a dataset gains an alias.
DATASET_ALIASES: dict[str, str] = {}


def canonicalize(
    name: str,
) -> str:
    """Normalize a user-facing dataset name into its canonical identifier."""
    normalized = name.strip().lower()

    return DATASET_ALIASES.get(
        normalized,
        normalized,
    )


def is_dataset(
    name: str,
    canonical_id: str,
) -> bool:
    """Return whether ``name`` refers to the given canonical dataset."""
    return canonicalize(name) == canonical_id
