from __future__ import annotations


def get_normalization_stats(
    dataset: str,
    tiny_imagenet_normalization: str = "imagenet",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the channel statistics used by AllTheMix preprocessing."""
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    if dataset_name == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    if dataset_name == "svhn_cropped":
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

    if dataset_name == "tiny_imagenet" and tiny_imagenet_normalization == "none":
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    if dataset_name in {
        "tiny_imagenet",
        "stl10",
        "oxford_iiit_pet",
        "cars196",
        "imagenet100",
        "caltech_birds2011",
    }:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    raise ValueError(
        f"Normalization statistics are not configured for dataset: {dataset}"
    )
