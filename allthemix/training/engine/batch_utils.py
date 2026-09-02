"""Batch-format utilities shared by the single and parallel engines."""

from __future__ import annotations

from typing import Any


def unpack_batch(
    batch: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Convert different dataset batch formats into a unified format:

        images, labels, aux_info

    Supported formats:

    1. Normal methods:
        batch = (images, labels)
        aux_info = {}

    2. SaliencyMix-style methods:
        batch = (images, labels, saliency_maps)
        aux_info = {"saliency_maps": saliency_maps}

    3. Future extensible format:
        batch = (images, labels, aux_info)
        aux_info is already a dict.
    """
    if isinstance(batch, dict):
        images = batch["images"]
        labels = batch["labels"]

        aux_info = {
            key: value
            for key, value in batch.items()
            if key not in {
                "images",
                "labels",
            }
        }

        return images, labels, aux_info

    if len(batch) == 2:
        images, labels = batch

        return images, labels, {}

    if len(batch) == 3:
        images, labels, third_item = batch

        if isinstance(third_item, dict):
            aux_info = third_item
        else:
            aux_info = {
                "saliency_maps": third_item,
            }

        return images, labels, aux_info

    raise ValueError(
        "Unsupported batch format. Expected either "
        "(images, labels), (images, labels, aux_info), "
        "or (images, labels, saliency_maps)."
    )
