from __future__ import annotations

from allthemix.data.datasets.registry import TINY_IMAGENET, is_dataset

TINY_IMAGENET_NAME = TINY_IMAGENET
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_ZIP_NAME = "tiny-imagenet-200.zip"
TINY_IMAGENET_DIR_NAME = "tiny-imagenet-200"


def is_tiny_imagenet_name(
    name: str,
) -> bool:
    """Return whether a dataset name is the canonical Tiny ImageNet name."""
    return is_dataset(name, TINY_IMAGENET)
