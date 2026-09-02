"""Saliency map path, build, and cache loading helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from allthemix.data.datasets.cars196 import is_cars196_name
from allthemix.data.datasets.imagenet100 import is_imagenet100_name
from allthemix.data.datasets.loader import load_train_dataset
from allthemix.data.datasets.registry import canonicalize
from allthemix.data.datasets.tiny_imagenet import is_tiny_imagenet_name
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.data.saliency.saliency_methods import (
    compute_saliency_map,
)
from allthemix.data.utils.cardinality import resolve_source_train_example_count

logger = logging.getLogger(__name__)


def canonical_dataset_name(
    dataset_name: str,
) -> str:
    """Normalize a dataset name via the central registry."""
    return canonicalize(dataset_name)


def get_train_saliency_path(
    dataset_name: str,
    saliency_dir: str,
) -> Path:
    """Path of the cached train-set saliency map for a dataset."""
    dataset_name = canonical_dataset_name(
        dataset_name,
    )

    return (
        Path(saliency_dir)
        / f"{dataset_name}_train_saliency.npy"
    )


def _load_train_dataset_for_saliency_cache(
    dataset_name: str,
    data_dir: str,
):
    """Load train examples in the same order used when pairing saliency maps."""
    if is_tiny_imagenet_name(
        dataset_name,
    ) or is_cars196_name(
        dataset_name,
    ) or is_imagenet100_name(
        dataset_name,
    ):
        return load_train_dataset(
            name=dataset_name,
            data_dir=data_dir,
        )

    return tfds.load(
        name=dataset_name,
        split="train",
        data_dir=data_dir,
        shuffle_files=False,
        download=True,
    )


def _resize_saliency_map_to_dataset_size(
    saliency_map: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Resize a saliency map to the fixed model input size before caching."""
    if saliency_map.shape == (
        image_size,
        image_size,
    ):
        return saliency_map.astype(np.float32)

    saliency_tensor = tf.convert_to_tensor(
        saliency_map[:, :, None],
        dtype=tf.float32,
    )

    saliency_tensor = tf.image.resize(
        saliency_tensor,
        size=[
            image_size,
            image_size,
        ],
        method="bilinear",
    )

    return saliency_tensor[:, :, 0].numpy().astype(np.float32)


def _close_memmap(
    array: np.memmap,
) -> None:
    """Flush and close an NPY memory map so it can be atomically moved."""
    memory_map = getattr(
        array,
        "_mmap",
        None,
    )
    if memory_map is None or not memory_map.closed:
        array.flush()
    if memory_map is not None and not memory_map.closed:
        memory_map.close()


def build_train_saliency_maps(
    dataset_name: str,
    data_dir: str,
    saliency_dir: str,
    method: str = "opencv",
    overwrite: bool = False,
) -> Path:
    """Build and save saliency maps for every training example."""
    from tqdm import tqdm

    dataset_name = canonical_dataset_name(
        dataset_name,
    )

    saliency_path = get_train_saliency_path(
        dataset_name=dataset_name,
        saliency_dir=saliency_dir,
    )

    if saliency_path.exists() and not overwrite:
        logger.info("Saliency maps already exist: %s", saliency_path)
        logger.info("Use --overwrite true to regenerate them.")

        return saliency_path

    saliency_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ds = _load_train_dataset_for_saliency_cache(
        dataset_name=dataset_name,
        data_dir=data_dir,
    )
    metadata = get_metadata(
        dataset_name,
    )

    expected_examples = resolve_source_train_example_count(
        dataset_name=dataset_name,
        data_dir=data_dir,
        metadata=metadata,
    )
    temporary_path = saliency_path.with_name(
        f".{saliency_path.name}.tmp",
    )
    temporary_path.unlink(
        missing_ok=True,
    )

    saliency_maps = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            expected_examples,
            metadata.image_size,
            metadata.image_size,
        ),
    )
    processed_examples = 0
    value_sum = 0
    squared_value_sum = 0
    value_min = 255
    value_max = 0

    try:
        for example in tqdm(
            tfds.as_numpy(train_ds),
            total=expected_examples,
            desc=f"Building {dataset_name} {method} saliency maps",
        ):
            if processed_examples >= expected_examples:
                raise ValueError(
                    "Training dataset contains more examples than expected: "
                    f"expected {expected_examples}."
                )

            image = example["image"]
            saliency_map = compute_saliency_map(
                image=image,
                method=method,
            )
            saliency_map = _resize_saliency_map_to_dataset_size(
                saliency_map=saliency_map,
                image_size=metadata.image_size,
            )
            quantized_map = np.rint(
                np.clip(
                    saliency_map,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(np.uint8)
            saliency_maps[processed_examples] = quantized_map

            unsigned_map = quantized_map.astype(
                np.uint64,
            )
            value_sum += int(
                unsigned_map.sum(),
            )
            squared_value_sum += int(
                (unsigned_map * unsigned_map).sum(),
            )
            value_min = min(
                value_min,
                int(quantized_map.min()),
            )
            value_max = max(
                value_max,
                int(quantized_map.max()),
            )
            processed_examples += 1

        if processed_examples != expected_examples:
            raise ValueError(
                "Training dataset cardinality does not match the expected "
                f"saliency cache size: got {processed_examples}, expected "
                f"{expected_examples}."
            )

        _close_memmap(
            saliency_maps,
        )
        os.replace(
            temporary_path,
            saliency_path,
        )
    except Exception:
        _close_memmap(
            saliency_maps,
        )
        temporary_path.unlink(
            missing_ok=True,
        )
        raise

    pixel_count = (
        processed_examples
        * metadata.image_size
        * metadata.image_size
    )
    mean_value = value_sum / pixel_count / 255.0
    second_moment = squared_value_sum / pixel_count / (255.0**2)
    standard_deviation = np.sqrt(
        max(
            second_moment - mean_value**2,
            0.0,
        )
    )

    logger.info("Saved saliency maps to: %s", saliency_path)
    logger.info("Method: %s", method)
    logger.info(
        "Shape: (%d, %d, %d)",
        processed_examples,
        metadata.image_size,
        metadata.image_size,
    )
    logger.info("Storage dtype: uint8 (decoded to [0, 1] while training)")
    logger.info("Min: %.6f", value_min / 255.0)
    logger.info("Max: %.6f", value_max / 255.0)
    logger.info("Mean: %.6f", mean_value)
    logger.info("Std: %.6f", standard_deviation)

    return saliency_path


def load_train_saliency_maps(
    dataset_name: str,
    saliency_dir: str,
) -> np.ndarray:
    """Load precomputed training saliency maps."""
    dataset_name = canonical_dataset_name(
        dataset_name,
    )

    saliency_path = get_train_saliency_path(
        dataset_name=dataset_name,
        saliency_dir=saliency_dir,
    )

    if not saliency_path.exists():
        raise FileNotFoundError(
            f"Saliency map file not found: {saliency_path}\n"
            "Generate it first with:\n"
            f"python -m allthemix.data.saliency "
            f"--dataset {dataset_name.lower()} "
            f"--data_dir ./data "
            f"--output_dir {saliency_dir}"
        )

    saliency_maps = np.load(
        saliency_path,
        mmap_mode="r",
    )

    return saliency_maps


__all__ = [
    "build_train_saliency_maps",
    "canonical_dataset_name",
    "get_train_saliency_path",
    "load_train_saliency_maps",
]
