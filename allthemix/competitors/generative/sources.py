"""Deterministic source-image adapters for offline generation methods."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from allthemix.data.datasets.cars196 import (
    CARS196_DIR_CANDIDATES,
    is_cars196_name,
)

_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}
_SPLIT_BUCKET_COUNT = 10_000
_SPLIT_MULTIPLIER = 1_103_515_245
_SPLIT_OFFSET = 12_345
_EXACT_SPLIT_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class SourceExample:
    """One raw source image with the label used by JAX training."""

    index: int
    source_id: str
    source_ref: str
    label: int
    class_name: str
    image: Image.Image


def validate_validation_split(validation_split: float) -> None:
    """Match the validation-split range accepted by AllTheMix."""
    if (
        not math.isfinite(validation_split)
        or validation_split < 0.0
        or validation_split >= 1.0
    ):
        raise ValueError(
            "validation_split must be in [0, 1). "
            f"Got {validation_split}."
        )


def is_validation_class_index(
    class_index: int,
    validation_split: float,
) -> bool:
    """Mirror AllTheMix's deterministic class-stratified split in Python."""
    validate_validation_split(validation_split)
    if validation_split == 0.0:
        return False

    period = int(round(1.0 / validation_split))
    is_reciprocal = (
        period > 1
        and abs(validation_split - 1.0 / period)
        <= _EXACT_SPLIT_TOLERANCE
    )
    if is_reciprocal:
        return class_index % period == 0

    threshold = int(round(validation_split * _SPLIT_BUCKET_COUNT))
    if threshold <= 0 or threshold >= _SPLIT_BUCKET_COUNT:
        raise ValueError(
            "validation_split is too close to 0 or 1 for deterministic "
            f"bucketing: {validation_split}."
        )
    bucket = (
        class_index * _SPLIT_MULTIPLIER + _SPLIT_OFFSET
    ) % _SPLIT_BUCKET_COUNT

    return bucket < threshold


def _class_directories(train_dir: Path) -> list[Path]:
    """List ImageFolder-style class directories in label order."""
    class_dirs = sorted(path for path in train_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class directories were found under: {train_dir}"
        )

    return class_dirs


def _class_images(class_dir: Path) -> list[Path]:
    """List supported image files recursively in stable path order."""
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def iter_class_folder_sources(
    train_dir: str | Path,
    validation_split: float,
) -> Iterator[SourceExample]:
    """Yield the training side of a class-folder dataset."""
    validate_validation_split(validation_split)
    root = Path(train_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Offline-generation train_dir does not exist: {root}"
        )

    global_index = 0
    for label, class_dir in enumerate(_class_directories(root)):
        image_paths = _class_images(class_dir)
        if not image_paths:
            raise ValueError(
                f"No images were found in class directory: {class_dir}"
            )

        for class_index, image_path in enumerate(image_paths):
            current_index = global_index
            global_index += 1
            if is_validation_class_index(
                class_index=class_index,
                validation_split=validation_split,
            ):
                continue

            relative_path = image_path.relative_to(root).as_posix()
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB").copy()

            yield SourceExample(
                index=current_index,
                source_id=f"class-folder:{relative_path}",
                source_ref=relative_path,
                label=label,
                class_name=class_dir.name,
                image=rgb_image,
            )


def _pil_from_raw_image(image: np.ndarray) -> Image.Image:
    """Convert one raw TF/TFDS image array to a detached RGB PIL image."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def _decode_class_name(value: object) -> str:
    """Decode optional dataset-provided class text into a prompt label."""
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")

    return str(value)


def _tfds_label_names(dataset: str, data_dir: str) -> tuple[str, ...]:
    """Read label names from TFDS metadata without changing source order."""
    try:
        import tensorflow_datasets as tfds

        builder = tfds.builder(dataset, data_dir=data_dir)
        feature = builder.info.features["label"]
        names = tuple(getattr(feature, "names", ()) or ())
    except (ImportError, KeyError, TypeError, ValueError):
        return ()

    return names


def clean_class_name(class_name: str) -> str:
    """Turn dataset folder labels into natural-language prompt text."""
    value = re.sub(r"^\d+[._ -]*", "", class_name.strip())
    value = value.replace("_", " ").replace(".", " ")
    value = re.sub(r"\s+", " ", value).strip()
    replacements = (
        ("Whip poor Will", "Eastern whip-poor-will"),
        ("Geococcyx", "Roadrunner"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)

    return value


def _local_class_folder_train_dir(
    dataset: str,
    data_dir: str,
) -> Path | None:
    """Resolve datasets whose local folder names carry prompt semantics."""
    if not is_cars196_name(dataset):
        return None

    data_root = Path(data_dir).expanduser()
    for candidate in CARS196_DIR_CANDIDATES:
        root = data_root / candidate
        train_root = root / "train"
        if train_root.is_dir() and any(
            (root / split).is_dir()
            for split in (
                "test",
                "val",
            )
        ):
            return train_root

    return None


def iter_allthemix_sources(
    dataset: str,
    data_dir: str,
    validation_split: float,
    download: bool = True,
) -> Iterator[SourceExample]:
    """Yield raw training examples through the existing AllTheMix loader."""
    validate_validation_split(validation_split)

    class_folder_train_dir = _local_class_folder_train_dir(
        dataset=dataset,
        data_dir=data_dir,
    )
    if class_folder_train_dir is not None:
        yield from iter_class_folder_sources(
            train_dir=class_folder_train_dir,
            validation_split=validation_split,
        )
        return

    # Delayed imports keep class-folder generation independent of TensorFlow.
    import tensorflow as tf

    from allthemix.data.datasets.loader import load_train_dataset

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError(
            "TensorFlow initialized a GPU before the source adapter could "
            "reserve it for PyTorch. Start generation in a fresh process."
        ) from error

    label_names = _tfds_label_names(dataset=dataset, data_dir=data_dir)
    raw_dataset = load_train_dataset(
        name=dataset,
        data_dir=data_dir,
        shuffle_files=False,
        download=download,
    )
    class_counts: dict[int, int] = {}

    for global_index, example in enumerate(raw_dataset.as_numpy_iterator()):
        if (
            not isinstance(example, dict)
            or "image" not in example
            or "label" not in example
        ):
            raise ValueError(
                "Offline dataset sources must yield dictionaries with "
                "'image' and 'label' fields."
            )

        label = int(example["label"])
        class_index = class_counts.get(label, 0)
        class_counts[label] = class_index + 1
        if is_validation_class_index(
            class_index=class_index,
            validation_split=validation_split,
        ):
            continue

        # CUB's TFDS builder exposes the human class name on each example,
        # while its label feature may contain only numeric strings. Prefer
        # the per-example value so diffusion prompts never become "class 0".
        raw_class_name = (
            _decode_class_name(example["label_name"])
            if "label_name" in example
            else (
                label_names[label]
                if 0 <= label < len(label_names)
                else str(label)
            )
        )
        yield SourceExample(
            index=global_index,
            source_id=f"{dataset}:{global_index:09d}",
            source_ref=f"{dataset}:train:{global_index}",
            label=label,
            class_name=raw_class_name,
            image=_pil_from_raw_image(example["image"]),
        )
