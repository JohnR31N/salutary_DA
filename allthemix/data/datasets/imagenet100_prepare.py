"""CMC ImageNet-100 preparation and verification logic."""

from __future__ import annotations

import csv
import filecmp
import json
import os
import shutil
from pathlib import Path

from allthemix.data.datasets.imagenet100 import (
    CMC_IMAGENET100_CLASS_NAMES,
    CMC_IMAGENET100_SOURCE_URL,
    CMC_IMAGENET100_SYNSETS,
    CMC_IMAGENET100_TRAIN_EXAMPLES,
    CMC_IMAGENET100_VAL_EXAMPLES,
    get_cmc_imagenet100_label_map,
    validate_cmc_imagenet100_class_names,
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}
MANIFEST_FILE_NAME = "cmc_imagenet100_synsets.txt"
LABEL_MAP_FILE_NAME = "cmc_imagenet100_labels.csv"
DATASET_INFO_FILE_NAME = "cmc_imagenet100_info.json"


def _list_image_files(
    class_dir: Path,
) -> tuple[Path, ...]:
    """List supported image files in one flat ImageNet class directory."""
    return tuple(
        sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )


def _validate_source_split(
    source_dir: Path,
    split: str,
    expected_examples: int,
) -> tuple[tuple[Path, ...], ...]:
    """Validate required CMC classes inside an ImageNet-1K source split."""
    split_root = source_dir / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Missing ImageNet-1K source split directory: {split_root}"
        )

    class_images = []
    missing_classes = []
    for synset in CMC_IMAGENET100_CLASS_NAMES:
        class_dir = split_root / synset
        if not class_dir.is_dir():
            missing_classes.append(
                synset,
            )
            continue

        image_paths = _list_image_files(
            class_dir=class_dir,
        )
        if not image_paths:
            raise ValueError(
                f"ImageNet-1K source class has no images: {class_dir}"
            )
        class_images.append(
            image_paths,
        )

    if missing_classes:
        raise FileNotFoundError(
            f"ImageNet-1K source {split} is missing "
            f"{len(missing_classes)} CMC classes: {missing_classes}"
        )

    total_examples = sum(
        len(image_paths)
        for image_paths in class_images
    )
    if total_examples != expected_examples:
        raise ValueError(
            f"CMC source {split} should contain {expected_examples} images "
            f"across the selected classes, but found {total_examples}."
        )

    return tuple(
        class_images,
    )


def _validate_prepared_split(
    data_dir: Path,
    split: str,
    expected_examples: int,
) -> tuple[int, ...]:
    """Validate one prepared split without assuming a canonical root name."""
    split_root = data_dir / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Missing prepared CMC split directory: {split_root}"
        )

    class_names = tuple(
        path.name
        for path in split_root.iterdir()
        if path.is_dir()
    )
    validate_cmc_imagenet100_class_names(
        class_names=class_names,
        split_name=split,
    )

    class_counts = []
    for synset in CMC_IMAGENET100_CLASS_NAMES:
        class_dir = split_root / synset
        image_count = len(
            _list_image_files(
                class_dir=class_dir,
            )
        )
        if image_count == 0:
            raise ValueError(
                f"Prepared CMC {split} class has no images: {class_dir}"
            )
        class_counts.append(
            image_count,
        )

    total_examples = sum(
        class_counts,
    )
    if total_examples != expected_examples:
        raise ValueError(
            f"Prepared CMC {split} should contain {expected_examples} "
            f"images, but found {total_examples} in {split_root}."
        )

    return tuple(
        class_counts,
    )


def _ensure_matching_file(
    source_path: Path,
    target_path: Path,
    link_mode: str,
) -> None:
    """Create one idempotent hardlink or copy without overwriting data."""
    if target_path.exists():
        if link_mode == "hardlink" and os.path.samefile(
            source_path,
            target_path,
        ):
            return
        if link_mode == "copy" and filecmp.cmp(
            source_path,
            target_path,
            shallow=False,
        ):
            return
        raise FileExistsError(
            f"Refusing to overwrite an existing dataset file: {target_path}"
        )

    if link_mode == "hardlink":
        os.link(
            source_path,
            target_path,
        )
    else:
        shutil.copy2(
            source_path,
            target_path,
        )


def _prepare_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    class_images: tuple[tuple[Path, ...], ...],
    link_mode: str,
) -> None:
    """Materialize one selected split with directory links or image files."""
    source_split_root = source_dir / split
    output_split_root = output_dir / split
    output_split_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_class_names = tuple(
        path.name
        for path in output_split_root.iterdir()
        if path.is_dir()
    )
    unexpected_classes = sorted(
        set(existing_class_names) - set(CMC_IMAGENET100_CLASS_NAMES),
    )
    if unexpected_classes:
        raise ValueError(
            f"Output {split} contains non-CMC class folders: "
            f"{unexpected_classes}"
        )

    for synset, image_paths in zip(
        CMC_IMAGENET100_CLASS_NAMES,
        class_images,
        strict=True,
    ):
        source_class_dir = source_split_root / synset
        output_class_dir = output_split_root / synset

        if link_mode == "symlink":
            if output_class_dir.is_symlink():
                if output_class_dir.resolve() == source_class_dir.resolve():
                    continue
                raise FileExistsError(
                    "Existing class symlink points to a different source: "
                    f"{output_class_dir}"
                )
            if output_class_dir.exists():
                raise FileExistsError(
                    "Cannot replace an existing class directory with a "
                    f"symlink: {output_class_dir}"
                )

            relative_source = os.path.relpath(
                source_class_dir,
                start=output_class_dir.parent,
            )
            try:
                output_class_dir.symlink_to(
                    relative_source,
                    target_is_directory=True,
                )
            except OSError as error:
                raise OSError(
                    "Could not create a directory symlink. On Windows, enable "
                    "Developer Mode or rerun with --link_mode hardlink/copy."
                ) from error
            continue

        output_class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for source_path in image_paths:
            _ensure_matching_file(
                source_path=source_path,
                target_path=output_class_dir / source_path.name,
                link_mode=link_mode,
            )


def _write_protocol_files(
    output_dir: Path,
    source_dir: Path,
    link_mode: str,
) -> None:
    """Record the exact CMC class and label protocol beside the dataset."""
    manifest_text = "\n".join(
        CMC_IMAGENET100_SYNSETS,
    ) + "\n"
    manifest_path = output_dir / MANIFEST_FILE_NAME
    if manifest_path.exists() and manifest_path.read_text(
        encoding="ascii",
    ) != manifest_text:
        raise FileExistsError(
            f"Existing CMC manifest has different contents: {manifest_path}"
        )
    manifest_path.write_text(
        manifest_text,
        encoding="ascii",
    )

    label_map_path = output_dir / LABEL_MAP_FILE_NAME
    label_rows = [
        (
            synset,
            label,
        )
        for synset, label in get_cmc_imagenet100_label_map().items()
    ]
    with label_map_path.open(
        "w",
        encoding="ascii",
        newline="",
    ) as file:
        writer = csv.writer(
            file,
        )
        writer.writerow(
            (
                "synset",
                "label",
            )
        )
        writer.writerows(
            label_rows,
        )

    info = {
        "dataset": "ImageNet-100 (CMC split)",
        "manifest_source": CMC_IMAGENET100_SOURCE_URL,
        "label_order": "sorted_synset_id",
        "num_classes": len(CMC_IMAGENET100_CLASS_NAMES),
        "num_train_examples": CMC_IMAGENET100_TRAIN_EXAMPLES,
        "num_val_examples": CMC_IMAGENET100_VAL_EXAMPLES,
        "source_dir": str(source_dir.resolve()),
        "link_mode": link_mode,
    }
    (output_dir / DATASET_INFO_FILE_NAME).write_text(
        json.dumps(
            info,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def verify_cmc_imagenet100_dataset(
    data_dir: str | Path,
) -> dict[str, object]:
    """Validate a prepared CMC dataset and return reproducibility metadata."""
    data_dir = Path(
        data_dir,
    )
    train_counts = _validate_prepared_split(
        data_dir=data_dir,
        split="train",
        expected_examples=CMC_IMAGENET100_TRAIN_EXAMPLES,
    )
    val_counts = _validate_prepared_split(
        data_dir=data_dir,
        split="val",
        expected_examples=CMC_IMAGENET100_VAL_EXAMPLES,
    )

    return {
        "dataset_root": str(data_dir.resolve()),
        "num_classes": len(train_counts),
        "num_train_examples": sum(train_counts),
        "num_val_examples": sum(val_counts),
        "train_class_count_min": min(train_counts),
        "train_class_count_max": max(train_counts),
        "val_examples_per_class": tuple(sorted(set(val_counts))),
    }


def prepare_cmc_imagenet100_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    link_mode: str = "symlink",
) -> dict[str, object]:
    """Create and verify CMC ImageNet-100 from class-folder ImageNet-1K."""
    if link_mode not in {
        "symlink",
        "hardlink",
        "copy",
    }:
        raise ValueError(
            f"Unsupported link_mode: {link_mode}"
        )

    source_dir = Path(
        source_dir,
    )
    output_dir = Path(
        output_dir,
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet-1K source directory does not exist: {source_dir}"
        )
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("source_dir and output_dir must be different.")

    train_images = _validate_source_split(
        source_dir=source_dir,
        split="train",
        expected_examples=CMC_IMAGENET100_TRAIN_EXAMPLES,
    )
    val_images = _validate_source_split(
        source_dir=source_dir,
        split="val",
        expected_examples=CMC_IMAGENET100_VAL_EXAMPLES,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    _prepare_split(
        source_dir=source_dir,
        output_dir=output_dir,
        split="train",
        class_images=train_images,
        link_mode=link_mode,
    )
    _prepare_split(
        source_dir=source_dir,
        output_dir=output_dir,
        split="val",
        class_images=val_images,
        link_mode=link_mode,
    )
    _write_protocol_files(
        output_dir=output_dir,
        source_dir=source_dir,
        link_mode=link_mode,
    )

    return verify_cmc_imagenet100_dataset(
        data_dir=output_dir,
    )


__all__ = [
    "DATASET_INFO_FILE_NAME",
    "LABEL_MAP_FILE_NAME",
    "MANIFEST_FILE_NAME",
    "prepare_cmc_imagenet100_dataset",
    "verify_cmc_imagenet100_dataset",
]
