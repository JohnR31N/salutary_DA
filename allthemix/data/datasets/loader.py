# allthemix/data/datasets/loader.py

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from allthemix.data.datasets.cars196 import (
    CARS196_DIR_CANDIDATES,
    is_cars196_name,
)
from allthemix.data.datasets.imagenet100 import (
    CMC_IMAGENET100_CLASS_NAMES,
    CMC_IMAGENET100_TRAIN_EXAMPLES,
    CMC_IMAGENET100_VAL_EXAMPLES,
    IMAGENET100_DIR_CANDIDATES,
    is_imagenet100_name,
)
from allthemix.data.datasets.tiny_imagenet import (
    TINY_IMAGENET_DIR_NAME,
    TINY_IMAGENET_URL,
    TINY_IMAGENET_ZIP_NAME,
    is_tiny_imagenet_name,
)

logger = logging.getLogger(__name__)

IMAGE_FILE_PATTERNS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
)


def _download_file(
    url: str,
    output_path: Path,
) -> None:
    """Support download file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Downloading %s", url)
    logger.info("Saving to %s", output_path)

    urllib.request.urlretrieve(
        url,
        output_path,
    )


def _extract_zip(
    zip_path: Path,
    output_dir: Path,
) -> None:
    """Support extract zip."""
    logger.info("Extracting %s to %s", zip_path, output_dir)

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zip_file:
        zip_file.extractall(
            output_dir,
        )


def _ensure_tiny_imagenet_downloaded(
    data_dir: str,
) -> Path:
    """Support ensure tiny imagenet downloaded."""
    data_root = Path(data_dir)
    tiny_root = data_root / TINY_IMAGENET_DIR_NAME
    zip_path = data_root / TINY_IMAGENET_ZIP_NAME

    if tiny_root.exists():
        return tiny_root

    data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not zip_path.exists():
        _download_file(
            url=TINY_IMAGENET_URL,
            output_path=zip_path,
        )

    _extract_zip(
        zip_path=zip_path,
        output_dir=data_root,
    )

    if not tiny_root.exists():
        raise FileNotFoundError(
            "Tiny ImageNet extraction failed. Expected directory:\n"
            f"  {tiny_root}"
        )

    return tiny_root


def _get_tiny_imagenet_root(
    data_dir: str,
) -> Path:
    """Support get tiny imagenet root."""
    root = _ensure_tiny_imagenet_downloaded(
        data_dir=data_dir,
    )

    required_paths = [
        root / "train",
        root / "val",
        root / "wnids.txt",
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                "Tiny ImageNet directory exists but is incomplete. "
                f"Missing: {required_path}"
            )

    return root


def _read_tiny_imagenet_wnids(
    root: Path,
) -> list[str]:
    """Support read tiny imagenet wnids."""
    wnids_path = root / "wnids.txt"

    if not wnids_path.exists():
        raise FileNotFoundError(
            f"Missing Tiny ImageNet wnids file: {wnids_path}"
        )

    with open(
        wnids_path,
        "r",
        encoding="utf-8",
    ) as f:
        wnids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(wnids) != 200:
        raise ValueError(
            "Tiny ImageNet should contain 200 class ids in wnids.txt, "
            f"but found {len(wnids)}."
        )

    return wnids


def _decode_image_path_example(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Support decode image path example."""
    image_bytes = tf.io.read_file(
        image_path,
    )

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    label = tf.cast(
        label,
        tf.int64,
    )

    return {
        "image": image,
        "label": label,
    }


def _build_image_dataset_from_paths(
    image_paths: list[str],
    labels: list[int],
    dataset_name: str,
) -> tf.data.Dataset:
    """Build an image dataset from path and label lists."""
    if len(image_paths) == 0:
        raise ValueError(
            f"No {dataset_name} images were found."
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
        )
    )

    dataset = dataset.map(
        _decode_image_path_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def _build_tiny_imagenet_dataset_from_paths(
    image_paths: list[str],
    labels: list[int],
) -> tf.data.Dataset:
    """Support build tiny imagenet dataset from paths."""
    return _build_image_dataset_from_paths(
        image_paths=image_paths,
        labels=labels,
        dataset_name="Tiny ImageNet",
    )


def _load_tiny_imagenet_train_dataset(
    data_dir: str,
) -> tf.data.Dataset:
    """Support load tiny imagenet train dataset."""
    root = _get_tiny_imagenet_root(
        data_dir=data_dir,
    )

    wnids = _read_tiny_imagenet_wnids(
        root=root,
    )

    wnid_to_label = {
        wnid: idx
        for idx, wnid in enumerate(
            wnids,
        )
    }

    image_paths: list[str] = []
    labels: list[int] = []

    for wnid in wnids:
        image_dir = root / "train" / wnid / "images"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing Tiny ImageNet train image dir: {image_dir}"
            )

        for image_path in sorted(
            image_dir.glob("*.JPEG"),
        ):
            image_paths.append(
                str(image_path),
            )

            labels.append(
                wnid_to_label[wnid],
            )

    return _build_tiny_imagenet_dataset_from_paths(
        image_paths=image_paths,
        labels=labels,
    )


def _load_tiny_imagenet_validation_dataset(
    data_dir: str,
) -> tf.data.Dataset:
    """Support load tiny imagenet validation dataset."""
    root = _get_tiny_imagenet_root(
        data_dir=data_dir,
    )

    wnids = _read_tiny_imagenet_wnids(
        root=root,
    )

    wnid_to_label = {
        wnid: idx
        for idx, wnid in enumerate(
            wnids,
        )
    }

    annotations_path = root / "val" / "val_annotations.txt"

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Missing Tiny ImageNet val annotations: {annotations_path}"
        )

    filename_to_wnid: dict[str, str] = {}

    with open(
        annotations_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            filename = parts[0]
            wnid = parts[1]

            filename_to_wnid[filename] = wnid

    image_paths: list[str] = []
    labels: list[int] = []

    image_dir = root / "val" / "images"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing Tiny ImageNet val image dir: {image_dir}"
        )

    for image_path in sorted(
        image_dir.glob("*.JPEG"),
    ):
        filename = image_path.name

        if filename not in filename_to_wnid:
            raise ValueError(
                f"Missing val annotation for image: {filename}"
            )

        wnid = filename_to_wnid[filename]

        if wnid not in wnid_to_label:
            raise ValueError(
                f"Unknown Tiny ImageNet wnid in val annotations: {wnid}"
            )

        image_paths.append(
            str(image_path),
        )

        labels.append(
            wnid_to_label[wnid],
        )

    return _build_tiny_imagenet_dataset_from_paths(
        image_paths=image_paths,
        labels=labels,
    )


def _find_class_folder_root(
    data_dir: str,
    dataset_display_name: str,
    directory_candidates: tuple[str, ...],
    train_split_name: str = "train",
    test_split_candidates: tuple[str, ...] = (
        "test",
        "val",
    ),
) -> Path:
    """Find a local class-folder dataset root."""
    data_root = Path(data_dir)

    for candidate in directory_candidates:
        root = data_root / candidate

        if not (root / train_split_name).is_dir():
            continue

        if any(
            (root / split_name).is_dir()
            for split_name in test_split_candidates
        ):
            return root

    raise FileNotFoundError(
        f"{dataset_display_name} local data was not found.\n"
        "This project expects a local class-folder copy.\n"
        "Supported layout:\n"
        f"  data/<dataset_root>/{train_split_name}/<class_name>/*.jpg\n"
        "  data/<dataset_root>/<test_or_val>/<class_name>/*.jpg\n"
        f"Dataset root candidates: {directory_candidates}\n"
        f"Looked under: {data_root}"
    )


def _list_images_in_class_dir(
    class_dir: Path,
) -> list[Path]:
    """List image files in a class directory with stable ordering."""
    image_paths: list[Path] = []

    for pattern in IMAGE_FILE_PATTERNS:
        image_paths.extend(
            class_dir.glob(
                pattern,
            )
        )

    return sorted(
        set(
            image_paths,
        )
    )


def _resolve_class_folder_split_root(
    root: Path,
    split: str,
) -> Path:
    """Resolve requested split name against common class-folder layouts."""
    split_candidates = (
        (split,)
        if split != "test"
        else (
            "test",
            "val",
        )
    )

    for split_name in split_candidates:
        split_root = root / split_name

        if split_root.is_dir():
            return split_root

    raise FileNotFoundError(
        f"Missing class-folder split under {root}: {split_candidates}"
    )


def _list_class_folder_names(
    split_root: Path,
) -> tuple[str, ...]:
    """List class-directory names in deterministic label order."""
    return tuple(
        sorted(
            class_dir.name
            for class_dir in split_root.iterdir()
            if class_dir.is_dir()
        )
    )


def _index_class_folder_split_dataset(
    data_dir: str,
    split: str,
    dataset_display_name: str,
    directory_candidates: tuple[str, ...],
    expected_num_classes: int,
    expected_class_names: tuple[str, ...] | None = None,
    expected_split_examples: int | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Index one local class-folder split and validate its structure."""
    root = _find_class_folder_root(
        data_dir=data_dir,
        dataset_display_name=dataset_display_name,
        directory_candidates=directory_candidates,
    )

    train_root = root / "train"
    split_root = _resolve_class_folder_split_root(
        root=root,
        split=split,
    )

    train_class_names = _list_class_folder_names(
        split_root=train_root,
    )

    if expected_class_names is not None:
        expected_class_names = tuple(
            expected_class_names,
        )
        if train_class_names != expected_class_names:
            missing = sorted(
                set(expected_class_names) - set(train_class_names),
            )
            unexpected = sorted(
                set(train_class_names) - set(expected_class_names),
            )
            raise ValueError(
                f"{dataset_display_name} train class folders do not match "
                "the required manifest. "
                f"Missing ({len(missing)}): {missing}; "
                f"unexpected ({len(unexpected)}): {unexpected}."
            )
        class_names = expected_class_names
    else:
        class_names = train_class_names

    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"{dataset_display_name} should contain "
            f"{expected_num_classes} train class folders, "
            f"but found {len(class_names)} in {train_root}."
        )

    split_class_names = _list_class_folder_names(
        split_root=split_root,
    )
    if split_class_names != class_names:
        missing = sorted(
            set(class_names) - set(split_class_names),
        )
        unexpected = sorted(
            set(split_class_names) - set(class_names),
        )
        raise ValueError(
            f"{dataset_display_name} {split} class folders do not match "
            "the train label space. "
            f"Missing ({len(missing)}): {missing}; "
            f"unexpected ({len(unexpected)}): {unexpected}."
        )

    class_to_label = {
        class_name: index
        for index, class_name in enumerate(
            class_names,
        )
    }

    image_paths: list[str] = []
    labels: list[int] = []
    class_counts: list[int] = []

    for class_name in class_names:
        class_dir = split_root / class_name

        class_image_paths = _list_images_in_class_dir(
            class_dir=class_dir,
        )
        if not class_image_paths and expected_class_names is not None:
            raise ValueError(
                f"{dataset_display_name} {split} class has no images: "
                f"{class_dir}"
            )

        class_counts.append(
            len(class_image_paths),
        )
        for image_path in class_image_paths:
            image_paths.append(
                str(image_path),
            )
            labels.append(
                class_to_label[class_name],
            )

    if (
        expected_split_examples is not None
        and len(image_paths) != expected_split_examples
    ):
        raise ValueError(
            f"{dataset_display_name} {split} should contain "
            f"{expected_split_examples} images, but found "
            f"{len(image_paths)} in {split_root}."
        )

    return (
        tuple(image_paths),
        tuple(labels),
        tuple(class_counts),
    )


def _load_class_folder_split_dataset(
    data_dir: str,
    split: str,
    dataset_name: str,
    dataset_display_name: str,
    directory_candidates: tuple[str, ...],
    expected_num_classes: int,
    expected_class_names: tuple[str, ...] | None = None,
    expected_split_examples: int | None = None,
) -> tf.data.Dataset:
    """Load a validated local class-folder dataset split."""
    image_paths, labels, _class_counts = _index_class_folder_split_dataset(
        data_dir=data_dir,
        split=split,
        dataset_display_name=dataset_display_name,
        directory_candidates=directory_candidates,
        expected_num_classes=expected_num_classes,
        expected_class_names=expected_class_names,
        expected_split_examples=expected_split_examples,
    )

    return _build_image_dataset_from_paths(
        image_paths=list(image_paths),
        labels=list(labels),
        dataset_name=f"{dataset_name} {split}",
    )


def _load_cars196_split_dataset(
    data_dir: str,
    split: str,
) -> tf.data.Dataset:
    """Load Cars196 from local class-folder train/test directories."""
    return _load_class_folder_split_dataset(
        data_dir=data_dir,
        split=split,
        dataset_name="Cars196",
        dataset_display_name="Cars196",
        directory_candidates=CARS196_DIR_CANDIDATES,
        expected_num_classes=196,
    )


def _load_cars196_train_dataset(
    data_dir: str,
) -> tf.data.Dataset:
    """Load the Cars196 train split from local class folders."""
    return _load_cars196_split_dataset(
        data_dir=data_dir,
        split="train",
    )


def _load_cars196_test_dataset(
    data_dir: str,
) -> tf.data.Dataset:
    """Load the Cars196 test split from local class folders."""
    return _load_cars196_split_dataset(
        data_dir=data_dir,
        split="test",
    )


def _load_imagenet100_split_dataset(
    data_dir: str,
    split: str,
) -> tf.data.Dataset:
    """Load the canonical CMC ImageNet-100 train or validation split."""
    expected_split_examples = (
        CMC_IMAGENET100_TRAIN_EXAMPLES
        if split == "train"
        else CMC_IMAGENET100_VAL_EXAMPLES
    )

    return _load_class_folder_split_dataset(
        data_dir=data_dir,
        split=split,
        dataset_name="CMC ImageNet-100",
        dataset_display_name="CMC ImageNet-100",
        directory_candidates=IMAGENET100_DIR_CANDIDATES,
        expected_num_classes=len(CMC_IMAGENET100_CLASS_NAMES),
        expected_class_names=CMC_IMAGENET100_CLASS_NAMES,
        expected_split_examples=expected_split_examples,
    )


def get_imagenet100_split_class_counts(
    data_dir: str,
    split: str,
) -> tuple[int, ...]:
    """Return exact per-class counts after validating the CMC local split."""
    if split not in {
        "train",
        "val",
    }:
        raise ValueError(
            "CMC ImageNet-100 split must be 'train' or 'val'. "
            f"Got {split!r}."
        )

    expected_split_examples = (
        CMC_IMAGENET100_TRAIN_EXAMPLES
        if split == "train"
        else CMC_IMAGENET100_VAL_EXAMPLES
    )
    _image_paths, _labels, class_counts = (
        _index_class_folder_split_dataset(
            data_dir=data_dir,
            split=split,
            dataset_display_name="CMC ImageNet-100",
            directory_candidates=IMAGENET100_DIR_CANDIDATES,
            expected_num_classes=len(CMC_IMAGENET100_CLASS_NAMES),
            expected_class_names=CMC_IMAGENET100_CLASS_NAMES,
            expected_split_examples=expected_split_examples,
        )
    )

    return class_counts


def get_runtime_train_class_counts(
    name: str,
    data_dir: str,
) -> tuple[int, ...] | None:
    """Return cheap filesystem class counts for supported local datasets."""
    if is_imagenet100_name(
        name,
    ):
        return get_imagenet100_split_class_counts(
            data_dir=data_dir,
            split="train",
        )

    return None


def _load_imagenet100_train_dataset(
    data_dir: str,
) -> tf.data.Dataset:
    """Load the ImageNet100 train split from local class folders."""
    return _load_imagenet100_split_dataset(
        data_dir=data_dir,
        split="train",
    )


def _load_imagenet100_test_dataset(
    data_dir: str,
) -> tf.data.Dataset:
    """Load the ImageNet100 validation split from local class folders."""
    return _load_imagenet100_split_dataset(
        data_dir=data_dir,
        split="val",
    )


def load_tfds_dataset(
    name: str,
    split: str,
    data_dir: str,
    shuffle_files: bool,
    seed: int | None = None,
    download: bool = True,
) -> tf.data.Dataset:
    """Load tfds dataset."""
    read_config = None
    if shuffle_files and seed is not None:
        read_config = tfds.ReadConfig(
            shuffle_seed=seed,
            shuffle_reshuffle_each_iteration=True,
        )

    dataset = tfds.load(
        name=name,
        split=split,
        data_dir=data_dir,
        shuffle_files=shuffle_files,
        read_config=read_config,
        download=download,
    )

    return dataset


def load_train_dataset(
    name: str,
    data_dir: str,
    shuffle_files: bool = True,
    seed: int | None = None,
    download: bool = True,
) -> tf.data.Dataset:
    """Load train dataset."""
    if is_tiny_imagenet_name(
        name,
    ):
        return _load_tiny_imagenet_train_dataset(
            data_dir=data_dir,
        )

    if is_cars196_name(
        name,
    ):
        return _load_cars196_train_dataset(
            data_dir=data_dir,
        )

    if is_imagenet100_name(
        name,
    ):
        return _load_imagenet100_train_dataset(
            data_dir=data_dir,
        )

    return load_tfds_dataset(
        name=name,
        split="train",
        data_dir=data_dir,
        shuffle_files=shuffle_files,
        seed=seed,
        download=download,
    )


def load_test_dataset(
    name: str,
    data_dir: str,
) -> tf.data.Dataset:
    """Load test dataset."""
    if is_tiny_imagenet_name(
        name,
    ):
        return _load_tiny_imagenet_validation_dataset(
            data_dir=data_dir,
        )

    if is_cars196_name(
        name,
    ):
        return _load_cars196_test_dataset(
            data_dir=data_dir,
        )

    if is_imagenet100_name(
        name,
    ):
        return _load_imagenet100_test_dataset(
            data_dir=data_dir,
        )

    return load_tfds_dataset(
        name=name,
        split="test",
        data_dir=data_dir,
        shuffle_files=False,
    )


def download_dataset(
    name: str,
    data_dir: str,
) -> None:
    """Download dataset."""
    load_train_dataset(
        name=name,
        data_dir=data_dir,
    )

    load_test_dataset(
        name=name,
        data_dir=data_dir,
    )
