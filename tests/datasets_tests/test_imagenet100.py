from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from allthemix.config import load_yaml_config
from allthemix.data.utils.cardinality import resolve_train_example_count
from allthemix.data.preprocessors.tfds_image import get_metadata
from allthemix.data.datasets.imagenet100 import (
    CMC_IMAGENET100_CLASS_NAMES,
    CMC_IMAGENET100_NUM_CLASSES,
    CMC_IMAGENET100_SYNSETS,
    CMC_IMAGENET100_TRAIN_EXAMPLES,
    CMC_IMAGENET100_VAL_EXAMPLES,
    get_cmc_imagenet100_label_map,
    validate_cmc_imagenet100_class_names,
)


def test_cmc_manifest_and_label_order_are_fixed() -> None:
    """The bundled CMC split must remain complete and ImageFolder-compatible."""
    assert len(CMC_IMAGENET100_SYNSETS) == CMC_IMAGENET100_NUM_CLASSES
    assert len(set(CMC_IMAGENET100_SYNSETS)) == CMC_IMAGENET100_NUM_CLASSES
    assert CMC_IMAGENET100_CLASS_NAMES == tuple(
        sorted(
            CMC_IMAGENET100_SYNSETS,
        )
    )
    assert get_cmc_imagenet100_label_map() == {
        synset: label
        for label, synset in enumerate(
            CMC_IMAGENET100_CLASS_NAMES,
        )
    }


def test_cmc_manifest_rejects_an_arbitrary_hundred_classes() -> None:
    """Any 100-folder subset is not equivalent to the published CMC split."""
    with pytest.raises(
        ValueError,
        match="official CMC split",
    ):
        validate_cmc_imagenet100_class_names(
            class_names=(
                f"class_{index:03d}"
                for index in range(100)
            ),
            split_name="train",
        )


def test_imagenet100_metadata_uses_exact_cmc_cardinality() -> None:
    """Training schedules must use exact CMC train and validation sizes."""
    metadata = get_metadata(
        "imagenet100",
    )
    assert metadata.num_classes == CMC_IMAGENET100_NUM_CLASSES
    assert metadata.num_train_examples == CMC_IMAGENET100_TRAIN_EXAMPLES
    assert metadata.num_test_examples == CMC_IMAGENET100_VAL_EXAMPLES


def test_runtime_counts_drive_exact_stratified_epoch_size(
    monkeypatch,
) -> None:
    """Local class counts should determine steps after a stratified split."""
    import allthemix.data.utils.cardinality as cardinality_module

    monkeypatch.setattr(
        cardinality_module,
        "get_runtime_train_class_counts",
        lambda **_kwargs: (
            3,
            4,
        ),
    )
    assert resolve_train_example_count(
        dataset_name="imagenet100",
        data_dir="unused",
        metadata=get_metadata(
            "imagenet100",
        ),
        validation_split=0.1,
    ) == 5


def test_prepare_cmc_dataset_supports_a_custom_output_root(
    monkeypatch,
    tmp_path,
) -> None:
    """Preparation should validate content, not require a magic folder name."""
    import allthemix.cli.prepare_imagenet100 as prepare_module

    source_dir = tmp_path / "licensed_imagenet1k"
    for split in (
        "train",
        "val",
    ):
        for class_index, synset in enumerate(CMC_IMAGENET100_CLASS_NAMES):
            class_dir = source_dir / split / synset
            class_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            (class_dir / f"{class_index:03d}.JPEG").write_bytes(
                b"test-image",
            )

    monkeypatch.setattr(
        prepare_module,
        "CMC_IMAGENET100_TRAIN_EXAMPLES",
        100,
    )
    monkeypatch.setattr(
        prepare_module,
        "CMC_IMAGENET100_VAL_EXAMPLES",
        100,
    )
    output_dir = tmp_path / "custom-cmc-root"
    summary = prepare_module.prepare_cmc_imagenet100_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        link_mode="copy",
    )

    assert summary["num_classes"] == 100
    assert summary["num_train_examples"] == 100
    assert summary["num_val_examples"] == 100
    assert (
        output_dir / prepare_module.MANIFEST_FILE_NAME
    ).read_text(
        encoding="ascii",
    ).splitlines() == list(CMC_IMAGENET100_SYNSETS)

    with (
        output_dir / prepare_module.LABEL_MAP_FILE_NAME
    ).open(
        newline="",
        encoding="ascii",
    ) as file:
        label_rows = list(
            csv.DictReader(
                file,
            )
        )
    assert label_rows[0] == {
        "synset": CMC_IMAGENET100_CLASS_NAMES[0],
        "label": "0",
    }

    info = json.loads(
        (output_dir / prepare_module.DATASET_INFO_FILE_NAME).read_text(
            encoding="utf-8",
        )
    )
    assert info["label_order"] == "sorted_synset_id"
    assert info["num_classes"] == 100


def test_imagenet100_method_configs_share_the_training_protocol() -> None:
    """Method comparisons must not silently change the base optimizer setup."""
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "imagenet100"
        / "preact_resnet18"
    )
    all_config_paths = sorted(
        config_dir.glob(
            "*.yaml",
        )
    )
    assert {
        config_path.stem
        for config_path in all_config_paths
    } == {
        "baseline",
        "baseline_syncdist",
        "catchupmix",
        "cutmix",
        "cutmix_sumix",
        "diffusemix",
        "fmix",
        "guided_sr",
        "mixup",
        "resize",
        "saliencymix",
    }
    # baseline is the single-device ERM variant and baseline_syncdist is its
    # distributed counterpart; every METHOD config must resolve to the
    # shared syncdist e400 protocol below.
    config_paths = [
        config_path
        for config_path in all_config_paths
        if config_path.stem not in {"baseline", "baseline_syncdist"}
    ]

    configs = []
    for config_path in config_paths:
        configs.append(
            load_yaml_config(
                config_path,
            )
        )

    expected_protocol = {
        "dataset": "imagenet100",
        "data_dir": "./data",
        "model": "preact_resnet18",
        "resnet_stem_type": "imagenet",
        "batch_size": 128,
        "epochs": 400,
        "validation_split": 0.1,
        "eval_on_test_each_epoch": False,
        "final_test": True,
        "final_test_checkpoint": "best",
        "learning_rate": 0.05,
        "momentum": 0.9,
        "nesterov": False,
        "weight_decay": 0.0001,
        "lr_schedule": "cosine",
        "distributed": True,
        "cross_device_shuffle": False,
        "sync_batch_stats": True,
    }
    for config in configs:
        assert {
            key: config[key]
            for key in expected_protocol
        } == expected_protocol
