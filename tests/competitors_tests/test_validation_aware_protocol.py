from __future__ import annotations

import sys

import pytest

from allthemix.cli.args import parse_args

MAIN_PROTOCOL_FIELDS = (
    "dataset",
    "data_dir",
    "model",
    "resnet_stem_type",
    "preact_stem_bn_relu",
    "preact_pytorch_default_init",
    "batch_size",
    "shuffle_buffer_size",
    "epochs",
    "max_train_steps",
    "max_eval_steps",
    "early_stop_enabled",
    "early_stop_start_epoch",
    "early_stop_patience",
    "early_stop_min_delta",
    "validation_split",
    "val_source",
    "eval_on_test_each_epoch",
    "final_test",
    "final_test_checkpoint",
    "learning_rate",
    "momentum",
    "nesterov",
    "weight_decay",
    "lr_schedule",
    "min_learning_rate",
    "warmup_epochs",
    "lr_decay_epochs",
    "lr_decay_rate",
    "basic_aug",
    "aug_recipe",
    "tiny_imagenet_normalization",
    "seed",
    "data_seed",
    "deterministic_data",
    "strict_determinism",
)

VALIDATION_AWARE_DATASETS = (
    "cifar10",
    "cifar100",
    "stl10",
    "cars196",
    "cub200",
)

SYNC_DIST_BASELINE_DATASETS = (
    *VALIDATION_AWARE_DATASETS,
    "tiny_imagenet",
    "imagenet100",
)

MIXDA_METHODS = (
    "mixup",
    "cutmix",
    "resize",
    "fmix",
    "saliencymix",
    "guided_sr",
    "catchupmix",
)

# Saliency-paired methods intentionally express augmentation through
# sal_aug_recipe, so compare the common classifier/evaluation protocol here.
MIXDA_PROTOCOL_FIELDS = tuple(
    field
    for field in MAIN_PROTOCOL_FIELDS
    if field not in {"basic_aug", "aug_recipe"}
)


def _parse_config(
    monkeypatch,
    path: str,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            path,
        ],
    )
    return parse_args()


@pytest.mark.parametrize(
    "dataset",
    SYNC_DIST_BASELINE_DATASETS,
)
def test_main_protocol_sources_validation_from_official_eval(
    monkeypatch,
    dataset: str,
) -> None:
    """Train on full official train and seal most official eval for final."""
    args = _parse_config(
        monkeypatch,
        f"configs/{dataset}/preact_resnet18/baseline.yaml",
    )

    assert args.val_source == "test"
    expected_validation_split = 0.5 if dataset == "stl10" else 0.1
    assert args.validation_split == pytest.approx(expected_validation_split)
    assert args.eval_on_test_each_epoch is False
    assert args.final_test is True
    assert args.final_test_checkpoint == "best"


@pytest.mark.parametrize(
    "dataset",
    VALIDATION_AWARE_DATASETS,
)
def test_validation_aware_configs_match_main_table_protocol(
    monkeypatch,
    dataset: str,
) -> None:
    """Keep classifier and evaluation settings identical across method types."""
    config_root = f"configs/{dataset}/preact_resnet18"
    baseline = _parse_config(
        monkeypatch,
        f"{config_root}/baseline.yaml",
    )
    expected = {
        field: getattr(
            baseline,
            field,
        )
        for field in MAIN_PROTOCOL_FIELDS
    }

    for method in (
        "metaaugment",
        "ifaugnet",
    ):
        candidate = _parse_config(
            monkeypatch,
            f"{config_root}/{method}.yaml",
        )
        actual = {
            field: getattr(
                candidate,
                field,
            )
            for field in MAIN_PROTOCOL_FIELDS
        }
        assert actual == expected


@pytest.mark.parametrize(
    "dataset",
    VALIDATION_AWARE_DATASETS,
)
@pytest.mark.parametrize(
    "method",
    [
        "metaaugment",
        "ifaugnet",
    ],
)
def test_validation_aware_configs_use_global_batch_statistics(
    monkeypatch,
    dataset: str,
    method: str,
) -> None:
    """Treat PMAP as an execution backend while preserving global BatchNorm."""
    args = _parse_config(
        monkeypatch,
        f"configs/{dataset}/preact_resnet18/{method}.yaml",
    )

    assert args.distributed is True
    assert args.cross_device_shuffle is False
    assert args.sync_batch_stats is True


@pytest.mark.parametrize(
    "dataset",
    SYNC_DIST_BASELINE_DATASETS,
)
def test_syncdist_baseline_inherits_dataset_protocol(
    monkeypatch,
    dataset: str,
) -> None:
    """The backend audit must not import optimizer settings across datasets."""
    config_root = f"configs/{dataset}/preact_resnet18"
    baseline = _parse_config(
        monkeypatch,
        f"{config_root}/baseline.yaml",
    )
    syncdist = _parse_config(
        monkeypatch,
        f"{config_root}/baseline_syncdist.yaml",
    )

    assert {
        field: getattr(syncdist, field)
        for field in MAIN_PROTOCOL_FIELDS
    } == {
        field: getattr(baseline, field)
        for field in MAIN_PROTOCOL_FIELDS
    }
    assert syncdist.method == "baseline"
    assert syncdist.distributed is True
    assert syncdist.cross_device_shuffle is False
    assert syncdist.sync_batch_stats is True


@pytest.mark.parametrize(
    "dataset",
    SYNC_DIST_BASELINE_DATASETS,
)
def test_mixda_configs_share_dataset_classifier_protocol(
    monkeypatch,
    dataset: str,
) -> None:
    """Keep every MixDA comparison on its dataset-specific ERM protocol."""
    config_root = f"configs/{dataset}/preact_resnet18"
    baseline = _parse_config(
        monkeypatch,
        f"{config_root}/baseline.yaml",
    )
    expected = {
        field: getattr(baseline, field)
        for field in MIXDA_PROTOCOL_FIELDS
    }

    for method in MIXDA_METHODS:
        candidate = _parse_config(
            monkeypatch,
            f"{config_root}/{method}.yaml",
        )
        assert {
            field: getattr(candidate, field)
            for field in MIXDA_PROTOCOL_FIELDS
        } == expected
