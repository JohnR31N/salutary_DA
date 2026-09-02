"""Tests for the shared YAML configuration loader."""

from pathlib import Path

import pytest

from allthemix.config import load_optional_yaml_config, load_yaml_config


def test_load_yaml_config_merges_ordered_bases(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    child = tmp_path / "child.yaml"
    first.write_text(
        "optimizer:\n  learning_rate: 0.1\n  momentum: 0.9\nseed: 1\n",
        encoding="utf-8",
    )
    second.write_text(
        "optimizer:\n  learning_rate: 0.05\nepochs: 200\n",
        encoding="utf-8",
    )
    child.write_text(
        "bases:\n  - first.yaml\n  - second.yaml\noptimizer:\n"
        "  momentum: 0.95\nseed: 7\n",
        encoding="utf-8",
    )

    assert load_yaml_config(child) == {
        "optimizer": {
            "learning_rate": 0.05,
            "momentum": 0.95,
        },
        "seed": 7,
        "epochs": 200,
    }


def test_load_yaml_config_reports_inheritance_cycle(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base: second.yaml\n", encoding="utf-8")
    second.write_text("base: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular config base reference"):
        load_yaml_config(first)


def test_load_optional_yaml_config_accepts_empty_path() -> None:
    assert load_optional_yaml_config(None) == {}
    assert load_optional_yaml_config("") == {}


def test_shell_entrypoints_do_not_import_private_config_loaders() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts"
    private_names = {
        "_load_raw_yaml_config",
        "_load_yaml_config",
        "_merge_config_dicts",
        "_validate_config_keys",
    }

    for script in scripts.rglob("*.sh"):
        contents = script.read_text(encoding="utf-8")
        assert all(name not in contents for name in private_names)


@pytest.mark.parametrize(
    ("relative_path", "manifest"),
    [
        (
            "configs/cifar10/preact_resnet18/diffusemix.yaml",
            "./data/diffusemix/cifar10/paper_1x_256_seed0",
        ),
        (
            "configs/cifar100/preact_resnet18/diffusemix.yaml",
            "./data/diffusemix/cifar100/paper_1x_256_seed0",
        ),
        (
            "configs/stl10/preact_resnet18/diffusemix.yaml",
            "./data/diffusemix/stl10/paper_1x_256_seed0",
        ),
        (
            "configs/tiny_imagenet/preact_resnet18_xla4/diffusemix.yaml",
            "./data/diffusemix/tiny_imagenet/paper_1x_256_seed0",
        ),
        (
            "configs/imagenet100/preact_resnet18/diffusemix.yaml",
            "./data/diffusemix/imagenet100/paper_1x_256_seed0",
        ),
        (
            "configs/cars196/preact_resnet18/diffusemix.yaml",
            "./data/diffusemix/cars196/paper_1x_512_seed0",
        ),
        (
            "configs/cub200/preact_resnet18/diffusemix.yaml",
            "./data/diffusemix/cub200/paper_1x_512_full_seed0",
        ),
    ],
)
def test_main_table_diffusemix_configs_use_resolution_profile(
    relative_path: str,
    manifest: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(root / relative_path)

    assert config["method"] == "diffusemix"
    assert config["diffusemix_manifest"] == manifest
    assert config["diffusemix_train_mode"] == "append"
    expected_validation_split = 0.5 if "/stl10/" in relative_path else 0.1
    assert config["validation_split"] == expected_validation_split
    assert config["final_test_checkpoint"] == "best"


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/cars196/preact_resnet18/baseline.yaml",
        "configs/cars196/preact_resnet18/diffusemix.yaml",
        "configs/cars196/preact_resnet18/alia.yaml",
    ],
)
def test_cars196_genda_configs_default_to_sync_distributed(
    relative_path: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(root / relative_path)

    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_cars196_alia_uses_matched_append_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/cars196/preact_resnet18/alia.yaml"
    )

    assert config["dataset"] == "cars196"
    assert config["method"] == "alia"
    assert config["alia_manifest"] == "./data/alia/cars196"
    assert config["alia_train_mode"] == "append"
    assert config["validation_split"] == pytest.approx(0.1)
    assert config["final_test_checkpoint"] == "best"


def test_cub200_diffusemix_defaults_to_sync_distributed() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/cub200/preact_resnet18/diffusemix.yaml"
    )

    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_every_stl10_training_config_uses_half_official_test_for_vdev() -> None:
    """Keep every STL-10 method on the shared 4000/4000 eval split."""

    root = Path(__file__).resolve().parents[2]
    config_root = root / "configs/stl10/preact_resnet18"

    for config_path in sorted(config_root.glob("*.yaml")):
        config = load_yaml_config(config_path)
        assert config["validation_split"] == pytest.approx(0.5), config_path


def test_imagenet100_mixup_matches_syncdist_e400_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/imagenet100/preact_resnet18/mixup.yaml"
    )

    assert config["method"] == "mixup"
    assert config["epochs"] == 400
    assert config["mixup_alpha"] == pytest.approx(0.2)
    assert config["validation_split"] == pytest.approx(0.1)
    assert config["final_test_checkpoint"] == "best"
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_imagenet100_resizemix_matches_syncdist_e400_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/imagenet100/preact_resnet18/resize.yaml"
    )

    assert config["method"] == "resizemix"
    assert config["epochs"] == 400
    assert config["resizemix_scope_min"] == pytest.approx(0.1)
    assert config["resizemix_scope_max"] == pytest.approx(0.8)
    assert config["resizemix_prob"] == pytest.approx(1.0)
    assert config["validation_split"] == pytest.approx(0.1)
    assert config["final_test_checkpoint"] == "best"
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_imagenet100_saliencymix_matches_syncdist_e400_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/imagenet100/preact_resnet18/saliencymix.yaml"
    )

    assert config["method"] == "saliencymix"
    assert config["epochs"] == 400
    assert config["saliencymix_alpha"] == pytest.approx(1.0)
    assert config["saliencymix_prob"] == pytest.approx(0.5)
    assert config["sal_basic_aug"] is False
    assert config["sal_aug_recipe"] == "imagenet"
    assert config["validation_split"] == pytest.approx(0.1)
    assert config["final_test_checkpoint"] == "best"
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_imagenet100_guided_sr_matches_syncdist_e400_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/imagenet100/preact_resnet18/guided_sr.yaml"
    )

    assert config["method"] == "guided_sr"
    assert config["epochs"] == 400
    assert config["guidedmixup_alpha"] == pytest.approx(1.0)
    assert config["guidedmixup_prob"] == pytest.approx(0.5)
    assert config["guidedmixup_blur_kernel"] == 7
    assert config["guidedmixup_condition"] == "greedy"
    # guided_sr computes saliency on the fly: the standard-pipeline aug keys
    # must be active and the precomputed-saliency keys must be absent, or
    # train.py rejects the config at startup.
    assert config["basic_aug"] is False
    assert config["aug_recipe"] == "imagenet"
    assert "sal_aug_recipe" not in config
    assert "sal_basic_aug" not in config
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_imagenet100_catchupmix_matches_syncdist_e400_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/imagenet100/preact_resnet18/catchupmix.yaml"
    )

    assert config["method"] == "catchupmix"
    assert config["epochs"] == 400
    assert config["catchupmix_alpha"] == pytest.approx(1.0)
    assert config["catchupmix_cutmix_alpha"] == pytest.approx(1.0)
    assert config["catchupmix_num_layers"] == 5
    assert config["catchupmix_no_repeat"] is True
    assert config["validation_split"] == pytest.approx(0.1)
    assert config["final_test_checkpoint"] == "best"
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


@pytest.mark.parametrize(
    ("config_name", "method", "manifest_key", "manifest"),
    [
        (
            "diffusemix.yaml",
            "diffusemix",
            "diffusemix_manifest",
            "./data/diffusemix/stl10/paper_1x_256_seed0",
        ),
        (
            "alia.yaml",
            "alia",
            "alia_manifest",
            "./data/alia/stl10",
        ),
        (
            "saspa.yaml",
            "saspa",
            "saspa_manifest",
            "./data/saspa/stl10",
        ),
    ],
)
def test_stl10_genda_configs_share_matched_syncdist_protocol(
    config_name: str,
    method: str,
    manifest_key: str,
    manifest: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(
        root / "configs/stl10/preact_resnet18" / config_name
    )

    assert config["dataset"] == "stl10"
    assert config["method"] == method
    assert config[manifest_key] == manifest
    assert config["validation_split"] == pytest.approx(0.5)
    assert config["final_test_checkpoint"] == "best"
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["cross_device_shuffle"] is False


def test_stl10_saspa_wrapper_uses_nonvacuous_top3_filter() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (
        root / "scripts/experiment_run/filter_dataset_saspa.sh"
    ).read_text(encoding="utf-8")

    stl_case = script.split("    stl10)", maxsplit=1)[1].split(
        "        ;;", maxsplit=1
    )[0]
    assert "DEFAULT_TOP_K=3" in stl_case
