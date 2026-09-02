from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_parse_args_accepts_cutmix_sumix_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts cutmix sumix config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cutmix_sumix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar100",
                "model: preact_resnet18",
                "method: cutmix_sumix",
                "cutmix_alpha: 0.2",
                "cutmix_prob: 1.0",
                "cutmix_no_repeat: true",
                "sumix_gamma: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.method == "cutmix_sumix"
    assert args.cutmix_alpha == pytest.approx(0.2)
    assert args.cutmix_prob == pytest.approx(1.0)
    assert args.cutmix_no_repeat is True
    assert args.sumix_gamma == pytest.approx(0.5)


def test_parse_args_accepts_cosine_lr_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts cosine lr config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cutmix_sumix_cosine.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar100",
                "model: resnet18",
                "method: cutmix_sumix",
                "lr_schedule: cosine",
                "min_learning_rate: 0.0",
                "cutmix_alpha: 0.2",
                "cutmix_no_repeat: true",
                "sumix_gamma: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.lr_schedule == "cosine"
    assert args.min_learning_rate == pytest.approx(0.0)
    assert args.method == "cutmix_sumix"


def test_parse_args_accepts_reproducibility_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that reproducibility controls can be configured from YAML."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "reproducibility.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: preact_resnet18",
                "method: baseline",
                "seed: 7",
                "data_seed: 29",
                "deterministic_data: true",
                "strict_determinism: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.seed == 7
    assert args.data_seed == 29
    assert args.deterministic_data is True
    assert args.strict_determinism is True


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        (
            "--seed",
            "-1",
            "seed must be >= 0",
        ),
        (
            "--data_seed",
            "-2",
            "data_seed must be -1 or >= 0",
        ),
    ],
)
def test_parse_args_rejects_invalid_reproducibility_seed(
    flag: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that negative user seeds fail before training starts."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            flag,
            value,
        ],
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        parse_args()


def test_parse_args_accepts_warmup_cosine_lr_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts warmup cosine lr config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "warmup_cosine.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "lr_schedule: warmup_cosine",
                "warmup_epochs: 5",
                "min_learning_rate: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.lr_schedule == "warmup_cosine"
    assert args.warmup_epochs == 5
    assert args.min_learning_rate == pytest.approx(0.0)


def test_parse_args_accepts_step_cosine_lr_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts step cosine lr config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "step_cosine.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "lr_schedule: step_cosine",
                "lr_decay_epochs: [150]",
                "min_learning_rate: 0.0001",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.lr_schedule == "step_cosine"
    assert args.lr_decay_epochs == [
        150,
    ]
    assert args.min_learning_rate == pytest.approx(0.0001)


def test_parse_args_accepts_catchupmix_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts catchupmix config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "catchupmix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar100",
                "model: preact_resnet18",
                "method: catchupmix",
                "catchupmix_alpha: 0.5",
                "catchupmix_cutmix_alpha: 1.0",
                "catchupmix_num_layers: 5",
                "catchupmix_no_repeat: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.method == "catchupmix"
    assert args.catchupmix_alpha == pytest.approx(0.5)
    assert args.catchupmix_cutmix_alpha == pytest.approx(1.0)
    assert args.catchupmix_num_layers == 5
    assert args.catchupmix_no_repeat is True


def test_parse_args_accepts_fmix_per_sample_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts per-sample FMix config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "fmix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: preact_resnet18",
                "method: fmix",
                "fmix_alpha: 1.0",
                "fmix_decay: 3.0",
                "fmix_prob: 1.0",
                "fmix_per_sample: true",
                "fmix_no_repeat: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.method == "fmix"
    assert args.fmix_per_sample is True
    assert args.fmix_no_repeat is True


def test_parse_args_accepts_cross_device_shuffle_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts cross-device shuffle config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "fmix_cross_device.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: fmix",
                "distributed: true",
                "cross_device_shuffle: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.method == "fmix"
    assert args.distributed is True
    assert args.cross_device_shuffle is True


def test_parse_args_accepts_sync_batch_stats_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts synchronized BatchNorm config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "sync_bn.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "distributed: true",
                "sync_batch_stats: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.distributed is True
    assert args.sync_batch_stats is True


def test_parse_args_accepts_tiny_imagenet_normalization_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts Tiny ImageNet normalization config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "tiny_baseline.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: baseline",
                "tiny_imagenet_normalization: none",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.dataset == "tiny_imagenet"
    assert args.tiny_imagenet_normalization == "none"


def test_parse_args_accepts_validation_protocol_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts validation/test protocol fields."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "baseline.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: preact_resnet18",
                "method: baseline",
                "validation_split: 0.1",
                "eval_on_test_each_epoch: false",
                "final_test: true",
                "final_test_checkpoint: best",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.validation_split == pytest.approx(0.1)
    assert args.eval_on_test_each_epoch is False
    assert args.final_test is True
    assert args.final_test_checkpoint == "best"


def test_parse_args_defaults_final_test_checkpoint_to_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify validation protocols select the best checkpoint by default."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: preact_resnet18",
                "method: baseline",
                "validation_split: 0.1",
                "eval_on_test_each_epoch: false",
                "final_test: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.final_test_checkpoint == "best"


def test_parse_args_rejects_best_checkpoint_without_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify test data cannot be used to select the best checkpoint."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--final_test",
            "true",
        ],
    )

    with pytest.raises(
        ValueError,
        match="requires a held-out validation split",
    ):
        parse_args()


def test_parse_args_accepts_shuffle_buffer_size_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that dataset shuffle buffer size can be configured."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "tiny_baseline.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: baseline",
                "shuffle_buffer_size: 90000",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.shuffle_buffer_size == 90000


def test_parse_args_accepts_base_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that configs can inherit defaults from a base YAML file."""
    from allthemix.cli.args import parse_args

    base_path = tmp_path / "base.yaml"
    config_path = tmp_path / "cutmix.yaml"

    base_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: baseline",
                "epochs: 200",
                "validation_split: 0.1",
                "eval_on_test_each_epoch: false",
                "final_test: true",
                "final_test_checkpoint: best",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        "\n".join(
            [
                "base: base.yaml",
                "method: cutmix",
                "cutmix_alpha: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.dataset == "tiny_imagenet"
    assert args.method == "cutmix"
    assert args.epochs == 200
    assert args.validation_split == pytest.approx(0.1)
    assert args.final_test_checkpoint == "best"
    assert args.cutmix_alpha == pytest.approx(0.5)


def test_parse_args_accepts_cutmix_variant_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CutMix can select the Torchbearer-compatible variant."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cutmix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "cutmix_alpha: 1.0",
                "cutmix_prob: 1.0",
                "cutmix_variant: torchbearer",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.method == "cutmix"
    assert args.cutmix_variant == "torchbearer"


def test_parse_args_accepts_torchbearer_area_cutmix_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CutMix can select the clipped-area Torchbearer variant."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cutmix_area.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "cutmix_variant: torchbearer_area",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.cutmix_variant == "torchbearer_area"


def test_parse_args_accepts_cutmix_stabilization_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify parse args accepts stabilized CutMix config keys."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cutmix_stable.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "cutmix_per_sample_lam: true",
                "cutmix_min_lam: 0.7",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.cutmix_per_sample_lam is True
    assert args.cutmix_min_lam == 0.7


def test_parse_args_accepts_early_stop_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify early-stop settings can be configured from YAML."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cutmix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: cutmix",
                "early_stop_enabled: true",
                "early_stop_start_epoch: 40",
                "early_stop_patience: 12",
                "early_stop_min_delta: 0.005",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.early_stop_enabled is True
    assert args.early_stop_start_epoch == 40
    assert args.early_stop_patience == 12
    assert args.early_stop_min_delta == pytest.approx(0.005)


def test_parse_args_accepts_preact_stem_bn_relu_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PreActResNet stem compatibility can be configured."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "tiny.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "preact_stem_bn_relu: true",
                "method: baseline",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.preact_stem_bn_relu is True


def test_parse_args_accepts_preact_pytorch_default_init_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PreActResNet PyTorch default init can be configured."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "tiny.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "preact_pytorch_default_init: true",
                "method: baseline",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.preact_pytorch_default_init is True


def test_parse_args_cli_override_beats_base_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify command-line overrides still take precedence over base configs."""
    from allthemix.cli.args import parse_args

    base_path = tmp_path / "base.yaml"
    config_path = tmp_path / "mixup.yaml"

    base_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: preact_resnet18",
                "method: baseline",
                "epochs: 200",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        "\n".join(
            [
                "base: base.yaml",
                "method: mixup",
                "mixup_alpha: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--epochs",
            "5",
        ],
    )

    args = parse_args()

    assert args.method == "mixup"
    assert args.mixup_alpha == pytest.approx(0.2)
    assert args.epochs == 5


def test_parse_args_rejects_validation_eval_without_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that validation evaluation requires a held-out split."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--eval_on_test_each_epoch",
            "false",
        ],
    )

    with pytest.raises(ValueError):
        parse_args()


def test_parse_args_resolves_legacy_sal_basic_aug_to_basic_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that legacy sal_basic_aug still selects paired basic aug."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "saliencymix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: preact_resnet18",
                "method: saliencymix",
                "basic_aug: false",
                "sal_basic_aug: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.sal_basic_aug is True
    assert args.sal_aug_recipe == "basic"


def test_parse_args_accepts_saliencymix_per_sample_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SaliencyMix per-sample mode can be configured."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "saliencymix_per_sample.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: saliencymix",
                "basic_aug: false",
                "saliencymix_per_sample: true",
                "sal_aug_recipe: tiny_official",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.saliencymix_per_sample is True
    assert args.sal_aug_recipe == "tiny_official"


def test_parse_args_accepts_saliency_imagenet_aug_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that saliency configs can request paired ImageNet aug."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cars_saliencymix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: cars196",
                "model: preact_resnet18",
                "method: saliencymix",
                "basic_aug: false",
                "sal_basic_aug: false",
                "sal_aug_recipe: imagenet",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.sal_basic_aug is False
    assert args.sal_aug_recipe == "imagenet"


def test_parse_args_accepts_cub_aug_recipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CUB configs can request regular and paired CUB recipes."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cub_saliencymix.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: caltech_birds2011",
                "model: preact_resnet18",
                "method: saliencymix",
                "basic_aug: false",
                "aug_recipe: cub",
                "sal_basic_aug: false",
                "sal_aug_recipe: cub",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.aug_recipe == "cub"
    assert args.sal_aug_recipe == "cub"


def test_parse_args_accepts_fine_grained_aug_recipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fine-grained configs support regular and paired recipes."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "cars_saliencymix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: cars196",
                "model: preact_resnet18",
                "method: saliencymix",
                "basic_aug: false",
                "aug_recipe: fine_grained",
                "sal_basic_aug: false",
                "sal_aug_recipe: fine_grained",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.aug_recipe == "fine_grained"
    assert args.sal_aug_recipe == "fine_grained"


def test_parse_args_accepts_tiny_specific_aug_recipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Tiny ImageNet parity recipes are accepted by config parsing."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "tiny_openmixup.yaml"

    config_path.write_text(
        "\n".join(
            [
                "dataset: tiny_imagenet",
                "model: preact_resnet18",
                "method: saliencymix",
                "basic_aug: false",
                "aug_recipe: tiny_openmixup",
                "sal_basic_aug: false",
                "sal_aug_recipe: tiny_official",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.aug_recipe == "tiny_openmixup"
    assert args.sal_aug_recipe == "tiny_official"


def test_all_repo_configs_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that all repo configs parse."""
    from allthemix.cli.args import parse_args

    config_paths = sorted(
        Path("configs").rglob("*.yaml"),
    )

    assert config_paths

    for config_path in config_paths:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                str(config_path),
                "--salda_ga_git_commit",
                "a" * 40,
            ],
        )

        parse_args()


def test_all_repo_validation_configs_declare_formal_source_protocol() -> None:
    """Prevent configs from silently falling back to train-sourced val."""
    from allthemix.config import load_yaml_config

    config_paths = sorted(
        Path("configs").rglob("*.yaml"),
    )
    formal_count = 0

    for config_path in config_paths:
        raw_config = config_path.read_text(
            encoding="utf-8",
        )
        if "validation_split:" in raw_config:
            assert "val_source:" in raw_config, config_path

        config = load_yaml_config(
            config_path,
        )
        validation_split = float(
            config.get("validation_split", 0.0),
        )
        if validation_split <= 0.0:
            continue

        if float(config.get("train_subset_fraction", 1.0)) < 1.0:
            # Budget-honest low-data ablations intentionally retain the
            # legacy train-sourced split.
            assert config.get("val_source") == "train", config_path
            continue

        if (
            config.get("salda_ga_mode", "off") != "off"
            and int(config.get("salda_ga_stop_epoch", -1)) != -1
        ):
            assert config.get("val_source") == "test", config_path
            assert config.get("eval_on_test_each_epoch") is False, config_path
            if config.get("final_test") is True:
                assert config.get("final_test_checkpoint") == "best", config_path
                assert config.get("save_checkpoint") is True, config_path
                assert config.get("save_best_only") is True, config_path
            else:
                assert config.get("final_test") is False, config_path
            continue

        formal_count += 1
        assert config.get("val_source") == "test", config_path
        assert config.get("eval_on_test_each_epoch") is False, config_path
        assert config.get("final_test") is True, config_path
        assert (
            config.get("final_test_checkpoint", "best") == "best"
        ), config_path

    assert formal_count > 0


def test_ifaugnet_validation_is_scoped_to_ifaugnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure IF-only options cannot reject an unrelated training method."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar10/preact_resnet18/baseline.yaml",
            "--ifaugnet_pretrain_steps",
            "0",
        ],
    )

    args = parse_args()

    assert args.method == "baseline"


def test_ifaugnet_accepts_skipped_pretraining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow the paper's no-pretraining IF-AugNet ablation."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar10/preact_resnet18/ifaugnet.yaml",
            "--ifaugnet_pretrain_steps",
            "0",
        ],
    )

    args = parse_args()

    assert args.ifaugnet_pretrain_steps == 0


def test_ifaugnet_rejects_negative_pretraining_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject negative IF-AugNet stage lengths."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar10/preact_resnet18/ifaugnet.yaml",
            "--ifaugnet_pretrain_steps",
            "-1",
        ],
    )

    with pytest.raises(
        ValueError,
        match="ifaugnet_pretrain_steps must be >= 0",
    ):
        parse_args()


def test_ifaugnet_accepts_pretrain_policy_for_retrain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a clean pretrain-only retraining control."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet_jax_stable.yaml",
            "--ifaugnet_stage",
            "retrain",
            "--ifaugnet_retrain_policy_source",
            "pretrain",
        ],
    )

    args = parse_args()

    assert args.ifaugnet_stage == "retrain"
    assert args.ifaugnet_retrain_policy_source == "pretrain"


def test_ifaugnet_rejects_pretrain_policy_outside_retrain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep pretrain-only policy selection scoped to the retrain stage."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet_jax_stable.yaml",
            "--ifaugnet_stage",
            "influence",
            "--ifaugnet_retrain_policy_source",
            "pretrain",
        ],
    )

    with pytest.raises(
        ValueError,
        match="requires ifaugnet_stage=retrain",
    ):
        parse_args()


def test_ifaugnet_accepts_early_policy_classifier_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow saving an unsaturated classifier on the original trajectory."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet_jax_stable.yaml",
            "--ifaugnet_stage",
            "classifier",
            "--ifaugnet_policy_classifier_checkpoint",
            "early",
            "--ifaugnet_policy_classifier_save_epoch",
            "50",
        ],
    )

    args = parse_args()

    assert args.ifaugnet_policy_classifier_checkpoint == "early"
    assert args.ifaugnet_policy_classifier_save_epoch == 50


def test_ifaugnet_rejects_early_policy_classifier_without_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an explicit capture epoch when training the classifier."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet_jax_stable.yaml",
            "--ifaugnet_stage",
            "classifier",
            "--ifaugnet_policy_classifier_checkpoint",
            "early",
        ],
    )

    with pytest.raises(
        ValueError,
        match="early policy classifier requires",
    ):
        parse_args()


def test_ifaugnet_accepts_zero_pretrain_influence_ablation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permit a random-initialized influence policy without GAN pretraining."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet_jax_stable.yaml",
            "--ifaugnet_stage",
            "influence",
            "--ifaugnet_pretrain_steps",
            "0",
        ],
    )

    args = parse_args()

    assert args.ifaugnet_stage == "influence"
    assert args.ifaugnet_pretrain_steps == 0


def test_debug_train_source_accepts_matched_val_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept the leakage-diagnostic override on the standard ERM protocol."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/baseline_syncdist.yaml",
            "--validation_split",
            "0.3",
            "--val_source",
            "train",
            "--eval_on_test_each_epoch",
            "false",
            "--val_select_split_fraction",
            "0.5",
            "--debug_train_source",
            "val_only",
        ],
    )

    args = parse_args()

    assert args.debug_train_source == "val_only"
    assert args.validation_split == pytest.approx(0.3)


def test_validation_source_defaults_to_formal_official_eval_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held-out split must not silently fall back to official train."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--validation_split",
            "0.1",
            "--eval_on_test_each_epoch",
            "false",
        ],
    )

    args = parse_args()

    assert args.val_source == "test"


def test_validation_source_without_holdout_keeps_full_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No split leaves the source selector inert and legacy eval available."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
        ],
    )

    args = parse_args()

    assert args.validation_split == 0.0
    assert args.val_source == "train"


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/cifar100/preact_resnet18/saliencymix.yaml",
        "configs/cub200/preact_resnet18/diffusemix.yaml",
        "configs/cub200/preact_resnet18/alia.yaml",
        "configs/cub200/preact_resnet18/saspa.yaml",
    ],
)
def test_official_eval_validation_accepts_specialized_pipelines(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
) -> None:
    """Saliency and offline GenDA paths share the formal validation source."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            config_path,
        ],
    )

    args = parse_args()

    assert args.val_source == "test"


def test_official_eval_validation_seals_final_from_debug_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject per-epoch diagnostics that inspect the final-eval complement."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
                "configs/cifar100/preact_resnet18/ifaugnet.yaml",
        ],
    )

    with pytest.raises(
        ValueError,
        match="sealed",
    ):
        parse_args()


def test_debug_train_source_requires_val_eval_each_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject the override when epochs evaluate on test instead of val."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/baseline_syncdist.yaml",
            "--validation_split",
            "0.3",
            "--eval_on_test_each_epoch",
            "true",
            "--final_test",
            "false",
            "--debug_train_source",
            "train_plus_val",
        ],
    )

    with pytest.raises(
        ValueError,
        match="debug_train_source requires",
    ):
        parse_args()


def test_debug_train_source_rejects_validation_aware_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject methods that bypass the standard train pipeline path."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
                "configs/cifar100/preact_resnet18/ifaugnet.yaml",
            "--debug_train_source",
            "val_only",
        ],
    )

    with pytest.raises(
        ValueError,
        match="standard pipeline path",
    ):
        parse_args()


def test_cifar100_salda_config_locks_full_vdev_and_distributed_syncbn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active protocol accepts only the registered full-Vdev topology."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/salda_ga.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_mode",
            "noop",
            "--salda_ga_stop_epoch",
            "10",
            "--final_test",
            "false",
        ],
    )
    args = parse_args()
    assert args.dataset == "cifar100"
    assert args.salda_ga_mode == "noop"
    assert args.validation_split == pytest.approx(0.5)
    assert args.val_select_split_fraction == 0.0
    assert args.max_eval_steps == -1
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.final_test is False
    assert args.salda_ga_validation_direction_mode == "full"
    assert args.salda_ga_validation_batch_size == 500
    assert args.salda_ga_validation_reanchor_interval == 50
    assert not hasattr(args, "salda_ga_validation_chunk_size")


def test_standard_baseline_keeps_all_ga_phase_controls_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the ordinary baseline outside the optional Val-aware strategy."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/stl10/preact_resnet18/baseline.yaml",
        ],
    )
    args = parse_args()
    assert args.salda_ga_mode == "off"
    assert args.salda_ga_score_start_epoch == 0
    assert args.salda_ga_score_stop_epoch == -1
    assert args.salda_ga_action_start_epoch == 0
    assert args.salda_ga_action_stop_epoch == -1


@pytest.mark.parametrize(
    ("config_name", "mode", "dose"),
    [
        ("salda_ga_origin_baseline30.yaml", "baseline", 0.01),
        ("salda_ga_origin_impulse30.yaml", "soft_label", 0.1),
    ],
)
def test_stl10_origin_impulse_configs_freeze_one_action_epoch(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    mode: str,
    dose: float,
) -> None:
    """Freeze the paired origin run and its single post-warmup GA epoch."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            f"configs/stl10/preact_resnet18/{config_name}",
            "--salda_ga_git_commit",
            "a" * 40,
        ],
    )
    args = parse_args()
    assert args.method == "baseline"
    assert args.salda_ga_mode == mode
    assert args.validation_split == pytest.approx(0.5)
    assert args.val_source == "test"
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.salda_ga_stop_epoch == 30
    assert args.salda_ga_score_start_epoch == 20
    assert args.salda_ga_score_stop_epoch == 21
    assert args.salda_ga_action_start_epoch == 20
    assert args.salda_ga_action_stop_epoch == 21
    assert args.salda_ga_maximum_rows == 128
    assert args.salda_ga_minimum_gain == 0.0
    assert args.salda_ga_soft_label_dose == pytest.approx(dose)
    assert args.final_test is True
    assert args.final_test_checkpoint == "best"
    assert args.save_checkpoint is True
    assert args.save_best_only is True


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--max_train_steps", "1", "max_train_steps"),
        ("--save_checkpoint", "false", "save_checkpoint"),
        ("--save_best_only", "false", "save_best_only"),
        ("--final_test_checkpoint", "last", "final_test_checkpoint"),
    ],
)
def test_stl10_origin_endpoint_requires_complete_best_checkpoint_run(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    message: str,
) -> None:
    """Reject truncated or non-best endpoint configurations before training."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/stl10/preact_resnet18/salda_ga_origin_impulse30.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            flag,
            value,
        ],
    )
    with pytest.raises(ValueError, match=message):
        parse_args()


def test_complete_salda_training_still_requires_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the one sealed endpoint required after a complete SalDA run."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/salda_ga.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_mode",
            "noop",
            "--final_test",
            "false",
        ],
    )
    with pytest.raises(ValueError, match="requires final_test=true"):
        parse_args()


def test_continuous_action_must_start_before_score_window_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an action phase that can never observe a scored batch."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/stl10/preact_resnet18/salda_ga_origin_impulse30.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_action_start_epoch",
            "21",
            "--salda_ga_action_stop_epoch",
            "-1",
        ],
    )
    with pytest.raises(ValueError, match="action_start_epoch must precede"):
        parse_args()


def test_ga_phase_window_must_finish_within_short_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every finite GA phase inside the registered 30-epoch run."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/stl10/preact_resnet18/salda_ga_origin_impulse30.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_score_stop_epoch",
            "31",
            "--salda_ga_action_stop_epoch",
            "30",
        ],
    )
    with pytest.raises(ValueError, match="score_stop_epoch must not exceed"):
        parse_args()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--max_eval_steps", "16", "max_eval_steps"),
        ("--val_select_split_fraction", "0.5", "val_select_split_fraction"),
        ("--sync_batch_stats", "false", "sync_batch_stats"),
        ("--final_test", "true", "final_test"),
        ("--salda_ga_soft_label_dose", "0", "bounded values"),
        ("--salda_ga_minimum_gain", "-1", "threshold values"),
        (
            "--salda_ga_validation_batch_size",
            "300",
            "validation_batch_size",
        ),
        (
            "--salda_ga_validation_reanchor_interval",
            "49",
            "validation_reanchor_interval",
        ),
    ],
)
def test_cifar100_salda_protocol_rejects_truncation_or_invalid_policy(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    message: str,
) -> None:
    """Protocol and continuous-policy boundaries fail before data loading."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/salda_ga.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_mode",
            "noop",
            "--salda_ga_stop_epoch",
            "10",
            "--final_test",
            "false",
            flag,
            value,
        ],
    )
    with pytest.raises(ValueError, match=message):
        parse_args()


def test_cifar100_salda_accepts_batch_aggregate_and_rejects_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register the exploratory mode only for a fresh classifier-head run."""

    from allthemix.cli.args import parse_args

    base_argv = [
        "train.py",
        "--config",
        "configs/cifar100/preact_resnet18/salda_ga.yaml",
        "--salda_ga_git_commit",
        "a" * 40,
        "--salda_ga_mode",
        "soft_label",
        "--salda_ga_stop_epoch",
        "10",
        "--final_test",
        "false",
        "--salda_ga_validation_direction_mode",
        "batch_aggregate",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    args = parse_args()
    assert args.salda_ga_validation_direction_mode == "batch_aggregate"
    assert args.salda_ga_parameter_scope == "classifier_head"

    monkeypatch.setattr(
        sys,
        "argv",
        [*base_argv, "--resume_checkpoint", "outputs/checkpoint"],
    )
    with pytest.raises(ValueError, match="does not support resume_checkpoint"):
        parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [*base_argv, "--salda_ga_parameter_scope", "full"],
    )
    with pytest.raises(ValueError, match="requires classifier_head"):
        parse_args()


def test_cifar100_salda_action_requires_head_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediate actions use the classifier-head scoring implementation."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/salda_ga.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_mode",
            "soft_label",
            "--salda_ga_stop_epoch",
            "10",
            "--final_test",
            "false",
            "--salda_ga_parameter_scope",
            "full",
        ],
    )
    with pytest.raises(ValueError, match="classifier_head"):
        parse_args()


def test_cifar100_salda_accepts_origin_and_shuffled_control_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Origin ERM and shuffled reweight use explicit registered identifiers."""
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/salda_ga.yaml",
            "--method",
            "baseline",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_mode",
            "shuffled_reweight",
            "--salda_ga_stop_epoch",
            "10",
            "--final_test",
            "false",
        ],
    )
    args = parse_args()
    assert args.method == "baseline"
    assert args.salda_ga_mode == "shuffled_reweight"
