from __future__ import annotations

from contextlib import nullcontext
import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from allthemix.cli.args import parse_args as parse_training_args
from allthemix.cli.train import _compute_steps_per_epoch
from allthemix.cli.train_setup import _count_original_diffusemix_examples
from allthemix.competitors.diffusemix.compose import (
    blend_fractal,
    build_instruction,
    is_near_black,
    merge_original_and_generated,
)
from allthemix.competitors.diffusemix.generate import parse_args
from allthemix.competitors.diffusemix.editor import InstructPix2PixEditor
from allthemix.competitors.diffusemix.manifest import (
    count_manifest_examples,
    load_manifest_dataset,
    validate_manifest_for_training,
)
from allthemix.competitors.diffusemix.sources import (
    is_validation_class_index,
)
from allthemix.methods.selector import get_mixer
from allthemix.data.pipeline import (
    _interleave_datasets_by_cardinality,
    build_train_pipeline,
)
from allthemix.training.losses.loss_selector import (
    compute_train_loss_and_targets,
)


def _solid_image(
    value: int,
    size: tuple[int, int] = (
        8,
        8,
    ),
) -> Image.Image:
    return Image.fromarray(
        np.full(
            (
                size[1],
                size[0],
                3,
            ),
            value,
            dtype=np.uint8,
        )
    )


def _write_manifest(
    root: Path,
    *,
    dataset: str = "cifar10",
    validation_split: float = 0.1,
    label: int = 2,
) -> Path:
    image_path = root / "images" / "000002" / "sample.png"
    image_path.parent.mkdir(
        parents=True,
    )
    _solid_image(
        127,
        size=(
            64,
            64,
        ),
    ).save(
        image_path,
    )
    manifest_path = root / "manifest.jsonl"
    record = {
        "schema_version": 1,
        "image_path": image_path.relative_to(
            root,
        ).as_posix(),
        "label": label,
        "source_id": "cifar10:000000001",
        "augmentation_index": 0,
        "dataset": dataset,
        "source_partition": "train",
        "prompt": "sunset",
        "mask": "generated_left",
        "fractal_alpha": 0.2,
        "validation_split": validation_split,
    }
    manifest_path.write_text(
        json.dumps(
            record,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest_path


@pytest.mark.parametrize(
    (
        "device_kind",
        "expected_context",
    ),
    [
        ("xla", "no_grad"),
        ("cuda", "inference_mode"),
        ("cpu", "inference_mode"),
    ],
)
def test_editor_uses_xla_safe_inference_context(
    device_kind: str,
    expected_context: str,
) -> None:
    calls: list[str] = []

    class FakeTorch:
        @staticmethod
        def no_grad():
            calls.append(
                "no_grad",
            )

            return nullcontext()

        @staticmethod
        def inference_mode():
            calls.append(
                "inference_mode",
            )

            return nullcontext()

    editor = object.__new__(
        InstructPix2PixEditor,
    )
    editor._torch = FakeTorch()
    editor.device_kind = device_kind

    with editor._inference_context():
        pass

    assert calls == [
        expected_context,
    ]


def test_paper_mask_keeps_exact_generated_half() -> None:
    original = _solid_image(
        0,
    )
    generated = _solid_image(
        255,
    )

    hybrid = np.asarray(
        merge_original_and_generated(
            original=original,
            generated=generated,
            mask_name="generated_left",
            seam_width=0,
        )
    )

    np.testing.assert_array_equal(
        hybrid[:, :4],
        255,
    )
    np.testing.assert_array_equal(
        hybrid[:, 4:],
        0,
    )


def test_official_soft_seam_transitions_at_center() -> None:
    hybrid = np.asarray(
        merge_original_and_generated(
            original=_solid_image(
                0,
            ),
            generated=_solid_image(
                255,
            ),
            mask_name="generated_right",
            seam_width=4,
        )
    )

    assert np.all(
        hybrid[:, 0] == 0
    )
    assert np.all(
        hybrid[:, -1] == 255
    )
    assert 0 < int(
        hybrid[0, 3, 0]
    ) < 255


def test_fractal_blend_uses_paper_lambda() -> None:
    blended = np.asarray(
        blend_fractal(
            hybrid=_solid_image(
                0,
            ),
            fractal=_solid_image(
                255,
            ),
            fractal_alpha=0.2,
        )
    )

    np.testing.assert_array_equal(
        blended,
        51,
    )


def test_release_quantization_truncates_instead_of_rounding() -> None:
    hybrid = _solid_image(
        0,
    )
    fractal = _solid_image(
        3,
    )

    rounded = np.asarray(
        blend_fractal(
            hybrid=hybrid,
            fractal=fractal,
            fractal_alpha=0.2,
            quantization="round",
        )
    )
    truncated = np.asarray(
        blend_fractal(
            hybrid=hybrid,
            fractal=fractal,
            fractal_alpha=0.2,
            quantization="truncate",
        )
    )

    np.testing.assert_array_equal(
        rounded,
        1,
    )
    np.testing.assert_array_equal(
        truncated,
        0,
    )


def test_prompt_template_and_black_filter() -> None:
    assert build_instruction(
        "sunset",
    ) == "A transformed version of image into sunset"
    assert is_near_black(
        _solid_image(
            0,
        )
    )
    assert not is_near_black(
        _solid_image(
            255,
        )
    )


def test_python_split_matches_reciprocal_class_split() -> None:
    validation_indices = [
        index
        for index in range(
            25,
        )
        if is_validation_class_index(
            class_index=index,
            validation_split=0.1,
        )
    ]

    assert validation_indices == [
        0,
        10,
        20,
    ]


def test_manifest_loader_resizes_diffusion_output_for_cifar(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
    )

    dataset = load_manifest_dataset(
        manifest_path=manifest_path,
        image_size=32,
    )
    example = next(
        iter(
            dataset,
        )
    )

    assert tuple(
        example["image"].shape,
    ) == (
        32,
        32,
        3,
    )
    assert example["image"].dtype.name == "uint8"
    assert int(
        example["label"],
    ) == 2
    assert count_manifest_examples(
        manifest_path,
    ) == 1


def test_manifest_runs_through_cifar_training_pipeline(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
    )

    dataset = build_train_pipeline(
        name="cifar10",
        data_dir=str(
            tmp_path,
        ),
        batch_size=1,
        shuffle_buffer_size=1,
        drop_remainder=True,
        use_basic_augmentation=False,
        augmentation_recipe="none",
        validation_split=0.1,
        train_manifest_path=str(
            manifest_path,
        ),
        train_manifest_mode="replace",
    )
    images, labels = next(
        iter(
            dataset,
        )
    )

    assert tuple(
        images.shape,
    ) == (
        1,
        32,
        32,
        3,
    )
    assert tuple(
        labels.shape,
    ) == (
        1,
    )


def test_append_interleave_consumes_both_streams_from_the_start() -> None:
    original_ds = tf.data.Dataset.from_tensor_slices(
        [
            "original-0",
            "original-1",
        ]
    )
    generated_ds = tf.data.Dataset.from_tensor_slices(
        [
            "generated-0",
            "generated-1",
            "generated-2",
            "generated-3",
        ]
    )
    mixed_ds = _interleave_datasets_by_cardinality(
        original_ds=original_ds,
        generated_ds=generated_ds,
        original_example_count=2,
        generated_example_count=4,
    )
    values = [
        value.decode(
            "utf-8",
        )
        for value in mixed_ds.as_numpy_iterator()
    ]

    assert values[:3] == [
        "original-0",
        "generated-0",
        "generated-1",
    ]
    assert sorted(
        values,
    ) == [
        "generated-0",
        "generated-1",
        "generated-2",
        "generated-3",
        "original-0",
        "original-1",
    ]


def test_diffusemix_append_uses_exact_uneven_class_count() -> None:
    assert _count_original_diffusemix_examples(
        dataset="unused",
        data_dir="unused",
        num_classes=3,
        num_train_examples=37,
        validation_split=0.1,
        known_class_counts=(
            25,
            11,
            1,
        ),
    ) == 31


def test_diffusemix_append_counts_unknown_class_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class_counts = (
        25,
        11,
        1,
    )
    labels = tf.concat(
        [
            tf.fill(
                (count,),
                tf.cast(
                    label,
                    tf.int64,
                ),
            )
            for label, count in enumerate(
                class_counts,
            )
        ],
        axis=0,
    )
    raw_train_ds = tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.zeros(
                (
                    sum(
                        class_counts,
                    ),
                    1,
                ),
                dtype=tf.float32,
            ),
            "label": labels,
        }
    )
    monkeypatch.setattr(
        "allthemix.cli.train_setup.load_train_dataset",
        lambda **_kwargs: raw_train_ds,
    )

    assert _count_original_diffusemix_examples(
        dataset="custom",
        data_dir="unused",
        num_classes=3,
        num_train_examples=sum(
            class_counts,
        ),
        validation_split=0.1,
        known_class_counts=None,
    ) == 31


def test_zero_step_training_is_rejected_explicitly() -> None:
    with pytest.raises(
        ValueError,
        match="no full batch",
    ):
        _compute_steps_per_epoch(
            train_examples=7,
            batch_size=8,
        )


def test_manifest_rejects_dataset_and_split_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
    )

    assert validate_manifest_for_training(
        manifest_path=manifest_path,
        dataset="cifar10",
        num_classes=10,
        validation_split=None,
    ) == 1

    with pytest.raises(
        ValueError,
        match="dataset mismatch",
    ):
        validate_manifest_for_training(
            manifest_path=manifest_path,
            dataset="cifar100",
            num_classes=100,
            validation_split=0.1,
        )

    with pytest.raises(
        ValueError,
        match="generation-source split mismatch",
    ):
        validate_manifest_for_training(
            manifest_path=manifest_path,
            dataset="cifar10",
            num_classes=10,
            validation_split=0.2,
        )


@pytest.mark.parametrize(
    ("dataset", "validation_split", "match"),
    [
        ("", 0.1, "dataset must be a nonempty name"),
        ("cifar10", float("nan"), "must be finite"),
    ],
)
def test_manifest_rejects_missing_dataset_and_nonfinite_split(
    tmp_path: Path,
    dataset: str,
    validation_split: float,
    match: str,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        dataset=dataset,
        validation_split=validation_split,
    )

    with pytest.raises(
        ValueError,
        match=match,
    ):
        validate_manifest_for_training(
            manifest_path=manifest_path,
            dataset="cifar10",
            num_classes=10,
            validation_split=0.1,
        )


def test_manifest_directory_rejects_missing_generation_shard(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
    )
    manifest_path.replace(
        tmp_path / "manifest-00000-of-00002.jsonl"
    )

    with pytest.raises(
        ValueError,
        match="artifact directory",
    ):
        count_manifest_examples(
            tmp_path / "manifest-00000-of-00002.jsonl",
        )

    with pytest.raises(
        ValueError,
        match="incomplete manifest shard set",
    ):
        count_manifest_examples(
            tmp_path,
        )


def test_manifest_rejects_mixed_generation_fingerprints(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
    )
    first = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )
    first["config_fingerprint"] = "a" * 64
    second_image_path = tmp_path / "images" / "000002" / "sample-2.png"
    _solid_image(
        128,
        size=(
            64,
            64,
        ),
    ).save(
        second_image_path,
    )
    second = {
        **first,
        "image_path": second_image_path.relative_to(
            tmp_path,
        ).as_posix(),
        "augmentation_index": 1,
        "config_fingerprint": "b" * 64,
    }
    manifest_path.write_text(
        json.dumps(
            first,
        )
        + "\n"
        + json.dumps(
            second,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="different config_fingerprint",
    ):
        count_manifest_examples(
            manifest_path,
        )


def test_presets_resolve_paper_and_release_semantics() -> None:
    common = [
        "--dataset",
        "cifar100",
        "--fractal-dir",
        "fractals",
        "--output-dir",
        "output",
    ]
    paper = parse_args(
        common,
    )
    release = parse_args(
        [
            *common,
            "--preset",
            "official_release",
        ]
    )

    assert paper.generation_size == 512
    assert paper.compact_output is False
    assert paper.output_size == 512
    assert paper.prompt_policy == "random"
    assert paper.mask_mode == "paper"
    assert paper.prompts[0] == "autumn"
    assert paper.prompt_template.startswith(
        "A transformed version"
    )
    assert release.generation_size == 512
    assert release.compact_output is False
    assert release.output_size == 512
    assert release.prompt_policy == "all"
    assert release.mask_mode == "official_code"
    assert release.prompts[:4] == (
        "Autumn",
        "snowy",
        "watercolor art",
        "sunset",
    )
    assert release.prompt_template == "{prompt}"


@pytest.mark.parametrize(
    (
        "dataset",
        "output_size",
    ),
    [
        ("cifar10", 32),
        ("cifar100", 32),
        ("svhn_cropped", 32),
        ("tiny_imagenet", 64),
        ("stl10", 96),
        ("caltech_birds2011", 224),
        ("cars196", 224),
        ("imagenet100", 224),
        ("oxford_iiit_pet", 224),
    ],
)
def test_compact_output_resolves_classifier_size(
    dataset: str,
    output_size: int,
) -> None:
    args = parse_args(
        [
            "--dataset",
            dataset,
            "--fractal-dir",
            "fractals",
            "--output-dir",
            "output",
            "--compact-output",
        ]
    )

    assert args.compact_output is True
    assert args.output_size == output_size


def test_compact_output_rejects_upscaling() -> None:
    with pytest.raises(
        SystemExit,
    ):
        parse_args(
            [
                "--dataset",
                "cars196",
                "--fractal-dir",
                "fractals",
                "--output-dir",
                "output",
                "--generation-size",
                "128",
                "--compact-output",
            ]
        )


def test_scheduler_config_normalization_is_set_order_stable() -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    first = {
        "_use_default_values": {
            "steps_offset",
            "timestep_spacing",
        },
        "num_train_timesteps": 1_000,
    }
    second = {
        "num_train_timesteps": 1_000,
        "_use_default_values": {
            "timestep_spacing",
            "steps_offset",
        },
    }

    assert generate_module._stable_json_value(
        first,
    ) == generate_module._stable_json_value(
        second,
    )


def test_scheduler_behavior_ignores_default_value_bookkeeping() -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    first = {
        "_use_default_values": {
            "steps_offset",
        },
        "beta_start": 0.00085,
        "num_train_timesteps": 1_000,
    }
    second = {
        "_use_default_values": {
            "timestep_spacing",
            "steps_offset",
        },
        "beta_start": 0.00085,
        "num_train_timesteps": 1_000,
    }

    assert generate_module._scheduler_behavior_config(
        first,
    ) == generate_module._scheduler_behavior_config(
        second,
    )


def test_run_config_reuses_equivalent_legacy_scheduler_fingerprint(
    tmp_path: Path,
) -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    editor = SimpleNamespace(
        device_kind="xla",
        device="xla:0",
        dtype="torch.bfloat16",
    )
    legacy_scheduler = {
        "_use_default_values": [
            "steps_offset",
        ],
        "beta_start": 0.00085,
    }
    current_scheduler = {
        "_use_default_values": [
            "steps_offset",
            "timestep_spacing",
        ],
        "beta_start": 0.00085,
    }
    common_config = {
        "dataset": "caltech_birds2011",
        "resolved_device": "xla",
    }
    legacy_config = {
        **common_config,
        "scheduler_config_sha256": generate_module._fingerprint(
            legacy_scheduler,
        ),
    }
    legacy_fingerprint = generate_module._fingerprint(
        legacy_config,
    )
    first_result = generate_module._write_run_config(
        output_dir=tmp_path,
        config=legacy_config,
        config_fingerprint=legacy_fingerprint,
        fractal_catalog=[],
        scheduler_config=legacy_scheduler,
        editor=editor,
    )
    current_config = {
        **common_config,
        "scheduler_config_sha256": generate_module._fingerprint(
            generate_module._scheduler_behavior_config(
                current_scheduler,
            )
        ),
    }
    current_fingerprint = generate_module._fingerprint(
        current_config,
    )
    second_result = generate_module._write_run_config(
        output_dir=tmp_path,
        config=current_config,
        config_fingerprint=current_fingerprint,
        fractal_catalog=[],
        scheduler_config=current_scheduler,
        editor=editor,
    )

    assert first_result == legacy_fingerprint
    assert second_result == legacy_fingerprint
    assert (
        current_config["scheduler_config_sha256"]
        == legacy_config["scheduler_config_sha256"]
    )
    assert not list(
        tmp_path.glob(
            "run_config.*.tmp",
        )
    )


def test_resume_accepts_equivalent_legacy_scheduler_fingerprint(
    tmp_path: Path,
) -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    legacy_scheduler = {
        "_use_default_values": (
            "{'timestep_spacing', 'steps_offset'}"
        ),
        "num_train_timesteps": 1_000,
    }
    current_scheduler = {
        "_use_default_values": {
            "steps_offset",
            "timestep_spacing",
        },
        "num_train_timesteps": 1_000,
    }
    common_config = {
        "dataset": "stl10",
        "resolved_device": "xla",
        "resolved_dtype": "bfloat16",
        "resolved_model_commit": "a" * 40,
    }
    legacy_scheduler_fingerprint = generate_module._fingerprint(
        legacy_scheduler,
    )
    existing_config = {
        **common_config,
        "scheduler_config_sha256": legacy_scheduler_fingerprint,
    }
    existing_fingerprint = generate_module._fingerprint(
        existing_config,
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config_fingerprint": existing_fingerprint,
                "config": existing_config,
                "runtime": {
                    "scheduler_config": legacy_scheduler,
                },
            }
        ),
        encoding="utf-8",
    )
    current_config = {
        **common_config,
        "scheduler_config_sha256": generate_module._fingerprint(
            generate_module._stable_json_value(
                current_scheduler,
            )
        ),
    }

    resolved = generate_module._resolve_config_fingerprint(
        output_dir=tmp_path,
        config=current_config,
        scheduler_config=current_scheduler,
    )

    assert resolved == existing_fingerprint
    assert (
        current_config["scheduler_config_sha256"]
        == legacy_scheduler_fingerprint
    )


def test_resume_rejects_changed_legacy_scheduler_config(
    tmp_path: Path,
) -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    legacy_scheduler = {
        "_use_default_values": "{'timestep_spacing'}",
        "beta_start": 0.00085,
    }
    common_config = {
        "dataset": "stl10",
        "resolved_device": "xla",
    }
    existing_config = {
        **common_config,
        "scheduler_config_sha256": generate_module._fingerprint(
            legacy_scheduler,
        ),
    }
    existing_fingerprint = generate_module._fingerprint(
        existing_config,
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config_fingerprint": existing_fingerprint,
                "config": existing_config,
                "runtime": {
                    "scheduler_config": legacy_scheduler,
                },
            }
        ),
        encoding="utf-8",
    )
    changed_scheduler = {
        "_use_default_values": {
            "steps_offset",
        },
        "beta_start": 0.001,
    }
    current_config = {
        **common_config,
        "scheduler_config_sha256": generate_module._fingerprint(
            generate_module._stable_json_value(
                changed_scheduler,
            )
        ),
    }
    expected_current = generate_module._fingerprint(
        current_config,
    )

    resolved = generate_module._resolve_config_fingerprint(
        output_dir=tmp_path,
        config=current_config,
        scheduler_config=changed_scheduler,
    )

    assert resolved == expected_current


def test_diffusemix_uses_identity_mixer_and_hard_label_loss() -> None:
    images = jnp.ones(
        (
            2,
            4,
            4,
            3,
        ),
        dtype=jnp.float32,
    )
    labels = jnp.asarray(
        [
            0,
            1,
        ],
        dtype=jnp.int32,
    )
    mixer = get_mixer(
        name="diffusemix",
        num_classes=2,
    )
    mixer_output = mixer(
        rng=jax.random.PRNGKey(
            0,
        ),
        images=images,
        labels=labels,
    )
    logits = jnp.asarray(
        [
            [
                1.0,
                0.0,
            ],
            [
                0.0,
                1.0,
            ],
        ]
    )

    loss, targets = compute_train_loss_and_targets(
        method="diffusemix",
        logits=logits,
        mixer_output=mixer_output,
        num_classes=2,
    )

    np.testing.assert_allclose(
        np.asarray(
            mixer_output[0],
        ),
        np.asarray(
            images,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            targets,
        ),
        np.asarray(
            labels,
        ),
    )
    assert float(
        loss,
    ) > 0.0


@pytest.mark.parametrize(
    (
        "compact_args",
        "expected_output_size",
    ),
    [
        ([], 64),
        (["--compact-output"], 32),
    ],
)
def test_generator_streams_manifest_and_resumes_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compact_args: list[str],
    expected_output_size: int,
) -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    train_dir = tmp_path / "train"
    class_dir = train_dir / "class-a"
    class_dir.mkdir(
        parents=True,
    )
    _solid_image(
        64,
        size=(
            16,
            16,
        ),
    ).save(
        class_dir / "source.png",
    )
    fractal_dir = tmp_path / "fractals"
    fractal_dir.mkdir()
    _solid_image(
        128,
        size=(
            16,
            16,
        ),
    ).save(
        fractal_dir / "f.png",
    )
    output_dir = tmp_path / "output"

    class FakeEditor:
        def __init__(
            self,
            _config,
        ) -> None:
            self.device_kind = "cpu"
            self.device = "cpu"
            self.dtype = "float32"
            self.pipeline = SimpleNamespace(
                scheduler=SimpleNamespace(
                    config={
                        "name": "fake-euler",
                    }
                ),
                config=SimpleNamespace(
                    _commit_hash="fake-model-commit",
                ),
            )

        def edit(
            self,
            image: Image.Image,
            instruction: str,
            seed: int,
        ) -> Image.Image:
            del instruction, seed

            return _solid_image(
                220,
                size=image.size,
            )

        def synchronize(self) -> None:
            return None

    monkeypatch.setattr(
        generate_module,
        "InstructPix2PixEditor",
        FakeEditor,
    )
    args = generate_module.parse_args(
        [
            "--train-dir",
            str(
                train_dir,
            ),
            "--dataset-name",
            "cifar10",
            "--fractal-dir",
            str(
                fractal_dir,
            ),
            "--output-dir",
            str(
                output_dir,
            ),
            "--generation-size",
            "64",
            *compact_args,
            "--device",
            "cpu",
        ]
    )

    first = generate_module.generate(
        args,
    )
    second = generate_module.generate(
        args,
    )
    initial_record = json.loads(
        (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8",
        )
    )
    generated_path = output_dir / initial_record["image_path"]
    with Image.open(
        generated_path,
    ) as generated_image:
        assert generated_image.size == (
            expected_output_size,
            expected_output_size,
        )
    assert initial_record["generation_size"] == 64
    assert initial_record["compact_output"] is bool(
        compact_args,
    )
    assert initial_record["output_size"] == expected_output_size
    assert initial_record["output_resize"] == (
        "pillow-bilinear" if compact_args else "none"
    )
    _solid_image(
        1,
        size=(
            16,
            16,
        ),
    ).save(
        generated_path,
    )
    repaired = generate_module.generate(
        args,
    )
    records = validate_manifest_for_training(
        manifest_path=output_dir,
        dataset="cifar10",
        num_classes=10,
        validation_split=0.0,
    )

    assert first["generated"] == 1
    assert first["resumed"] == 0
    assert second["generated"] == 0
    assert second["resumed"] == 1
    assert repaired["generated"] == 1
    assert repaired["resumed"] == 0
    assert records == 1
    record = json.loads(
        (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8",
        )
    )
    assert record["source_partition"] == "train"
    assert len(
        record["output_png_sha256"],
    ) == 64
    assert len(
        record["fractal_sha256"],
    ) == 64
    summary = json.loads(
        (output_dir / "summary.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["complete"] is True
    assert summary["manifest_record_count"] == 1

    summary["complete"] = False
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="not complete",
    ):
        validate_manifest_for_training(
            manifest_path=output_dir,
            dataset="cifar10",
            num_classes=10,
            validation_split=0.0,
        )

    summary["complete"] = True
    summary["manifest_record_count"] = 2
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="manifest_record_count mismatch",
    ):
        validate_manifest_for_training(
            manifest_path=output_dir,
            dataset="cifar10",
            num_classes=10,
            validation_split=0.0,
        )


def test_incomplete_summary_keeps_source_catalog_baseline(
    tmp_path: Path,
) -> None:
    generate_module = importlib.import_module(
        "allthemix.competitors.diffusemix.generate"
    )
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.jsonl"
    catalog_sha256 = "a" * 64
    fingerprint = "test-fingerprint"

    generate_module._write_generation_summary(
        summary_path=summary_path,
        manifest_path=manifest_path,
        shard_index=0,
        num_shards=1,
        config_fingerprint=fingerprint,
        complete=False,
        manifest_record_count=0,
        source_catalog_sha256=catalog_sha256,
    )

    assert generate_module._previous_source_catalog_sha256(
        summary_path=summary_path,
        config_fingerprint=fingerprint,
    ) == catalog_sha256


def test_training_cli_requires_manifest_for_diffusemix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--method",
            "diffusemix",
        ],
    )

    with pytest.raises(
        ValueError,
        match="requires diffusemix_manifest",
    ):
        parse_training_args()


def test_training_cli_accepts_diffusemix_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--method",
            "diffusemix",
            "--diffusemix_manifest",
            "artifact/manifest.jsonl",
            "--diffusemix_train_mode",
            "append",
        ],
    )

    args = parse_training_args()

    assert args.diffusemix_manifest == "artifact/manifest.jsonl"
    assert args.diffusemix_train_mode == "append"


def test_training_cli_normalizes_yaml_diffusemix_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "diffusemix.yaml"
    config_path.write_text(
        "method: diffusemix\n"
        "diffusemix_manifest: artifact/manifest.jsonl\n"
        "diffusemix_train_mode: Append\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--config",
            str(
                config_path,
            ),
        ],
    )

    args = parse_training_args()

    assert args.diffusemix_train_mode == "append"


def test_cub200_generation_wrapper_preserves_full_512px_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper = (
        root / "scripts/experiment_run/generate_cub200_diffusemix.sh"
    ).read_text(encoding="utf-8")

    assert "export GENERATION_SIZE=512" in wrapper
    assert "export COMPACT_OUTPUT=false" in wrapper
    assert "paper_1x_512_full_seed0" in wrapper
