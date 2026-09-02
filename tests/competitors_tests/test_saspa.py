from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from allthemix.competitors.generative.sources import SourceExample
from allthemix.competitors.saspa.cli import parse_args as parse_saspa_args
from allthemix.competitors.saspa.editor import SaSPAEditor
from allthemix.competitors.saspa.filtering import (
    filter_classifier_stage,
    score_semantic_stage,
)
from allthemix.competitors.saspa.generation import generate_images
from allthemix.competitors.saspa.manifest import (
    read_stage_records,
    replacement_catalog,
    validate_manifest_for_training,
)
from allthemix.competitors.saspa.prompts import (
    PROMPT_SOURCES,
    SEMANTIC_PROMPTS,
    format_scene_prompt,
    load_official_prompts,
    normalize_dataset_name,
    semantic_negative_prompts,
)
from allthemix.data.utils.source_replacement import (
    attach_source_indices,
    build_source_replacer,
)
from allthemix.data.pipeline import build_train_pipeline
from allthemix.methods.selector import get_mixer
from allthemix.training.losses.loss_selector import (
    compute_train_loss_and_targets,
)


def _image(value: int, size: int = 16) -> Image.Image:
    return Image.fromarray(
        np.full((size, size, 3), value, dtype=np.uint8)
    )


class _FakeBfloatTensor:
    def __init__(self, values, trace=None) -> None:
        self.values = np.asarray(values)
        self.trace = trace if trace is not None else []

    def __getitem__(self, index):
        return _FakeBfloatTensor(self.values[index], self.trace)

    def detach(self):
        self.trace.append("detach")

        return self

    def float(self):
        self.trace.append("float")

        return self

    def cpu(self):
        self.trace.append("cpu")

        return self

    def permute(self, *dimensions):
        self.trace.append("permute")
        self.values = np.transpose(self.values, dimensions)

        return self

    def numpy(self):
        self.trace.append("numpy")
        if "float" not in self.trace:
            raise TypeError("Got unsupported ScalarType BFloat16")

        return self.values.astype(np.float32)


class _FakePipeline:
    def __init__(self) -> None:
        self.kwargs = None
        self.trace = []

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        values = np.array(
            [
                [
                    [[0.0, 1.0], [0.5, 0.25]],
                    [[1.0, 0.0], [0.5, 0.25]],
                    [[0.0, 0.5], [1.0, 0.25]],
                ]
            ],
            dtype=np.float32,
        )

        return SimpleNamespace(
            images=_FakeBfloatTensor(values, self.trace),
        )


class _FakeEditorRuntime:
    def __init__(self, trace=None) -> None:
        self.marked = False
        self.trace = trace

    def inference_context(self):
        return nullcontext()

    def cpu_generator(self, seed: int):
        return seed

    def mark_step(self) -> None:
        self.marked = True
        if self.trace is not None:
            self.trace.append("mark_step")


def test_saspa_editor_casts_bfloat_output_before_numpy() -> None:
    editor = SaSPAEditor.__new__(SaSPAEditor)
    editor.config = SimpleNamespace(
        guidance_scale=7.5,
        num_inference_steps=30,
        negative_prompt="bad image",
    )
    editor.pipeline = _FakePipeline()
    editor.runtime = _FakeEditorRuntime(editor.pipeline.trace)

    image = editor.edit(
        reference_image=_image(10, size=2),
        conditioning_image=_image(20, size=2),
        prompt="a bird on a branch",
        superclass="bird",
        seed=3,
        width=2,
        height=2,
    )

    assert editor.pipeline.kwargs["output_type"] == "pt"
    assert editor.runtime.marked is True
    assert editor.pipeline.trace == [
        "detach",
        "float",
        "mark_step",
        "cpu",
        "permute",
        "numpy",
    ]
    np.testing.assert_array_equal(
        np.asarray(image),
        np.array(
            [
                [[0, 255, 0], [255, 0, 128]],
                [[128, 128, 255], [64, 64, 64]],
            ],
            dtype=np.uint8,
        ),
    )


class _FakeEditor:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace(device_kind="cpu")
        self.config = SimpleNamespace(
            model_id="fake/saspa",
            guidance_scale=7.5,
            num_inference_steps=30,
            negative_prompt="bad image",
        )
        self.resolved_model_commit = "fake-commit"
        self.calls = []
        self.synchronized = False

    def edit(self, **kwargs) -> Image.Image:
        self.calls.append(kwargs)

        return _image(180, size=kwargs["width"])

    def synchronize(self) -> None:
        self.synchronized = True


class _FakeSemanticScorer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_id="fake/clip",
            logit_scale=100.0,
        )
        self.resolved_model_commit = "fake-clip-commit"
        self.positive_prompts = ("a photo of a bird",)
        self.negative_prompts = ("a photo of an object",)
        self.synchronized = False

    def score(self, images):
        return [
            {
                "semantic_pass": True,
                "clip_prediction_index": 0,
                "clip_prediction_text": "a photo of a bird",
                "clip_positive_probability": 0.9,
            }
            for _image_value in images
        ]

    def synchronize(self) -> None:
        self.synchronized = True


def _generate_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    import allthemix.competitors.saspa.generation as generation

    monkeypatch.setattr(
        generation,
        "make_canny_image",
        lambda image, **_kwargs: image,
    )
    sources = [
        SourceExample(
            index=0,
            source_id="cub:0",
            source_ref="cub:train:0",
            label=0,
            class_name="Black footed Albatross",
            image=_image(10),
        ),
        SourceExample(
            index=1,
            source_id="cub:1",
            source_ref="cub:train:1",
            label=1,
            class_name="Laysan Albatross",
            image=_image(20),
        ),
    ]
    editor = _FakeEditor()
    summary = generate_images(
        editor=editor,
        sources=sources,
        prompts=("a photo of a bird on a branch.",),
        output_dir=tmp_path,
        dataset="caltech_birds2011",
        validation_split=0.1,
        superclass="bird",
        images_per_source=1,
        generation_size=64,
        source_resize="letterbox",
        compact_size=16,
    )

    assert summary["record_count"] == 2
    assert editor.synchronized is True
    assert all(
        call["reference_image"].size == (64, 64)
        for call in editor.calls
    )

    return tmp_path


def test_official_prompt_assets_and_class_injection() -> None:
    cub_prompts = load_official_prompts("caltech_birds2011")
    cars_prompts = load_official_prompts("cars196")
    stl_prompts = load_official_prompts("stl10")

    assert len(cub_prompts) == 100
    assert len(cars_prompts) == 100
    assert len(stl_prompts) == 20
    assert format_scene_prompt(
        "a photo of a bird on rocks.",
        "001.Black_footed_Albatross",
        "bird",
    ) == "a photo of a Black footed Albatross bird on rocks"
    assert format_scene_prompt(
        stl_prompts[0],
        "airplane",
        "object",
    ) == "a photo showing airplane outdoors during daytime"


def test_saspa_stl10_extension_has_nonvacuous_semantic_contract() -> None:
    assert normalize_dataset_name("STL10") == "stl10"
    assert PROMPT_SOURCES["stl10"] == "allthemix-stl10-extension-v1"
    assert SEMANTIC_PROMPTS["stl10"] == (
        "a photo of an animal",
        "a photo of a vehicle",
    )
    assert "a photo of an object" not in semantic_negative_prompts("stl10")

    args = parse_saspa_args(
        [
            "generate",
            "--dataset",
            "stl10",
            "--output-dir",
            "unused",
        ]
    )
    assert args.num_inference_steps == 0
    assert args.images_per_source == 2


def test_saspa_cli_uses_official_dataset_step_defaults() -> None:
    args = parse_saspa_args(
        [
            "generate",
            "--dataset",
            "cars196",
            "--output-dir",
            "unused",
        ]
    )

    assert args.num_inference_steps == 0
    assert args.images_per_source == 2
    assert args.guidance_scale == pytest.approx(7.5)


def test_semantic_filter_module_imports_without_jax() -> None:
    code = """
import builtins

original_import = builtins.__import__

def import_without_jax(name, *args, **kwargs):
    if name == "jax" or name.startswith("jax."):
        raise ModuleNotFoundError("blocked JAX import in XLA environment")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_jax
from allthemix.competitors.saspa.filtering import score_semantic_stage
assert callable(score_semantic_stage)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_generation_semantic_and_classifier_filter_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _generate_artifact(tmp_path, monkeypatch)
    generated = read_stage_records(
        root,
        stage="generated",
        require_complete=True,
    )
    assert len(generated) == 2
    assert all(record["reference_same_class"] for record in generated)

    scorer = _FakeSemanticScorer()
    semantic_summary = score_semantic_stage(
        scorer=scorer,
        artifact_dir=root,
        batch_size=2,
    )
    assert semantic_summary["semantic_pass"] == 2
    assert scorer.synchronized is True

    import allthemix.competitors.saspa.filtering as filtering

    config_path = root / "baseline.yaml"
    checkpoint_path = root / "best"
    config_path.write_text("method: baseline\n", encoding="utf-8")
    checkpoint_path.write_text("fake checkpoint", encoding="utf-8")
    monkeypatch.setattr(filtering, "_resolve_checkpoint", lambda path: Path(path))
    monkeypatch.setattr(
        filtering,
        "_load_baseline_state",
        lambda _config, _checkpoint: (
            {
                "dataset": "caltech_birds2011",
                "validation_split": 0.1,
                "model": "preact_resnet18",
            },
            SimpleNamespace(num_classes=200),
            object(),
        ),
    )
    monkeypatch.setattr(
        filtering,
        "_checkpoint_identity",
        lambda _path: "fake-checkpoint-identity",
    )
    monkeypatch.setattr(filtering, "get_preprocessor", lambda *_a, **_k: object())
    monkeypatch.setattr(
        filtering,
        "_preprocess_image",
        lambda _image_value, label, _preprocess: np.full(
            (2, 2, 3), label, dtype=np.float32
        ),
    )

    def predictor(images):
        probabilities = np.zeros((len(images), 200), dtype=np.float32)
        for index, image in enumerate(images):
            label = int(image[0, 0, 0])
            probabilities[index, label] = 1.0

        return probabilities

    monkeypatch.setattr(
        filtering,
        "_make_predictor",
        lambda **_kwargs: predictor,
    )
    summary = filter_classifier_stage(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        artifact_dir=root,
        top_k=10,
        batch_size=2,
        distributed=False,
    )

    assert summary["kept_records"] == 2
    assert validate_manifest_for_training(
        root,
        dataset="caltech_birds2011",
        num_classes=200,
        validation_split=0.1,
    ) == 2
    assert validate_manifest_for_training(
        root,
        dataset="caltech_birds2011",
        num_classes=200,
        validation_split=None,
    ) == 2
    indices, paths, labels = replacement_catalog(root)
    assert indices == [0, 1]
    assert labels == [0, 1]
    assert all(len(source_paths) == 1 for source_paths in paths)


def test_source_replacement_is_aligned_and_cardinality_preserving(
    tmp_path: Path,
) -> None:
    replacement_path = tmp_path / "replacement.png"
    _image(255, size=4).save(replacement_path)
    source = tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.constant(
                np.stack(
                    [
                        np.zeros((4, 4, 3), dtype=np.uint8),
                        np.full((4, 4, 3), 10, dtype=np.uint8),
                    ]
                )
            ),
            "label": tf.constant([0, 1], dtype=tf.int64),
        }
    )
    replacer = build_source_replacer(
        source_indices=[1],
        generated_paths=[[str(replacement_path)]],
        source_labels=[1],
        probability=1.0,
    )
    dataset = attach_source_indices(source).map(
        lambda example: replacer(
            example,
            tf.constant([0, 7], dtype=tf.int64),
        )
    )
    rows = list(dataset.as_numpy_iterator())

    assert len(rows) == 2
    assert int(rows[0]["image"].max()) == 0
    assert int(rows[1]["image"].min()) == 255
    assert [int(row["label"]) for row in rows] == [0, 1]


def test_train_pipeline_runs_saspa_sample_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement_path = tmp_path / "replacement.png"
    _image(200, size=32).save(replacement_path)
    source = tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.constant(
                np.stack(
                    [
                        np.zeros((32, 32, 3), dtype=np.uint8),
                        np.full((32, 32, 3), 10, dtype=np.uint8),
                    ]
                )
            ),
            "label": tf.constant([0, 1], dtype=tf.int64),
        }
    )
    import allthemix.data.pipeline as pipeline
    import allthemix.competitors.saspa.manifest as saspa_manifest

    monkeypatch.setattr(
        pipeline,
        "load_train_dataset",
        lambda **_kwargs: source,
    )
    monkeypatch.setattr(
        saspa_manifest,
        "replacement_catalog",
        lambda **_kwargs: (
            [0, 1],
            [[str(replacement_path)], [str(replacement_path)]],
            [0, 1],
        ),
    )
    dataset = build_train_pipeline(
        name="cifar10",
        data_dir="unused",
        batch_size=2,
        shuffle_buffer_size=2,
        use_basic_augmentation=False,
        augmentation_recipe="none",
        train_manifest_path="unused",
        train_manifest_kind="saspa",
        train_manifest_mode="sample",
        train_manifest_prevalidated=True,
        train_replacement_probability=1.0,
        seed=0,
        deterministic_data=True,
    )
    images, labels = next(iter(dataset))

    assert images.shape == (2, 32, 32, 3)
    assert sorted(labels.numpy().tolist()) == [0, 1]
    np.testing.assert_allclose(images[0].numpy(), images[1].numpy())


def test_saspa_uses_baseline_mixer_and_loss() -> None:
    images = jnp.zeros((2, 4, 4, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    output = get_mixer("saspa", num_classes=2)(
        jax.random.PRNGKey(0),
        images,
        labels,
    )
    logits = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    loss, targets = compute_train_loss_and_targets(
        method="saspa",
        logits=logits,
        mixer_output=output,
        num_classes=2,
    )

    assert float(loss) > 0.0
    np.testing.assert_array_equal(np.asarray(targets), np.asarray(labels))


def test_training_args_require_saspa_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from allthemix.cli.args import parse_args as parse_training_args

    manifest = tmp_path / "manifest.jsonl"
    config = tmp_path / "saspa.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: caltech_birds2011",
                "method: saspa",
                f"saspa_manifest: {manifest.as_posix()}",
                "saspa_replacement_probability: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config)],
    )
    args = parse_training_args()

    assert args.method == "saspa"
    assert args.saspa_manifest == manifest.as_posix()
    assert args.saspa_replacement_probability == pytest.approx(0.1)
