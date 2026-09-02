"""Tests for the staged ALIA competitor integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from allthemix.cli.args import parse_args as parse_training_args
from allthemix.competitors.alia.ablation import (
    build_paired_ablation_artifacts,
)
from allthemix.competitors.alia.cli import parse_args as parse_alia_args
from allthemix.competitors.alia.filtering import (
    compute_confident_thresholds,
    filter_generated_records,
    strict_filter_generated_records,
)
from allthemix.competitors.alia.manifest import (
    load_manifest_dataset,
    read_stage_records,
    validate_manifest_for_training,
)
from allthemix.competitors.alia.official_artifact import (
    CubTrainSource,
    import_official_cub_artifact,
)
from allthemix.competitors.alia.prompts import (
    CARS196_GENERIC_PROMPTS,
    CUB_PAPER_PROMPTS,
    CUB_RELEASE_PROMPTS,
    STL10_GENERIC_PROMPTS,
    build_prompt_request,
    format_prompt,
    generic_prompt_payload,
    paper_prompt_payload,
    parse_prompt_response,
    release_prompt_payload,
    write_prompt_payload,
)
from allthemix.competitors.alia.scoring import _load_baseline_state
from allthemix.competitors.alia.stages import (
    create_prompt_artifact,
    filter_scored_stage,
    generate_edits,
    publish_strict_filtered_stage,
    score_clip_stage,
)
from allthemix.competitors.alia.visualize import (
    rank_quality_records,
    visualize_filtered_quality,
)
from allthemix.competitors.generative.artifacts import sha256_file
from allthemix.competitors.generative.sources import (
    SourceExample,
    clean_class_name,
    iter_allthemix_sources,
)
from allthemix.data.pipeline import build_train_pipeline
from allthemix.methods.selector import get_mixer
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.train import create_train_state
from allthemix.training.losses.loss_selector import (
    compute_train_loss_and_targets,
)
from allthemix.utils.checkpoint import save_best_checkpoint
from allthemix.training.utils.lr_scheduler import build_cosine_lr_schedule


def _solid_image(value: int, size: int = 16) -> Image.Image:
    """Create one deterministic RGB image fixture."""
    return Image.fromarray(
        np.full((size, size, 3), value, dtype=np.uint8),
        mode="RGB",
    )


def _write_final_manifest(
    root: Path,
    dataset: str = "cifar10",
    validation_split: float = 0.1,
    label: int = 2,
) -> Path:
    """Write one complete checksum-protected ALIA training artifact."""
    image_path = root / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    _solid_image(127, size=32).save(image_path)
    record = {
        "schema_version": 1,
        "method": "alia",
        "stage": "final",
        "record_id": "alia-test-record",
        "image_path": image_path.relative_to(root).as_posix(),
        "output_png_sha256": sha256_file(image_path),
        "dataset": dataset,
        "source_id": f"{dataset}:000000001",
        "source_partition": "train",
        "augmentation_index": 0,
        "label": label,
        "prompt": "a photo of a class bird on rocks.",
        "validation_split": validation_split,
        "config_fingerprint": "generation-fingerprint",
        "filter_status": "keep",
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "final_summary.json").write_text(
        json.dumps({"complete": True, "kept_records": 1}),
        encoding="utf-8",
    )

    return manifest


def _write_paired_final_manifest(root: Path) -> Path:
    """Write two accepted generated images with distinct source IDs."""
    records = []
    for index, (label, value) in enumerate(((2, 96), (7, 192))):
        image_path = root / "images" / f"generated-{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _solid_image(value).save(image_path)
        records.append(
            {
                "schema_version": 1,
                "method": "alia",
                "stage": "final",
                "record_id": f"alia-generated-{index}",
                "image_path": image_path.relative_to(root).as_posix(),
                "output_png_sha256": sha256_file(image_path),
                "dataset": "caltech_birds2011",
                "source_id": f"caltech_birds2011:{index:09d}",
                "source_partition": "train",
                "augmentation_index": 0,
                "label": label,
                "prompt": "a photo of a bird on rocks.",
                "validation_split": 0.1,
                "config_fingerprint": "paired-generation-fingerprint",
                "filter_status": "keep",
                "semantic_pass": True,
                "clip_positive_probability": 0.9 - index * 0.1,
                "classifier_predicted_label": label if index == 0 else 3,
                "classifier_max_probability": 0.4 - index * 0.1,
                "classifier_assigned_label_probability": 0.4 - index * 0.3,
                "class_confident_threshold": 0.5,
            }
        )

    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "final_summary.json").write_text(
        json.dumps({"complete": True, "kept_records": len(records)}),
        encoding="utf-8",
    )

    return manifest


def test_cub_paper_prompts_and_class_name_formatting() -> None:
    """Keep the seven CUB prompts reported in the ALIA paper immutable."""
    assert len(CUB_PAPER_PROMPTS) == 7
    assert format_prompt(CUB_PAPER_PROMPTS[0], "001.Black_footed_Albatross") == (
        "a photo of a Black footed Albatross bird interacting with flowers."
    )


def test_cub_release_prompts_match_official_config() -> None:
    """Keep the seven generation roots named by the GitHub release stable."""
    assert CUB_RELEASE_PROMPTS == (
        "a photo of a {class_name} bird flying.",
        "a photo of a {class_name} bird interacting with flowers.",
        "a photo of a {class_name} bird in the water.",
        "a photo of a {class_name} bird on a branch.",
        "a photo of a {class_name} bird on rocks.",
        "a photo of a {class_name} bird perched on a birdfeeder.",
        "a photo of a {class_name} bird perched on a fence.",
    )
    payload = release_prompt_payload("caltech_birds2011")
    assert payload["prompt_source"] == "official-github-release"
    assert clean_class_name("042.Geococcyx_californianus") == (
        "Roadrunner californianus"
    )


def test_stl10_generic_prompt_artifact_is_dataset_aware(
    tmp_path: Path,
) -> None:
    """Keep the non-paper STL extension explicit and reproducible."""
    payload = generic_prompt_payload("stl10")

    assert payload["prompt_source"] == "allthemix-generic-stl10-v1"
    assert payload["prompts"] == list(STL10_GENERIC_PROMPTS)
    assert payload["semantic_prompts"] == [
        "a photo of an animal",
        "a photo of a vehicle",
    ]
    assert format_prompt(STL10_GENERIC_PROMPTS[0], "airplane") == (
        "a photo showing airplane outdoors during daytime."
    )

    output = tmp_path / "prompts.json"
    written = create_prompt_artifact(
        dataset="stl10",
        output_path=output,
        mode="generic",
    )
    assert written == payload
    assert output.is_file()

    args = parse_alia_args(
        [
            "prompts",
            "--dataset",
            "stl10",
            "--mode",
            "generic",
            "--output",
            str(output),
        ]
    )
    assert args.mode == "generic"


def test_cars196_generic_prompt_artifact_uses_vehicle_semantics(
    tmp_path: Path,
) -> None:
    """Keep the non-paper Cars196 extension explicit and class-aware."""
    payload = generic_prompt_payload("cars196")

    assert payload["prompt_source"] == "allthemix-generic-cars196-v1"
    assert payload["prompts"] == list(CARS196_GENERIC_PROMPTS)
    assert payload["semantic_prompts"] == [
        "a photo of a car",
        "a photo of an automobile",
        "a photo of a road vehicle",
    ]
    assert format_prompt(
        CARS196_GENERIC_PROMPTS[0],
        "001.Audi_R8_Coupe_2012",
    ) == "a photo of the Audi R8 Coupe 2012 parked outdoors."

    output = tmp_path / "cars196-prompts.json"
    written = create_prompt_artifact(
        dataset="cars196",
        output_path=output,
        mode="generic",
    )
    assert written == payload
    assert output.is_file()


def test_cars196_generation_sources_preserve_folder_class_names(
    tmp_path: Path,
) -> None:
    """Use model names, stable labels, and the shared stratified split."""
    root = tmp_path / "cars196"
    class_names = (
        "001.Audi_R8_Coupe_2012",
        "002.BMW_M3_Coupe_2012",
    )
    for class_name in class_names:
        class_dir = root / "train" / class_name
        class_dir.mkdir(parents=True)
        for index in range(10):
            _solid_image(index).save(class_dir / f"{index:03d}.jpg")
    (root / "test").mkdir()

    sources = list(
        iter_allthemix_sources(
            dataset="cars196",
            data_dir=str(tmp_path),
            validation_split=0.1,
            download=False,
        )
    )

    assert len(sources) == 18
    assert {source.label for source in sources} == {0, 1}
    assert {source.class_name for source in sources} == set(class_names)
    assert all(source.source_id.startswith("class-folder:") for source in sources)
    assert all(not source.source_ref.endswith("000.jpg") for source in sources)


def test_build_paired_alia_quality_ablation(tmp_path: Path) -> None:
    """Match every accepted edit to the exact labeled source image."""
    source_artifact = tmp_path / "filtered"
    _write_paired_final_manifest(source_artifact)
    sources = [
        SourceExample(
            index=index,
            source_id=f"caltech_birds2011:{index:09d}",
            source_ref=f"train:{index}",
            label=label,
            class_name=str(label),
            image=_solid_image(value),
        )
        for index, (label, value) in enumerate(((2, 32), (7, 224)))
    ]

    output = tmp_path / "ablation"
    summary = build_paired_ablation_artifacts(
        artifact_dir=source_artifact,
        output_dir=output,
        dataset="caltech_birds2011",
        data_dir="unused",
        validation_split=0.1,
        max_records=2,
        seed=0,
        sources=sources,
    )

    assert summary["pair_count"] == 2
    for subset in ("generated", "original"):
        assert validate_manifest_for_training(
            output / subset,
            dataset="caltech_birds2011",
            num_classes=200,
            validation_split=0.1,
        ) == 2

    generated = read_stage_records(output / "generated")
    originals = read_stage_records(output / "original")
    assert [record["source_id"] for record in generated] == [
        record["source_id"] for record in originals
    ]
    assert [record["label"] for record in generated] == [2, 7]
    assert [record["label"] for record in originals] == [2, 7]
    pairs = [
        json.loads(line)
        for line in (output / "pairs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [pair["source_id"] for pair in pairs] == [
        "caltech_birds2011:000000000",
        "caltech_birds2011:000000001",
    ]


def test_alia_cli_parses_paired_ablation() -> None:
    """Expose the paired quality check as a reproducible CLI stage."""
    args = parse_alia_args(
        [
            "paired-ablation",
            "--artifact-dir",
            "data/alia/cub200",
            "--output-dir",
            "outputs/paired",
            "--dataset",
            "caltech_birds2011",
            "--max-records",
            "1000",
            "--seed",
            "3",
        ]
    )

    assert args.command == "paired-ablation"
    assert args.max_records == 1000
    assert args.seed == 3


def test_visualize_ranked_alia_quality(tmp_path: Path) -> None:
    """Render exact generated/source pairs and preserve ranking metadata."""
    source_artifact = tmp_path / "filtered"
    _write_paired_final_manifest(source_artifact)
    sources = [
        SourceExample(
            index=index,
            source_id=f"caltech_birds2011:{index:09d}",
            source_ref=f"train:{index}",
            label=label,
            class_name=f"bird-{label}",
            image=_solid_image(value),
        )
        for index, (label, value) in enumerate(((2, 32), (7, 224)))
    ]
    records = read_stage_records(source_artifact)
    ranked = rank_quality_records(
        records,
        ranking="best",
        max_per_class=1,
        num_samples=2,
    )
    assert [record["record_id"] for record in ranked] == [
        "alia-generated-0",
        "alia-generated-1",
    ]

    output = tmp_path / "quality" / "best.png"
    summary = visualize_filtered_quality(
        artifact_dir=source_artifact,
        output_path=output,
        dataset="caltech_birds2011",
        data_dir="unused",
        validation_split=0.1,
        num_samples=2,
        pairs_per_row=2,
        tile_size=96,
        sources=sources,
    )

    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    assert output.with_suffix(".json").is_file()
    assert summary["num_samples"] == 2
    assert summary["label_agreement_count"] == 1
    with Image.open(output) as contact_sheet:
        assert contact_sheet.width == 384
        assert contact_sheet.height > 96


def test_alia_cli_parses_quality_visualization() -> None:
    """Expose ranked visual auditing without loading model dependencies."""
    args = parse_alia_args(
        [
            "visualize",
            "--artifact-dir",
            "data/alia/cub200",
            "--output",
            "outputs/visualize/alia_best.png",
            "--dataset",
            "caltech_birds2011",
            "--num-samples",
            "12",
        ]
    )

    assert args.command == "visualize"
    assert args.ranking == "best"
    assert args.num_samples == 12


def test_prompt_response_parser_builds_class_aware_templates() -> None:
    """Convert an external LLM response into validated prompt templates."""
    prompts = parse_prompt_response(
        "1. a photo of a {class_name} bird in snow\n"
        "- a photo of a {class_name} bird near water.",
        prefix="a photo of a {class_name} bird",
    )

    assert prompts == (
        "a photo of a {class_name} bird in snow.",
        "a photo of a {class_name} bird near water.",
    )


def test_prompt_request_samples_captions_reproducibly() -> None:
    """Use a seeded random sample like the official prompt discovery step."""
    captions = [f"caption {index}" for index in range(30)]

    first = build_prompt_request(captions, prefix="a bird", seed=7)
    second = build_prompt_request(captions, prefix="a bird", seed=7)
    different = build_prompt_request(captions, prefix="a bird", seed=8)

    assert first == second
    assert first != different
    assert sum(line.startswith("- caption ") for line in first.splitlines()) == 20


def test_confident_thresholds_match_per_assigned_class_average() -> None:
    """Compute Cleanlab-style self-confidence means for every class."""
    thresholds = compute_confident_thresholds(
        [
            {"label": 0, "label_probability": 0.6},
            {"label": 0, "label_probability": 0.8},
            {"label": 1, "label_probability": 0.3},
            {"label": 1, "label_probability": 0.5},
        ],
        num_classes=2,
    )

    np.testing.assert_allclose(thresholds, [0.7, 0.4])


def test_alia_baseline_restore_ignores_optimizer_schedule_state(
    tmp_path: Path,
) -> None:
    """Restore classifier leaves when training and scoring optimizers differ."""
    model = build_model(
        name="simple_cnn",
        num_classes=10,
    )
    schedule = build_cosine_lr_schedule(
        base_learning_rate=0.1,
        min_learning_rate=0.0,
        total_steps=8,
    )
    trained_state = create_train_state(
        rng=jax.random.PRNGKey(3),
        model=model,
        learning_rate=schedule,
        momentum=0.9,
        weight_decay=0.0005,
        input_shape=(1, 32, 32, 3),
    )
    trained_state = trained_state.replace(
        params=jax.tree_util.tree_map(
            lambda value: value + 0.125,
            trained_state.params,
        ),
        batch_stats=jax.tree_util.tree_map(
            lambda value: value + 0.25,
            trained_state.batch_stats,
        ),
    )
    save_best_checkpoint(
        state=trained_state,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: cifar10",
                "model: simple_cnn",
                "method: baseline",
                "seed: 17",
                "momentum: 0.9",
                "weight_decay: 0.0005",
            ]
        ),
        encoding="utf-8",
    )

    _, metadata, restored_state = _load_baseline_state(
        config_path=config_path,
        checkpoint_path=tmp_path / "checkpoints" / "best",
    )

    assert metadata.num_classes == 10
    assert (
        jax.tree_util.tree_structure(restored_state.opt_state)
        != jax.tree_util.tree_structure(trained_state.opt_state)
    )
    for expected, actual in zip(
        jax.tree_util.tree_leaves(trained_state.params),
        jax.tree_util.tree_leaves(restored_state.params),
    ):
        np.testing.assert_allclose(actual, expected)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(trained_state.batch_stats),
        jax.tree_util.tree_leaves(restored_state.batch_stats),
    ):
        np.testing.assert_allclose(actual, expected)


def test_filter_matches_official_code_assigned_label_threshold() -> None:
    """Preserve the released code behavior where paper notation differs."""
    base_scores = [
        {"label": 0, "label_probability": 0.8},
        {"label": 1, "label_probability": 0.4},
    ]
    records = [
        {
            "record_id": "keep-under-assigned-threshold",
            "source_id": "source-0",
            "label": 0,
            "semantic_pass": True,
            "classifier_predicted_label": 1,
            "classifier_max_probability": 0.7,
        },
        {
            "record_id": "too-easy",
            "source_id": "source-1",
            "label": 0,
            "semantic_pass": True,
            "classifier_predicted_label": 0,
            "classifier_max_probability": 0.9,
        },
        {
            "record_id": "mislabeled",
            "source_id": "source-2",
            "label": 1,
            "semantic_pass": True,
            "classifier_predicted_label": 0,
            "classifier_max_probability": 0.5,
        },
        {
            "record_id": "semantic-reject",
            "source_id": "source-3",
            "label": 1,
            "semantic_pass": False,
            "classifier_predicted_label": 1,
            "classifier_max_probability": 0.1,
        },
    ]

    accepted, summary = filter_generated_records(
        records,
        base_scores=base_scores,
        num_classes=2,
        extra_ratio=-1.0,
    )

    assert [record["record_id"] for record in accepted] == [
        "keep-under-assigned-threshold"
    ]
    assert summary["rejected_too_easy"] == 1
    assert summary["rejected_mislabeled"] == 1
    assert summary["rejected_semantic"] == 1


def test_final_manifest_validation_and_dataset_loading(tmp_path: Path) -> None:
    """Reject provenance drift and decode accepted generated images."""
    manifest = _write_final_manifest(tmp_path)

    assert validate_manifest_for_training(
        manifest,
        dataset="cifar10",
        num_classes=10,
        validation_split=0.1,
    ) == 1
    assert validate_manifest_for_training(
        manifest,
        dataset="cifar10",
        num_classes=10,
        validation_split=None,
    ) == 1
    example = next(iter(load_manifest_dataset(manifest, image_size=32)))
    assert tuple(example["image"].shape) == (32, 32, 3)
    assert int(example["label"].numpy()) == 2

    with pytest.raises(ValueError, match="generation-source split mismatch"):
        validate_manifest_for_training(
            manifest,
            dataset="cifar10",
            num_classes=10,
            validation_split=0.2,
        )


def test_pipeline_selects_alia_manifest_loader(tmp_path: Path) -> None:
    """Feed a final ALIA artifact through the shared classifier pipeline."""
    manifest = _write_final_manifest(tmp_path)
    dataset = build_train_pipeline(
        name="cifar10",
        data_dir=str(tmp_path),
        batch_size=1,
        shuffle_buffer_size=1,
        drop_remainder=True,
        train_manifest_path=str(manifest),
        train_manifest_kind="alia",
        train_manifest_mode="replace",
        train_manifest_prevalidated=True,
    )
    images, labels = next(iter(dataset))

    assert tuple(images.shape) == (1, 32, 32, 3)
    assert labels.numpy().tolist() == [2]


def test_alia_append_projects_cub_examples_to_common_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append generated CUB images to raw TFDS records with extra metadata."""
    manifest = _write_final_manifest(
        tmp_path,
        dataset="caltech_birds2011",
        validation_split=0.0,
        label=2,
    )

    def original_examples():
        """Yield the variable-size structure exposed by CUB in TFDS."""
        yield {
            "bbox": np.zeros((4,), dtype=np.float32),
            "image": np.full((240, 320, 3), 64, dtype=np.uint8),
            "image/filename": b"cub-example.jpg",
            "label": np.int64(1),
            "label_name": b"bird",
            "segmentation_mask": np.zeros((240, 320, 1), dtype=np.uint8),
        }

    original_ds = tf.data.Dataset.from_generator(
        original_examples,
        output_signature={
            "bbox": tf.TensorSpec((4,), tf.float32),
            "image": tf.TensorSpec((None, None, 3), tf.uint8),
            "image/filename": tf.TensorSpec((), tf.string),
            "label": tf.TensorSpec((), tf.int64),
            "label_name": tf.TensorSpec((), tf.string),
            "segmentation_mask": tf.TensorSpec((None, None, 1), tf.uint8),
        },
    )
    monkeypatch.setattr(
        "allthemix.data.pipeline.load_train_dataset",
        lambda **_kwargs: original_ds,
    )

    dataset = build_train_pipeline(
        name="caltech_birds2011",
        data_dir=str(tmp_path),
        batch_size=2,
        shuffle_buffer_size=2,
        drop_remainder=True,
        use_basic_augmentation=False,
        validation_split=0.0,
        train_manifest_path=str(manifest),
        train_manifest_kind="alia",
        train_manifest_mode="append",
        train_original_example_count=1,
        train_manifest_example_count=1,
        train_manifest_prevalidated=True,
        seed=0,
        deterministic_data=True,
    )
    images, labels = next(iter(dataset))

    assert tuple(images.shape) == (2, 224, 224, 3)
    assert sorted(labels.numpy().tolist()) == [1, 2]
    assert np.isfinite(images.numpy()).all()


def test_generate_stage_is_resumable_and_checksum_protected(
    tmp_path: Path,
) -> None:
    """Generate paper-prompt jobs once and resume every completed image."""
    prompt_path = tmp_path / "prompts.json"
    write_prompt_payload(
        prompt_path,
        paper_prompt_payload("caltech_birds2011"),
    )

    class FakeEditor:
        """Small deterministic substitute for Stable Diffusion."""

        config = SimpleNamespace(
            model_id="fake-editor",
            strength=0.6,
            guidance_scale=7.5,
            num_inference_steps=2,
        )
        resolved_model_commit = "fake-commit"
        runtime = SimpleNamespace(device_kind="cpu")

        def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
            """Encode the stable job seed into a solid image."""
            del image, prompt
            return _solid_image(seed % 255, size=16)

        def synchronize(self) -> None:
            """Match the real editor synchronization interface."""

    source = SourceExample(
        index=0,
        source_id="cub:0",
        source_ref="cub:train:0",
        label=0,
        class_name="001.Black_footed_Albatross",
        image=_solid_image(64, size=16),
    )
    kwargs = {
        "editor": FakeEditor(),
        "sources": [source],
        "prompt_path": prompt_path,
        "output_dir": tmp_path,
        "dataset": "caltech_birds2011",
        "validation_split": 0.1,
        "images_per_prompt": 1,
        "generation_size": 16,
        "seed": 7,
    }
    first = generate_edits(**kwargs)
    second = generate_edits(**kwargs)
    records = read_stage_records(tmp_path, stage="generated")

    assert first["generated"] == 7
    assert second["generated"] == 0
    assert second["resumed"] == 7
    assert len(records) == 7


def test_clip_stage_durably_resumes_completed_batches(tmp_path: Path) -> None:
    """Resume CLIP scoring after a failure without rescoring prior batches."""
    generated_records = []
    for index in range(3):
        image_path = tmp_path / "images" / f"generated-{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _solid_image(64 + index).save(image_path)
        generated_records.append(
            {
                "schema_version": 1,
                "method": "alia",
                "stage": "generated",
                "record_id": f"alia-generated-{index}",
                "image_path": image_path.relative_to(tmp_path).as_posix(),
                "output_png_sha256": sha256_file(image_path),
                "dataset": "caltech_birds2011",
                "source_id": f"cub:{index}",
                "source_partition": "train",
                "augmentation_index": 0,
                "label": index,
                "prompt": "a photo of a bird",
                "validation_split": 0.1,
                "config_fingerprint": "generation-fingerprint",
            }
        )
    (tmp_path / "generated.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in generated_records),
        encoding="utf-8",
    )
    (tmp_path / "generated_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "stage": "generated",
                "record_count": len(generated_records),
            }
        ),
        encoding="utf-8",
    )

    class FakeClipScorer:
        """Deterministic scorer that can fail after a completed batch."""

        config = SimpleNamespace(
            model_id="fake-clip",
            model_revision="",
            logit_scale=100.0,
        )
        resolved_model_commit = "fake-commit"
        positive_prompts = ("a photo of a bird",)
        negative_prompts = ("an image",)

        def __init__(self, fail_after: int | None = None) -> None:
            self.fail_after = fail_after
            self.calls = 0

        def score(self, images: list[Image.Image]) -> list[dict[str, object]]:
            """Return one semantic decision per image or simulate failure."""
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("simulated interruption")
            return [
                {
                    "semantic_pass": True,
                    "clip_prediction_index": 0,
                    "clip_prediction_text": "a photo of a bird",
                    "clip_positive_probability": 0.9,
                }
                for _image in images
            ]

        def synchronize(self) -> None:
            """Match the production scorer interface."""

    interrupted = FakeClipScorer(fail_after=1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        score_clip_stage(interrupted, tmp_path, batch_size=1)

    partial = (tmp_path / ".clip.inprogress.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    partial_summary = json.loads(
        (tmp_path / "clip_summary.json").read_text(encoding="utf-8")
    )
    assert len(partial) == 1
    assert partial_summary["complete"] is False
    assert partial_summary["record_count"] == 1

    resumed = FakeClipScorer()
    result = score_clip_stage(resumed, tmp_path, batch_size=1)
    records = read_stage_records(
        tmp_path,
        stage="clip",
        require_complete=True,
    )

    assert resumed.calls == 2
    assert result == {
        "records": 3,
        "semantic_pass": 3,
        "semantic_reject": 0,
    }
    assert len(records) == 3
    assert not (tmp_path / ".clip.inprogress.jsonl").exists()


def test_filter_stage_publishes_training_artifact(tmp_path: Path) -> None:
    """Combine semantic and classifier scores into a final manifest."""
    image_path = tmp_path / "images" / "generated.png"
    image_path.parent.mkdir(parents=True)
    _solid_image(100).save(image_path)
    record = {
        "schema_version": 1,
        "method": "alia",
        "stage": "classifier",
        "record_id": "alia-generated",
        "image_path": image_path.relative_to(tmp_path).as_posix(),
        "output_png_sha256": sha256_file(image_path),
        "dataset": "caltech_birds2011",
        "source_id": "cub:0",
        "source_partition": "train",
        "augmentation_index": 0,
        "label": 0,
        "prompt": "bird near water",
        "validation_split": 0.1,
        "config_fingerprint": "generation-fingerprint",
        "semantic_pass": True,
        "classifier_predicted_label": 1,
        "classifier_max_probability": 0.2,
    }
    (tmp_path / "classifier.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "classifier_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "stage": "classifier",
                "record_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "base_scores.jsonl").write_text(
        json.dumps({"label": 0, "label_probability": 0.8}) + "\n"
        + json.dumps({"label": 1, "label_probability": 0.7})
        + "\n",
        encoding="utf-8",
    )

    summary = filter_scored_stage(
        artifact_dir=tmp_path,
        num_classes=2,
        extra_ratio=-1.0,
    )

    assert summary["kept_records"] == 1
    assert validate_manifest_for_training(
        tmp_path,
        dataset="caltech_birds2011",
        num_classes=2,
        validation_split=0.1,
    ) == 1


def test_strict_filter_rejects_mismatches_and_ranks_quality() -> None:
    """Prefer class-preserving edits without filling weak classes."""
    base_scores = [
        {"label": 0, "label_probability": 0.8},
        {"label": 1, "label_probability": 0.7},
    ]

    def record(
        record_id: str,
        label: int,
        predicted: int,
        assigned_probability: float,
        source_id: str,
        semantic_pass: bool = True,
    ) -> dict[str, object]:
        return {
            "record_id": record_id,
            "label": label,
            "classifier_predicted_label": predicted,
            "classifier_assigned_label_probability": assigned_probability,
            "classifier_max_probability": max(assigned_probability, 0.9),
            "clip_positive_probability": 0.9,
            "semantic_pass": semantic_pass,
            "source_id": source_id,
        }

    records = [
        record("same-source-weaker", 0, 0, 0.6, "source-a"),
        record("same-source-best", 0, 0, 0.7, "source-a"),
        record("other-source", 0, 0, 0.65, "source-b"),
        record("mismatch", 1, 0, 0.01, "source-c"),
        record("too-low", 1, 1, 0.1, "source-d"),
        record("too-easy", 1, 1, 0.8, "source-e"),
        record("class-one-good", 1, 1, 0.5, "source-f"),
        record("semantic-fail", 1, 1, 0.6, "source-g", False),
    ]

    accepted, summary = strict_filter_generated_records(
        records=records,
        base_scores=base_scores,
        num_classes=2,
        min_assigned_probability=0.2,
        per_class=1,
        max_per_source=1,
    )

    assert {record["record_id"] for record in accepted} == {
        "same-source-best",
        "class-one-good",
    }
    assert summary["rejected_label_mismatch"] == 1
    assert summary["rejected_low_assigned_probability"] == 1
    assert summary["rejected_too_easy"] == 1
    assert summary["rejected_semantic"] == 1
    assert summary["removed_by_source_cap"] == 1
    assert summary["removed_by_class_cap"] == 1
    assert summary["classes_covered"] == 2


def test_strict_filter_applies_exact_balanced_total_budget() -> None:
    """Match artifact comparisons without favoring one populous class."""
    base_scores = [
        {"label": label, "label_probability": 0.95}
        for label in range(3)
    ]
    records = [
        {
            "record_id": f"record-{label}-{index}",
            "source_id": f"source-{label}-{index}",
            "label": label,
            "semantic_pass": True,
            "classifier_predicted_label": label,
            "classifier_assigned_label_probability": 0.8 - index * 0.01,
            "classifier_max_probability": 0.8 - index * 0.01,
            "clip_positive_probability": 0.9,
        }
        for label in range(3)
        for index in range(5)
    ]

    accepted, summary = strict_filter_generated_records(
        records=records,
        base_scores=base_scores,
        num_classes=3,
        per_class=5,
        max_records=8,
        seed=7,
        exclude_too_easy=False,
    )
    class_counts = {
        label: sum(int(record["label"]) == label for record in accepted)
        for label in range(3)
    }

    assert len(accepted) == 8
    assert max(class_counts.values()) - min(class_counts.values()) == 1
    assert summary["removed_by_total_cap"] == 7
    assert summary["max_records"] == 8
    assert summary["seed"] == 7


def test_strict_filter_publishes_separate_valid_artifact(tmp_path: Path) -> None:
    """Reference existing PNGs without overwriting the official manifest."""
    source_root = tmp_path / "source"
    image_path = source_root / "images" / "generated.png"
    image_path.parent.mkdir(parents=True)
    _solid_image(100).save(image_path)
    record = {
        "schema_version": 1,
        "method": "alia",
        "stage": "classifier",
        "record_id": "alia-generated",
        "image_path": image_path.relative_to(source_root).as_posix(),
        "output_png_sha256": sha256_file(image_path),
        "dataset": "caltech_birds2011",
        "source_id": "cub:0",
        "source_partition": "train",
        "augmentation_index": 0,
        "label": 0,
        "prompt": "bird near water",
        "validation_split": 0.1,
        "config_fingerprint": "generation-fingerprint",
        "semantic_pass": True,
        "clip_positive_probability": 0.95,
        "classifier_predicted_label": 0,
        "classifier_max_probability": 0.6,
        "classifier_assigned_label_probability": 0.6,
    }
    (source_root / "classifier.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    (source_root / "classifier_summary.json").write_text(
        json.dumps(
            {"complete": True, "stage": "classifier", "record_count": 1}
        ),
        encoding="utf-8",
    )
    (source_root / "base_scores.jsonl").write_text(
        json.dumps({"label": 0, "label_probability": 0.8}) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "strict"
    summary = publish_strict_filtered_stage(
        artifact_dir=source_root,
        output_dir=output,
        num_classes=1,
        min_assigned_probability=0.2,
        per_class=5,
    )

    assert summary["strict_filter"] is True
    assert summary["kept_records"] == 1
    assert not (source_root / "manifest.jsonl").exists()
    assert validate_manifest_for_training(
        output,
        dataset="caltech_birds2011",
        num_classes=1,
        validation_split=0.1,
    ) == 1
    strict_record = read_stage_records(output)[0]
    assert Path(strict_record["resolved_image_path"]) == image_path.resolve()


def test_alia_cli_parses_strict_filter() -> None:
    """Keep strict filtering explicitly separate from official filtering."""
    args = parse_alia_args(
        [
            "strict-filter",
            "--artifact-dir",
            "data/alia/cub200",
            "--output-dir",
            "data/alia/cub200_strict",
            "--num-classes",
            "200",
            "--min-assigned-probability",
            "0.2",
            "--per-class",
            "5",
        ]
    )

    assert args.command == "strict-filter"
    assert args.min_assigned_probability == 0.2
    assert args.per_class == 5
    assert args.max_records == -1
    assert args.exclude_too_easy is True


def test_alia_cli_parses_official_artifact_import() -> None:
    """Expose a separate, leakage-safe official W&B import stage."""
    args = parse_alia_args(
        [
            "import-official",
            "--artifact-dir",
            "downloads/cub_generic",
            "--output-dir",
            "data/alia/cub_official",
        ]
    )

    assert args.command == "import-official"
    assert args.dataset == "caltech_birds2011"
    assert args.validation_split == pytest.approx(0.1)
    assert args.artifact_ref == "clipinvariance/ALIA/cub_generic:v0"
    assert args.overwrite is False


def test_import_official_cub_artifact_maps_sources_and_excludes_val(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map released source indices and remove generated validation sources."""
    artifact = tmp_path / "download"
    sources = []
    for label in range(200):
        class_name = (
            "105.Whip_poor_Will"
            if label == 104
            else f"{label + 1:03d}.Bird_{label}"
        )
        class_dir = artifact / class_name
        class_dir.mkdir(parents=True)

        # The released newCub2011 class keeps the first two after dropping
        # the final 15. class_index=0 belongs to the 10% validation split;
        # class_index=1 is a legal generated training source.
        for class_index in range(17):
            if class_index == 0:
                filename = f"{class_name}/a.jpg"
            elif class_index == 1:
                filename = f"{class_name}/b.jpg"
            else:
                filename = f"{class_name}/z_{class_index:02d}.jpg"
            sources.append(
                CubTrainSource(
                    global_index=label * 17 + class_index,
                    label=label,
                    class_index=class_index,
                    filename=filename,
                    class_name="whip_poor_will" if label == 104 else str(label),
                    is_validation=class_index == 0,
                )
            )

        for source_index in (label * 2, label * 2 + 1):
            _solid_image((label + source_index) % 255).save(
                class_dir / f"{source_index}-0.png"
            )
        if label == 0:
            preview_dir = class_dir / "samples"
            preview_dir.mkdir()
            _solid_image(127).save(preview_dir / "0.png")

    monkeypatch.setattr(
        "allthemix.competitors.alia.official_artifact._load_cub_train_sources",
        lambda **_kwargs: sources,
    )
    output = tmp_path / "imported"
    summary = import_official_cub_artifact(
        artifact_dir=artifact,
        output_dir=output,
        data_dir=str(tmp_path),
        validation_split=0.1,
    )
    records = read_stage_records(
        output,
        stage="generated",
        require_complete=True,
    )

    assert summary["source_layout"] == "newCub2011_train_minus_15"
    assert summary["artifact_records"] == 400
    assert summary["ignored_preview_pngs"] == 1
    assert summary["excluded_validation_records"] == 200
    assert summary["excluded_validation_sources"] == 200
    assert summary["record_count"] == 200
    assert len(records) == 200
    assert {record["label"] for record in records} == set(range(200))
    assert all(int(record["official_source_index"]) % 2 == 1 for record in records)
    assert {
        record["source_id"] for record in records
    } == {
        f"caltech_birds2011:{label * 17 + 1:09d}"
        for label in range(200)
    }


def test_alia_uses_identity_mixer_and_hard_label_loss() -> None:
    """Train accepted ALIA files as ordinary labeled examples."""
    images = jnp.ones((2, 4, 4, 3), dtype=jnp.float32)
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    mixer = get_mixer(name="alia", num_classes=2)
    output = mixer(
        rng=jax.random.PRNGKey(0),
        images=images,
        labels=labels,
        aux_info={},
    )
    logits = jnp.asarray([[3.0, -1.0], [-1.0, 3.0]])
    loss, targets = compute_train_loss_and_targets(
        method="alia",
        logits=logits,
        mixer_output=output,
        num_classes=2,
    )

    assert jnp.array_equal(output[0], images)
    assert jnp.array_equal(targets, labels)
    assert float(loss) < 0.1


def test_training_cli_requires_and_accepts_alia_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an explicit filtered artifact for method=alia."""
    monkeypatch.setattr("sys.argv", ["train", "--method", "alia"])
    with pytest.raises(ValueError, match="requires alia_manifest"):
        parse_training_args()

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--method",
            "alia",
            "--alia_manifest",
            "artifact",
            "--alia_train_mode",
            "append",
        ],
    )
    args = parse_training_args()

    assert args.alia_manifest == "artifact"
    assert args.alia_train_mode == "append"


def test_alia_cli_parses_official_cub_edit_defaults() -> None:
    """Expose official CUB Stable Diffusion settings through the CLI."""
    args = parse_alia_args(
        [
            "edit",
            "--dataset",
            "caltech_birds2011",
            "--prompts",
            "prompts.json",
            "--output-dir",
            "artifact",
        ]
    )

    assert args.strength == 0.6
    assert args.guidance_scale == 7.5
    assert args.images_per_prompt == 2
    assert args.num_inference_steps == 50

    prompt_args = parse_alia_args(
        [
            "prompts",
            "--dataset",
            "caltech_birds2011",
            "--output",
            "prompts.json",
        ]
    )
    assert prompt_args.mode == "release"


def test_filter_keeps_at_most_one_edit_per_source() -> None:
    """Mirror the official Img2ImgDataset one-variant source sampling."""
    base_scores = [
        {"label": 0, "label_probability": 0.8},
        {"label": 1, "label_probability": 0.8},
    ]
    records = [
        {
            "record_id": f"record-{index}",
            "source_id": "shared-source" if index < 3 else "other-source",
            "label": 0,
            "semantic_pass": True,
            "classifier_predicted_label": 1,
            "classifier_max_probability": 0.1,
        }
        for index in range(4)
    ]

    accepted, summary = filter_generated_records(
        records,
        base_scores=base_scores,
        num_classes=2,
        extra_ratio=-1.0,
        max_per_source=1,
    )

    assert len(accepted) == 2
    assert len({record["source_id"] for record in accepted}) == 2
    assert summary["removed_by_source_cap"] == 2


def test_incomplete_generated_stage_is_rejected(tmp_path: Path) -> None:
    """Do not let a crashed generation shard flow into later stages."""
    prompt_path = tmp_path / "prompts.json"
    write_prompt_payload(
        prompt_path,
        paper_prompt_payload("caltech_birds2011"),
    )

    class FakeEditor:
        """Generate a single deterministic fixture image."""

        config = SimpleNamespace(
            model_id="fake-editor",
            strength=0.6,
            guidance_scale=7.5,
            num_inference_steps=2,
        )
        resolved_model_commit = "fake-commit"
        runtime = SimpleNamespace(device_kind="cpu")

        def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
            """Return one detached generated image."""
            del image, prompt, seed
            return _solid_image(100)

        def synchronize(self) -> None:
            """Match the production editor interface."""

    source = SourceExample(
        index=0,
        source_id="cub:0",
        source_ref="cub:train:0",
        label=0,
        class_name="001.Black_footed_Albatross",
        image=_solid_image(64),
    )
    generate_edits(
        editor=FakeEditor(),
        sources=[source],
        prompt_path=prompt_path,
        output_dir=tmp_path,
        dataset="caltech_birds2011",
        validation_split=0.1,
        images_per_prompt=1,
        generation_size=16,
    )
    summary_path = tmp_path / "generated_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["complete"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="summary is incomplete"):
        read_stage_records(
            tmp_path,
            stage="generated",
            require_complete=True,
        )
