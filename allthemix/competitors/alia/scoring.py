"""Score ALIA sources and edits with an AllTheMix JAX baseline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from allthemix.competitors.alia.manifest import (
    read_stage_records,
    stage_manifest_name,
)
from allthemix.competitors.alia.stages import iter_sources
from allthemix.competitors.generative.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    sha256_file,
)
from allthemix.config import load_yaml_config
from allthemix.data.preprocessors.selector import get_metadata, get_preprocessor
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.train import create_train_state
from allthemix.utils.checkpoint import restore_matching_pretrained_checkpoint


def _checkpoint_identity(path: Path) -> str:
    """Fingerprint checkpoint contents without loading a second copy."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"ALIA baseline checkpoint is missing: {path}")
    catalog = []
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = file_path.stat()
        catalog.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    return config_fingerprint(catalog)


def _resolve_checkpoint(path: str | Path) -> Path:
    """Accept either the exact Orbax item or its run directory."""
    candidate = Path(path).expanduser().resolve()
    best = candidate / "best"
    if best.exists():
        return best
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"ALIA baseline checkpoint is missing: {candidate}")


def _load_baseline_state(
    config_path: Path,
    checkpoint_path: Path,
):
    """Reconstruct and restore the classifier used by ALIA filtering."""
    config = load_yaml_config(config_path)
    method = normalize_method_name(str(config.get("method", "baseline")))
    if method not in {"baseline", "erm"}:
        raise ValueError(
            "ALIA confidence filtering requires a checkpoint trained on the "
            f"original data with method=baseline, not {method!r}."
        )
    dataset = str(config.get("dataset", ""))
    if not dataset:
        raise ValueError("ALIA baseline config must define dataset.")
    metadata = get_metadata(dataset)
    model = build_model(
        name=str(config.get("model", "simple_cnn")),
        num_classes=metadata.num_classes,
        resnet_stem_type=str(config.get("resnet_stem_type", "cifar")),
        preact_stem_bn_relu=bool(config.get("preact_stem_bn_relu", False)),
        preact_pytorch_default_init=bool(
            config.get("preact_pytorch_default_init", False)
        ),
    )
    state = create_train_state(
        rng=jax.random.PRNGKey(int(config.get("seed", 0))),
        model=model,
        learning_rate=0.0,
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        nesterov=bool(config.get("nesterov", False)),
        input_shape=(
            1,
            metadata.image_size,
            metadata.image_size,
            metadata.channels,
        ),
    )
    # Scoring only needs model variables; optimizer trees depend on the
    # training learning-rate schedule and need not match the inference state.
    state, loaded_keys, skipped_keys = restore_matching_pretrained_checkpoint(
        state=state,
        checkpoint_path=str(checkpoint_path),
    )
    loaded_params = [key for key in loaded_keys if key.startswith("params/")]
    loaded_batch_stats = [
        key for key in loaded_keys if key.startswith("batch_stats/")
    ]
    skipped_model_keys = [
        key
        for key in skipped_keys
        if key.startswith(("params/", "batch_stats/"))
    ]
    requires_batch_stats = bool(jax.tree_util.tree_leaves(state.batch_stats))
    if (
        not loaded_params
        or (requires_batch_stats and not loaded_batch_stats)
        or skipped_model_keys
    ):
        raise ValueError(
            "ALIA baseline checkpoint is incompatible with the configured "
            "model: every parameter and required batch-statistics leaf must "
            f"match (skipped={skipped_model_keys})."
        )

    return config, metadata, state


def _make_predictor(
    state,
    distributed: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a single-device JIT or data-parallel inference function."""
    devices = jax.local_devices()
    use_parallel = distributed and len(devices) > 1
    if use_parallel:
        params = jax.device_put_replicated(state.params, devices)
        batch_stats = jax.device_put_replicated(state.batch_stats, devices)

        def apply_one(replica_params, replica_stats, images):
            """Evaluate one equally sized replica shard."""
            return state.apply_fn(
                {
                    "params": replica_params,
                    "batch_stats": replica_stats,
                },
                images,
                training=False,
            )

        parallel_apply = jax.pmap(apply_one)

        def predict(images: np.ndarray) -> np.ndarray:
            """Pad, shard, predict, and remove inference-only padding."""
            count = images.shape[0]
            remainder = count % len(devices)
            if remainder:
                padding = len(devices) - remainder
                images = np.concatenate(
                    [images, np.repeat(images[-1:], padding, axis=0)],
                    axis=0,
                )
            sharded = images.reshape(
                len(devices),
                images.shape[0] // len(devices),
                *images.shape[1:],
            )
            logits = parallel_apply(params, batch_stats, sharded)
            probabilities = jax.nn.softmax(logits, axis=-1)

            return np.asarray(probabilities).reshape(-1, probabilities.shape[-1])[:count]

        return predict

    @jax.jit
    def apply(images):
        """Evaluate one unsharded inference batch."""
        logits = state.apply_fn(
            {
                "params": state.params,
                "batch_stats": state.batch_stats,
            },
            images,
            training=False,
        )

        return jax.nn.softmax(logits, axis=-1)

    def predict(images: np.ndarray) -> np.ndarray:
        """Return host softmax probabilities for one image batch."""
        return np.asarray(apply(jnp.asarray(images)))

    return predict


def _preprocess_image(
    image: Image.Image,
    label: int,
    preprocess,
) -> np.ndarray:
    """Apply the baseline's deterministic evaluation preprocessing."""
    import tensorflow as tf

    image_tensor, _ = preprocess(
        {
            "image": tf.convert_to_tensor(np.asarray(image.convert("RGB"))),
            "label": tf.convert_to_tensor(label, dtype=tf.int64),
        },
        False,
        None,
    )

    return np.asarray(image_tensor, dtype=np.float32)


def _batched(
    values: Iterable[tuple[dict[str, Any], Image.Image]],
    batch_size: int,
) -> Iterator[list[tuple[dict[str, Any], Image.Image]]]:
    """Yield finite image batches while releasing PIL objects promptly."""
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _score_items(
    items: Iterable[tuple[dict[str, Any], Image.Image]],
    preprocess,
    predict: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
    total: int | None,
    stage_name: str,
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """Preprocess and classify a stream without retaining decoded images."""
    completed = 0
    for batch in _batched(items, batch_size=batch_size):
        images = np.stack(
            [
                _preprocess_image(image, int(record["label"]), preprocess)
                for record, image in batch
            ]
        )
        probabilities = predict(images)
        for (record, _image), probability in zip(batch, probabilities):
            yield record, probability
        completed += len(batch)
        suffix = f"/{total}" if total is not None else ""
        print(f"ALIA JAX {stage_name}: {completed}{suffix} images scored")


def _source_items(
    dataset: str,
    data_dir: str,
    validation_split: float,
) -> Iterator[tuple[dict[str, Any], Image.Image]]:
    """Yield baseline train-side source records and detached PIL images."""
    for source in iter_sources(
        dataset=dataset,
        data_dir=data_dir,
        train_dir="",
        validation_split=validation_split,
    ):
        yield (
            {
                "source_id": source.source_id,
                "source_ref": source.source_ref,
                "label": source.label,
                "class_name": source.class_name,
            },
            source.image,
        )


def _generated_items(
    records: Iterable[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], Image.Image]]:
    """Yield generated records with eagerly detached RGB images."""
    for record in records:
        with Image.open(record["resolved_image_path"]) as image:
            yield record, image.convert("RGB").copy()


def score_checkpoint_stage(
    config_path: str | Path,
    checkpoint_path: str | Path,
    artifact_dir: str | Path,
    batch_size: int = 64,
    distributed: bool = True,
    input_stage: str = "clip",
) -> dict[str, Any]:
    """Score original and edited images for official confidence filtering."""
    if batch_size < 1:
        raise ValueError("ALIA scoring batch_size must be positive.")
    config_file = Path(config_path).expanduser().resolve()
    checkpoint = _resolve_checkpoint(checkpoint_path)
    root = Path(artifact_dir).expanduser().resolve()
    config, metadata, state = _load_baseline_state(config_file, checkpoint)
    dataset = str(config["dataset"])
    data_dir = str(config.get("data_dir", "./data"))
    validation_split = float(config.get("validation_split", 0.0))
    generated_records = read_stage_records(
        root,
        stage=input_stage,
        check_images=True,
        require_complete=True,
    )
    for record in generated_records:
        if str(record["dataset"]).lower() != dataset.lower():
            raise ValueError(
                "ALIA generated dataset does not match baseline config: "
                f"{record['dataset']!r} vs {dataset!r}."
            )
        if abs(float(record["validation_split"]) - validation_split) > 1.0e-9:
            raise ValueError(
                "ALIA generated and baseline validation splits differ: "
                f"{record['validation_split']} vs {validation_split}."
            )

    scoring_config = {
        "stage": "classifier",
        "dataset": dataset,
        "validation_split": validation_split,
        "model": config.get("model"),
        "config_sha256": sha256_file(config_file),
        "checkpoint_path": str(checkpoint),
        "checkpoint_identity": _checkpoint_identity(checkpoint),
        "input_stage": input_stage,
    }
    scoring_fingerprint = config_fingerprint(scoring_config)
    atomic_write_json(
        root / "classifier_summary.json",
        {
            "complete": False,
            "stage": "classifier",
            "record_count": 0,
            "scoring_fingerprint": scoring_fingerprint,
        },
    )
    preprocess = get_preprocessor(
        dataset,
        tiny_imagenet_normalization=str(
            config.get("tiny_imagenet_normalization", "imagenet")
        ),
    )
    predict = _make_predictor(state=state, distributed=distributed)

    base_scores = []
    for record, probabilities in _score_items(
        _source_items(dataset, data_dir, validation_split),
        preprocess=preprocess,
        predict=predict,
        batch_size=batch_size,
        total=None,
        stage_name="baseline",
    ):
        label = int(record["label"])
        predicted = int(np.argmax(probabilities))
        base_scores.append(
            {
                **record,
                "label_probability": float(probabilities[label]),
                "predicted_label": predicted,
                "max_probability": float(probabilities[predicted]),
                "scoring_fingerprint": scoring_fingerprint,
            }
        )
    observed_classes = {int(record["label"]) for record in base_scores}
    expected_classes = set(range(metadata.num_classes))
    if observed_classes != expected_classes:
        raise ValueError(
            "ALIA baseline score set does not cover every class: missing "
            f"{sorted(expected_classes - observed_classes)}."
        )

    classifier_records = []
    for record, probabilities in _score_items(
        _generated_items(generated_records),
        preprocess=preprocess,
        predict=predict,
        batch_size=batch_size,
        total=len(generated_records),
        stage_name="generated",
    ):
        label = int(record["label"])
        predicted = int(np.argmax(probabilities))
        updated = {
            key: value
            for key, value in record.items()
            if key not in {"resolved_image_path", "manifest_path"}
        }
        updated.update(
            {
                "stage": "classifier",
                "classifier_predicted_label": predicted,
                "classifier_max_probability": float(probabilities[predicted]),
                "classifier_assigned_label_probability": float(
                    probabilities[label]
                ),
                "classifier_scoring_fingerprint": scoring_fingerprint,
            }
        )
        classifier_records.append(updated)

    atomic_write_jsonl(root / "base_scores.jsonl", base_scores)
    atomic_write_jsonl(
        root / stage_manifest_name("classifier"),
        classifier_records,
    )
    summary = {
        "complete": True,
        "stage": "classifier",
        "record_count": len(classifier_records),
        "base_record_count": len(base_scores),
        "num_classes": metadata.num_classes,
        "distributed": bool(distributed and len(jax.local_devices()) > 1),
        "local_device_count": len(jax.local_devices()),
        "scoring_fingerprint": scoring_fingerprint,
        "scoring_config": scoring_config,
    }
    atomic_write_json(root / "classifier_summary.json", summary)

    return summary
