"""Official SaSPA semantic and classifier top-k filtering stages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from allthemix.competitors.generative.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    sha256_file,
)
from allthemix.competitors.saspa.manifest import read_stage_records


def _checkpoint_identity(*args, **kwargs):
    from allthemix.competitors.alia.scoring import (
        _checkpoint_identity as implementation,
    )

    return implementation(*args, **kwargs)


def _load_baseline_state(*args, **kwargs):
    from allthemix.competitors.alia.scoring import (
        _load_baseline_state as implementation,
    )

    return implementation(*args, **kwargs)


def _make_predictor(*args, **kwargs):
    from allthemix.competitors.alia.scoring import (
        _make_predictor as implementation,
    )

    return implementation(*args, **kwargs)


def _preprocess_image(*args, **kwargs):
    from allthemix.competitors.alia.scoring import (
        _preprocess_image as implementation,
    )

    return implementation(*args, **kwargs)


def _resolve_checkpoint(*args, **kwargs):
    from allthemix.competitors.alia.scoring import (
        _resolve_checkpoint as implementation,
    )

    return implementation(*args, **kwargs)


def get_preprocessor(*args, **kwargs):
    from allthemix.data.preprocessors.selector import (
        get_preprocessor as implementation,
    )

    return implementation(*args, **kwargs)


def _jax_local_devices():
    import jax

    return jax.local_devices()


def _plain_record(record: dict[str, Any]) -> dict[str, Any]:
    """Strip paths added only while reading a manifest."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"resolved_image_path", "manifest_path"}
    }


def _batches(values: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield finite batches without retaining decoded images globally."""
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_semantic_stage(
    scorer,
    artifact_dir: str | Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Apply SaSPA's official CLIP positive-vs-negative argmax rule."""
    if batch_size < 1:
        raise ValueError("SaSPA CLIP batch_size must be positive.")
    root = Path(artifact_dir).expanduser().resolve()
    records = read_stage_records(
        root,
        stage="generated",
        check_images=True,
        require_complete=True,
    )
    semantic_config = {
        "stage": "semantic",
        "model_id": getattr(
            scorer.config,
            "model_id",
            getattr(scorer.config, "model_name", "unknown"),
        ),
        "model_commit": scorer.resolved_model_commit,
        "positive_prompts": list(scorer.positive_prompts),
        "negative_prompts": list(scorer.negative_prompts),
        "logit_scale": (
            float(scorer.config.logit_scale)
            if hasattr(scorer.config, "logit_scale")
            else "learned-openai-clip-scale"
        ),
        "input_config_fingerprint": records[0]["config_fingerprint"],
    }
    semantic_fingerprint = config_fingerprint(semantic_config)
    summary_path = root / "semantic_summary.json"
    atomic_write_json(
        summary_path,
        {
            "complete": False,
            "stage": "semantic",
            "record_count": 0,
            "semantic_fingerprint": semantic_fingerprint,
        },
    )

    output_records = []
    completed = 0
    for batch in _batches(records, batch_size=batch_size):
        images = []
        for record in batch:
            with Image.open(record["resolved_image_path"]) as image:
                images.append(image.convert("RGB").copy())
        scores = scorer.score(images)
        if len(scores) != len(batch):
            raise RuntimeError("SaSPA CLIP scorer returned the wrong batch size.")
        for record, score in zip(batch, scores):
            output_records.append(
                {
                    **_plain_record(record),
                    **score,
                    "stage": "semantic",
                    "semantic_scoring_fingerprint": semantic_fingerprint,
                }
            )
        completed += len(batch)
        print(f"SaSPA CLIP: {completed}/{len(records)} images scored")

    scorer.synchronize()
    atomic_write_jsonl(root / "semantic.jsonl", output_records)
    summary = {
        "complete": True,
        "stage": "semantic",
        "record_count": len(output_records),
        "semantic_pass": sum(
            bool(record["semantic_pass"]) for record in output_records
        ),
        "semantic_fingerprint": semantic_fingerprint,
        "semantic_config": semantic_config,
    }
    atomic_write_json(summary_path, summary)

    return summary


def _score_classifier_batches(
    records: list[dict[str, Any]],
    preprocess,
    predict,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """Classify generated images with deterministic eval preprocessing."""
    completed = 0
    for batch in _batches(records, batch_size=batch_size):
        images = []
        for record in batch:
            with Image.open(record["resolved_image_path"]) as image:
                images.append(
                    _preprocess_image(
                        image.convert("RGB"),
                        int(record["label"]),
                        preprocess,
                    )
                )
        probabilities = predict(np.stack(images))
        for record, probability in zip(batch, probabilities):
            yield record, probability
        completed += len(batch)
        print(
            f"SaSPA JAX classifier: {completed}/{len(records)} images scored"
        )


def filter_classifier_stage(
    config_path: str | Path,
    checkpoint_path: str | Path,
    artifact_dir: str | Path,
    top_k: int = 10,
    batch_size: int = 64,
    distributed: bool = True,
    require_semantic_pass: bool = True,
) -> dict[str, Any]:
    """Publish images whose assigned class is in the baseline's top-k."""
    if top_k < 1 or batch_size < 1:
        raise ValueError("SaSPA top_k and batch_size must be positive.")
    root = Path(artifact_dir).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    checkpoint = _resolve_checkpoint(checkpoint_path)
    config, metadata, state = _load_baseline_state(config_file, checkpoint)
    if top_k > metadata.num_classes:
        raise ValueError(
            f"SaSPA top_k={top_k} exceeds {metadata.num_classes} classes."
        )
    dataset = str(config["dataset"])
    validation_split = float(config.get("validation_split", 0.0))
    records = read_stage_records(
        root,
        stage="semantic",
        check_images=True,
        require_complete=True,
    )
    for record in records:
        if str(record["dataset"]).lower() != dataset.lower():
            raise ValueError(
                "SaSPA artifact and baseline config datasets differ: "
                f"{record['dataset']!r} vs {dataset!r}."
            )
        if abs(float(record["validation_split"]) - validation_split) > 1.0e-9:
            raise ValueError(
                "SaSPA artifact and baseline validation splits differ: "
                f"{record['validation_split']} vs {validation_split}."
            )

    scoring_config = {
        "stage": "final",
        "method": "saspa",
        "filter": "official-clip-plus-classifier-top-k",
        "dataset": dataset,
        "validation_split": validation_split,
        "top_k": top_k,
        "require_semantic_pass": require_semantic_pass,
        "model": config.get("model"),
        "config_sha256": sha256_file(config_file),
        "checkpoint_path": str(checkpoint),
        "checkpoint_identity": _checkpoint_identity(checkpoint),
        "input_semantic_fingerprint": records[0].get(
            "semantic_scoring_fingerprint"
        ),
    }
    scoring_fingerprint = config_fingerprint(scoring_config)
    summary_path = root / "final_summary.json"
    atomic_write_json(
        summary_path,
        {
            "complete": False,
            "stage": "final",
            "kept_records": 0,
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

    kept = []
    rejected_semantic = 0
    rejected_top_k = 0
    for record, probabilities in _score_classifier_batches(
        records,
        preprocess=preprocess,
        predict=predict,
        batch_size=batch_size,
    ):
        label = int(record["label"])
        top_labels = np.argsort(-probabilities)[:top_k]
        top_k_pass = bool(np.any(top_labels == label))
        semantic_pass = bool(record.get("semantic_pass", False))
        if require_semantic_pass and not semantic_pass:
            rejected_semantic += 1
            continue
        if not top_k_pass:
            rejected_top_k += 1
            continue
        predicted = int(top_labels[0])
        kept.append(
            {
                **_plain_record(record),
                "stage": "final",
                "filter_status": "keep",
                "classifier_top_k": top_k,
                "classifier_top_k_pass": True,
                "classifier_top_k_labels": [
                    int(value) for value in top_labels
                ],
                "classifier_predicted_label": predicted,
                "classifier_max_probability": float(probabilities[predicted]),
                "classifier_assigned_label_probability": float(
                    probabilities[label]
                ),
                "classifier_scoring_fingerprint": scoring_fingerprint,
            }
        )

    atomic_write_jsonl(root / "manifest.jsonl", kept)
    local_devices = _jax_local_devices()
    summary = {
        "complete": True,
        "stage": "final",
        "record_count": len(records),
        "kept_records": len(kept),
        "rejected_semantic": rejected_semantic,
        "rejected_top_k": rejected_top_k,
        "top_k": top_k,
        "require_semantic_pass": require_semantic_pass,
        "distributed": bool(distributed and len(local_devices) > 1),
        "local_device_count": len(local_devices),
        "scoring_fingerprint": scoring_fingerprint,
        "scoring_config": scoring_config,
    }
    atomic_write_json(summary_path, summary)

    return summary
