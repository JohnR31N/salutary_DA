"""Artifact contract between SaSPA generation, filtering, and JAX."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from allthemix.competitors.generative.artifacts import sha256_file

SASPA_SCHEMA_VERSION = 1
STAGES = ("generated", "semantic", "final")
_STAGE_FILES = {
    "generated": "generated.jsonl",
    "semantic": "semantic.jsonl",
    "final": "manifest.jsonl",
}
_SHARD_PATTERN = re.compile(r"generated-(\d+)-of-(\d+)\.jsonl")
_REQUIRED_KEYS = (
    "schema_version",
    "method",
    "record_id",
    "image_path",
    "output_png_sha256",
    "dataset",
    "source_id",
    "source_index",
    "source_partition",
    "augmentation_index",
    "label",
    "prompt",
    "validation_split",
    "config_fingerprint",
)


def _normalize_stage(stage: str) -> str:
    value = stage.strip().lower()
    if value not in STAGES:
        raise ValueError(f"SaSPA stage must be one of {STAGES}. Got {stage!r}.")

    return value


def stage_manifest_name(
    stage: str,
    shard_index: int = 0,
    num_shards: int = 1,
) -> str:
    """Return the canonical manifest filename for one stage."""
    stage_name = _normalize_stage(stage)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            "Invalid SaSPA shard coordinates: "
            f"index={shard_index}, total={num_shards}."
        )
    if num_shards == 1:
        return _STAGE_FILES[stage_name]
    if stage_name != "generated":
        raise ValueError("Only SaSPA generation manifests may be sharded.")

    return f"generated-{shard_index:05d}-of-{num_shards:05d}.jsonl"


def stage_manifest_paths(
    artifact_path: str | Path,
    stage: str = "final",
) -> list[Path]:
    """Resolve one file or a complete manifest stage under a directory."""
    stage_name = _normalize_stage(stage)
    path = Path(artifact_path).expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"SaSPA artifact does not exist: {path}")

    single = path / _STAGE_FILES[stage_name]
    shards = sorted(path.glob("generated-*-of-*.jsonl"))
    if single.is_file() and stage_name == "generated" and shards:
        raise ValueError(
            "SaSPA artifact mixes a single generated manifest with shards: "
            f"{path}."
        )
    if single.is_file():
        return [single.resolve()]
    if stage_name != "generated" or not shards:
        raise FileNotFoundError(
            f"No SaSPA {stage_name} manifest was found under: {path}"
        )

    coordinates = []
    for shard in shards:
        match = _SHARD_PATTERN.fullmatch(shard.name)
        if match is None:
            raise ValueError(f"Invalid SaSPA shard name: {shard.name}.")
        coordinates.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _index, total in coordinates}
    if len(totals) != 1:
        raise ValueError("SaSPA generated shards disagree on shard count.")
    total = next(iter(totals))
    indices = {index for index, _total in coordinates}
    missing = sorted(set(range(total)) - indices)
    if missing:
        raise ValueError(f"SaSPA generated shard set is missing {missing}.")

    return [shard.resolve() for shard in shards]


def _resolve_image_path(record: dict[str, Any], manifest: Path) -> Path:
    path = Path(str(record["image_path"]))
    if not path.is_absolute():
        path = manifest.parent / path

    return path.resolve()


def _validate_record(
    record: dict[str, Any],
    manifest: Path,
    line_number: int,
    check_images: bool,
) -> dict[str, Any]:
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(
            f"Missing SaSPA keys {missing} at {manifest}:{line_number}."
        )
    if record["schema_version"] != SASPA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SaSPA schema at {manifest}:{line_number}."
        )
    if str(record["method"]).strip().lower() != "saspa":
        raise ValueError(
            f"SaSPA record has method={record['method']!r} at "
            f"{manifest}:{line_number}."
        )
    for key in ("label", "source_index", "augmentation_index"):
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"SaSPA {key} must be a nonnegative integer at "
                f"{manifest}:{line_number}."
            )
    validation_split = float(record["validation_split"])
    if not math.isfinite(validation_split) or not 0.0 <= validation_split < 1.0:
        raise ValueError(
            f"Invalid SaSPA validation_split at {manifest}:{line_number}."
        )
    for key in ("record_id", "source_id", "config_fingerprint"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(
                f"SaSPA {key} must be a nonempty string at "
                f"{manifest}:{line_number}."
            )

    image_path = _resolve_image_path(record, manifest)
    if check_images and not image_path.is_file():
        raise FileNotFoundError(
            f"SaSPA image is missing at {manifest}:{line_number}: "
            f"{image_path}"
        )
    expected_digest = str(record["output_png_sha256"]).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError(
            f"Invalid SaSPA image digest at {manifest}:{line_number}."
        )
    if check_images and sha256_file(image_path) != expected_digest:
        raise ValueError(
            f"SaSPA image checksum mismatch at {manifest}:{line_number}."
        )

    normalized = dict(record)
    normalized["validation_split"] = validation_split
    normalized["resolved_image_path"] = str(image_path)
    normalized["manifest_path"] = str(manifest)

    return normalized


def iter_stage_records(
    artifact_path: str | Path,
    stage: str = "final",
    check_images: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream validated records while rejecting duplicate jobs and files."""
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    fingerprints: set[str] = set()
    count = 0
    for manifest in stage_manifest_paths(artifact_path, stage=stage):
        with manifest.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid SaSPA JSON at {manifest}:{line_number}."
                    ) from error
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"SaSPA JSONL rows must be objects at "
                        f"{manifest}:{line_number}."
                    )
                record = _validate_record(
                    raw,
                    manifest=manifest,
                    line_number=line_number,
                    check_images=check_images,
                )
                record_id = record["record_id"]
                image_path = record["resolved_image_path"]
                if record_id in seen_ids:
                    raise ValueError(f"Duplicate SaSPA record_id: {record_id}.")
                if image_path in seen_paths:
                    raise ValueError(f"Duplicate SaSPA image_path: {image_path}.")
                seen_ids.add(record_id)
                seen_paths.add(image_path)
                fingerprints.add(str(record["config_fingerprint"]))
                count += 1
                yield record

    if count == 0:
        raise ValueError(f"SaSPA {stage} manifest is empty: {artifact_path}.")
    if len(fingerprints) != 1:
        raise ValueError("SaSPA stage combines different generation configs.")


def read_stage_records(
    artifact_path: str | Path,
    stage: str = "final",
    check_images: bool = True,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    """Materialize one SaSPA artifact stage."""
    records = list(
        iter_stage_records(
            artifact_path=artifact_path,
            stage=stage,
            check_images=check_images,
        )
    )
    if require_complete:
        _validate_completion(artifact_path, stage, len(records))

    return records


def _summary_path(manifest: Path, stage: str) -> Path:
    match = _SHARD_PATTERN.fullmatch(manifest.name)
    if match is not None:
        return manifest.parent / (
            f"generated_summary-{int(match.group(1)):05d}-of-"
            f"{int(match.group(2)):05d}.json"
        )

    return manifest.parent / f"{stage}_summary.json"


def _validate_completion(
    artifact_path: str | Path,
    stage: str,
    record_count: int,
) -> None:
    """Require every producer completion marker and exact count."""
    stage_name = _normalize_stage(stage)
    manifests = stage_manifest_paths(artifact_path, stage=stage_name)
    total = 0
    for manifest in manifests:
        summary_path = _summary_path(manifest, stage_name)
        if not summary_path.is_file():
            raise ValueError(
                f"SaSPA {stage_name} artifact is incomplete: "
                f"missing {summary_path}."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is not True:
            raise ValueError(f"SaSPA summary is incomplete: {summary_path}.")
        count_key = "kept_records" if stage_name == "final" else "record_count"
        summary_count = summary.get(count_key)
        if not isinstance(summary_count, int) or summary_count < 0:
            raise ValueError(f"Invalid SaSPA count in {summary_path}.")
        total += summary_count
    if total != record_count:
        raise ValueError(
            f"SaSPA {stage_name} summary count mismatch: "
            f"summary={total}, actual={record_count}."
        )


def validate_manifest_for_training(
    manifest_path: str | Path,
    dataset: str,
    num_classes: int,
    validation_split: float | None,
    check_images: bool = True,
) -> int:
    """Validate final filtering and source provenance before JAX training.

    A ``None`` split means the held-out validation examples come from the
    official evaluation source rather than from official training data.
    """
    expected_dataset = dataset.strip().lower()
    records = read_stage_records(
        manifest_path,
        stage="final",
        check_images=check_images,
        require_complete=True,
    )
    for record in records:
        if str(record["dataset"]).strip().lower() != expected_dataset:
            raise ValueError(
                "SaSPA dataset mismatch: generated for "
                f"{record['dataset']!r}, requested {dataset!r}."
            )
        if record["label"] >= num_classes:
            raise ValueError(
                f"SaSPA label {record['label']} is outside {num_classes} "
                "classifier classes."
            )
        if str(record["source_partition"]).lower() != "train":
            raise ValueError("SaSPA may use only train-partition sources.")
        if (
            validation_split is not None
            and abs(record["validation_split"] - validation_split) > 1.0e-9
        ):
            raise ValueError(
                "SaSPA generation-source split mismatch. Regenerate from "
                "the same training-source partition."
            )
        if record.get("filter_status") != "keep":
            raise ValueError(
                f"SaSPA final manifest contains a rejected record: "
                f"{record['record_id']}."
            )
        if record.get("classifier_top_k_pass") is not True:
            raise ValueError(
                f"SaSPA final record did not pass top-k filtering: "
                f"{record['record_id']}."
            )

    return len(records)


def replacement_catalog(
    manifest_path: str | Path,
    check_images: bool = True,
) -> tuple[list[int], list[list[str]], list[int]]:
    """Group accepted generated paths and labels by original source index."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_stage_records(
        manifest_path,
        stage="final",
        check_images=check_images,
    ):
        grouped[int(record["source_index"])].append(record)
    indices = sorted(grouped)
    paths = []
    labels = []
    for source_index in indices:
        records = sorted(
            grouped[source_index],
            key=lambda record: int(record["augmentation_index"]),
        )
        source_ids = {str(record["source_id"]) for record in records}
        source_labels = {int(record["label"]) for record in records}
        if len(source_ids) != 1 or len(source_labels) != 1:
            raise ValueError(
                "SaSPA records for one source_index disagree on source or "
                f"label: {source_index}."
            )
        paths.append([str(record["resolved_image_path"]) for record in records])
        labels.append(next(iter(source_labels)))

    return indices, paths, labels


def count_manifest_examples(manifest_path: str | Path) -> int:
    """Count accepted SaSPA records without hashing image contents."""
    return sum(
        1
        for _record in iter_stage_records(
            manifest_path,
            stage="final",
            check_images=False,
        )
    )


def load_manifest_dataset(
    manifest_path: str | Path,
    image_size: int | None = None,
    check_images: bool = True,
):
    """Build a raw tf.data dataset for append/replace compatibility."""
    import tensorflow as tf

    records = read_stage_records(
        manifest_path,
        stage="final",
        check_images=check_images,
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            [record["resolved_image_path"] for record in records],
            [record["label"] for record in records],
        )
    )

    def decode(image_path, label):
        image = tf.io.decode_image(
            tf.io.read_file(image_path),
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        if image_size is not None:
            image = tf.image.resize(
                image,
                [image_size, image_size],
                method="bilinear",
                antialias=True,
            )
            image = tf.cast(
                tf.clip_by_value(tf.round(image), 0.0, 255.0),
                tf.uint8,
            )
            image = tf.ensure_shape(image, [image_size, image_size, 3])

        return {"image": image, "label": tf.cast(label, tf.int64)}

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
