"""Artifact contract between ALIA's offline stages and JAX training."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from allthemix.competitors.generative.artifacts import sha256_file

ALIA_SCHEMA_VERSION = 1
STAGE_NAMES = (
    "generated",
    "clip",
    "classifier",
    "final",
)
_STAGE_FILE_NAMES = {
    "generated": "generated.jsonl",
    "clip": "clip.jsonl",
    "classifier": "classifier.jsonl",
    "final": "manifest.jsonl",
}
_SHARD_PATTERN = re.compile(
    r"(generated|clip|classifier)-(\d+)-of-(\d+)\.jsonl"
)
_REQUIRED_RECORD_KEYS = (
    "schema_version",
    "method",
    "record_id",
    "image_path",
    "label",
    "dataset",
    "source_id",
    "source_partition",
    "augmentation_index",
    "prompt",
    "validation_split",
    "config_fingerprint",
    "output_png_sha256",
)


def _validate_stage(stage: str) -> str:
    """Normalize and validate an ALIA artifact stage name."""
    value = stage.strip().lower()
    if value not in STAGE_NAMES:
        raise ValueError(
            f"ALIA stage must be one of {STAGE_NAMES}. Got {stage!r}."
        )

    return value


def _validate_shard_set(paths: list[Path], stage: str) -> None:
    """Require a complete, consistently named manifest shard set."""
    coordinates = []
    for path in paths:
        match = _SHARD_PATTERN.fullmatch(path.name)
        if match is None or match.group(1) != stage:
            raise ValueError(f"Invalid ALIA {stage} shard name: {path.name}")
        coordinates.append((int(match.group(2)), int(match.group(3))))

    totals = {total for _index, total in coordinates}
    if len(totals) != 1:
        raise ValueError(
            f"ALIA {stage} shards disagree on total shard count: {totals}."
        )
    total = next(iter(totals))
    indices = {index for index, _total in coordinates}
    missing = sorted(set(range(total)) - indices)
    if missing:
        raise ValueError(
            f"ALIA {stage} shard set is incomplete; missing {missing}."
        )


def stage_manifest_paths(
    manifest_path: str | Path,
    stage: str = "final",
) -> list[Path]:
    """Resolve one manifest file or a complete stage directory."""
    stage = _validate_stage(stage)
    path = Path(manifest_path).expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"ALIA artifact does not exist: {path}")

    single = path / _STAGE_FILE_NAMES[stage]
    shards = sorted(path.glob(f"{stage}-*-of-*.jsonl"))
    if single.is_file() and shards:
        raise ValueError(
            f"ALIA artifact contains both single and sharded {stage} files: "
            f"{path}."
        )
    if single.is_file():
        return [single.resolve()]
    if not shards:
        raise FileNotFoundError(
            f"No ALIA {stage} manifest was found under: {path}"
        )

    resolved = [candidate.resolve() for candidate in shards]
    _validate_shard_set(resolved, stage)

    return resolved


def stage_manifest_name(
    stage: str,
    shard_index: int = 0,
    num_shards: int = 1,
) -> str:
    """Return the canonical filename for one stage shard."""
    stage = _validate_stage(stage)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            "Invalid ALIA shard coordinates: "
            f"index={shard_index}, total={num_shards}."
        )
    if num_shards == 1:
        return _STAGE_FILE_NAMES[stage]
    if stage == "final":
        raise ValueError("The final ALIA training manifest is not sharded.")

    return f"{stage}-{shard_index:05d}-of-{num_shards:05d}.jsonl"


def _resolve_image_path(record: dict[str, Any], manifest_file: Path) -> Path:
    """Resolve one record's image path relative to its artifact directory."""
    image_path = Path(str(record["image_path"]))
    if not image_path.is_absolute():
        image_path = manifest_file.parent / image_path

    return image_path.resolve()


def _validate_record(
    record: dict[str, Any],
    manifest_file: Path,
    line_number: int,
    check_images: bool,
) -> dict[str, Any]:
    """Validate common ALIA record provenance and image integrity."""
    missing = [key for key in _REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(
            f"Missing ALIA manifest keys {missing} at "
            f"{manifest_file}:{line_number}."
        )
    if record["schema_version"] != ALIA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ALIA schema at {manifest_file}:{line_number}: "
            f"{record['schema_version']!r}."
        )
    if str(record["method"]).lower() != "alia":
        raise ValueError(
            f"ALIA record has method={record['method']!r} at "
            f"{manifest_file}:{line_number}."
        )
    if (
        isinstance(record["label"], bool)
        or not isinstance(record["label"], int)
        or record["label"] < 0
    ):
        raise ValueError(
            f"ALIA label must be a nonnegative integer at "
            f"{manifest_file}:{line_number}."
        )
    if not isinstance(record["record_id"], str) or not record["record_id"]:
        raise ValueError(
            f"ALIA record_id must be nonempty at "
            f"{manifest_file}:{line_number}."
        )
    split = float(record["validation_split"])
    if not math.isfinite(split) or not 0.0 <= split < 1.0:
        raise ValueError(
            f"Invalid ALIA validation_split at {manifest_file}:"
            f"{line_number}: {split!r}."
        )

    image_path = _resolve_image_path(record, manifest_file)
    if check_images and not image_path.is_file():
        raise FileNotFoundError(
            f"ALIA image is missing at {manifest_file}:{line_number}: "
            f"{image_path}"
        )
    expected_digest = str(record["output_png_sha256"]).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError(
            f"Invalid ALIA image digest at {manifest_file}:{line_number}."
        )
    if check_images and sha256_file(image_path) != expected_digest:
        raise ValueError(
            f"ALIA image checksum mismatch at {manifest_file}:"
            f"{line_number}: {image_path}"
        )

    normalized = dict(record)
    normalized["validation_split"] = split
    normalized["resolved_image_path"] = str(image_path)
    normalized["manifest_path"] = str(manifest_file)

    return normalized


def iter_stage_records(
    manifest_path: str | Path,
    stage: str = "final",
    check_images: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream validated records from one ALIA artifact stage."""
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    fingerprints: set[str] = set()
    count = 0
    for manifest_file in stage_manifest_paths(manifest_path, stage=stage):
        with manifest_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid ALIA JSON at {manifest_file}:"
                        f"{line_number}."
                    ) from error
                if not isinstance(raw_record, dict):
                    raise ValueError(
                        f"ALIA JSONL rows must be objects at "
                        f"{manifest_file}:{line_number}."
                    )
                record = _validate_record(
                    raw_record,
                    manifest_file=manifest_file,
                    line_number=line_number,
                    check_images=check_images,
                )
                record_id = record["record_id"]
                image_path = record["resolved_image_path"]
                if record_id in seen_ids:
                    raise ValueError(f"Duplicate ALIA record_id: {record_id}")
                if image_path in seen_paths:
                    raise ValueError(f"Duplicate ALIA image_path: {image_path}")
                seen_ids.add(record_id)
                seen_paths.add(image_path)
                fingerprints.add(str(record["config_fingerprint"]))
                count += 1
                yield record

    if count == 0:
        raise ValueError(
            f"ALIA {stage} manifest contains no records: {manifest_path}"
        )
    if len(fingerprints) != 1:
        raise ValueError(
            "ALIA stage combines records from different generation configs."
        )


def read_stage_records(
    manifest_path: str | Path,
    stage: str = "final",
    check_images: bool = True,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    """Materialize one validated ALIA stage."""
    records = list(iter_stage_records(manifest_path, stage, check_images))
    if require_complete:
        _validate_stage_summaries(
            manifest_path=manifest_path,
            stage=stage,
            records=records,
        )

    return records


def _stage_summary_path(manifest_file: Path, stage: str) -> Path:
    """Resolve the completion marker paired with one stage manifest file."""
    match = _SHARD_PATTERN.fullmatch(manifest_file.name)
    if match is None:
        return manifest_file.parent / f"{stage}_summary.json"

    shard_index = int(match.group(2))
    num_shards = int(match.group(3))
    return manifest_file.parent / (
        f"{stage}_summary-{shard_index:05d}-of-{num_shards:05d}.json"
    )


def _validate_stage_summaries(
    manifest_path: str | Path,
    stage: str,
    records: list[dict[str, Any]],
) -> None:
    """Reject partial stage output even when every shard file exists."""
    stage = _validate_stage(stage)
    if stage == "final":
        manifest_files = stage_manifest_paths(manifest_path, stage=stage)
        _validate_final_summary(manifest_files[0].parent, len(records))
        return

    counts: dict[Path, int] = {}
    for record in records:
        manifest_file = Path(str(record["manifest_path"])).resolve()
        counts[manifest_file] = counts.get(manifest_file, 0) + 1

    for manifest_file in stage_manifest_paths(manifest_path, stage=stage):
        summary_path = _stage_summary_path(manifest_file, stage)
        if not summary_path.is_file():
            raise ValueError(
                f"ALIA {stage} artifact is incomplete: missing "
                f"{summary_path}."
            )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid ALIA {stage} summary: {summary_path}"
            ) from error
        if summary.get("complete") is not True:
            raise ValueError(
                f"ALIA {stage} summary is incomplete: {summary_path}"
            )
        if summary.get("stage") != stage:
            raise ValueError(
                f"ALIA stage summary mismatch at {summary_path}: "
                f"{summary.get('stage')!r}."
            )
        record_count = counts.get(manifest_file.resolve(), 0)
        if summary.get("record_count") != record_count:
            raise ValueError(
                f"ALIA {stage} summary count does not match "
                f"{manifest_file}: summary={summary.get('record_count')!r}, "
                f"actual={record_count}."
            )


def _validate_final_summary(
    artifact_dir: Path,
    record_count: int,
) -> None:
    """Require the filter stage's atomic completion marker."""
    summary_path = artifact_dir / "final_summary.json"
    if not summary_path.is_file():
        raise ValueError(
            f"ALIA final artifact is incomplete: missing {summary_path}."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid ALIA final summary: {summary_path}") from error
    if summary.get("complete") is not True:
        raise ValueError(f"ALIA final summary is incomplete: {summary_path}")
    if summary.get("kept_records") != record_count:
        raise ValueError(
            "ALIA final summary count does not match manifest: "
            f"summary={summary.get('kept_records')!r}, actual={record_count}."
        )


def validate_manifest_for_training(
    manifest_path: str | Path,
    dataset: str,
    num_classes: int,
    validation_split: float | None,
    check_images: bool = True,
) -> int:
    """Reject unsafe provenance or rejected edits before JAX training.

    A ``None`` split means validation is sourced from official evaluation
    data, so every train-source artifact remains disjoint from validation.
    """
    expected_dataset = dataset.strip().lower()
    count = 0
    manifest_files = stage_manifest_paths(manifest_path, stage="final")
    for record in iter_stage_records(
        manifest_path,
        stage="final",
        check_images=check_images,
    ):
        if str(record["dataset"]).strip().lower() != expected_dataset:
            raise ValueError(
                "ALIA dataset mismatch: generated for "
                f"{record['dataset']!r}, requested {dataset!r}."
            )
        if record["label"] >= num_classes:
            raise ValueError(
                f"ALIA label {record['label']} is outside {num_classes} "
                "classifier classes."
            )
        if str(record["source_partition"]).lower() != "train":
            raise ValueError(
                "ALIA training may use only train-source edits; got "
                f"{record['source_partition']!r}."
            )
        if (
            validation_split is not None
            and abs(record["validation_split"] - validation_split) > 1.0e-9
        ):
            raise ValueError(
                "ALIA generation-source split mismatch. Regenerate edits "
                "from the same training-source partition: "
                f"artifact={record['validation_split']}, "
                f"training={validation_split}."
            )
        if record.get("filter_status") != "keep":
            raise ValueError(
                f"ALIA final manifest contains an unaccepted record: "
                f"{record['record_id']}."
            )
        count += 1

    _validate_final_summary(manifest_files[0].parent, count)

    return count


def count_manifest_examples(manifest_path: str | Path) -> int:
    """Count final accepted examples without hashing image contents."""
    return sum(
        1
        for _ in iter_stage_records(
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
    """Build a raw tf.data dataset from accepted ALIA image records."""
    import tensorflow as tf

    records = read_stage_records(
        manifest_path,
        stage="final",
        check_images=check_images,
    )
    image_paths = [record["resolved_image_path"] for record in records]
    labels = [record["label"] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def decode(image_path, label):
        """Decode one generated RGB file and restore classifier resolution."""
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
