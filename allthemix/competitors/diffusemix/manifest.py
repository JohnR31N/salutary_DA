"""Manifest contract between offline PyTorch generation and JAX training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
_SHARD_MANIFEST_PATTERN = re.compile(
    r"manifest-(\d+)-of-(\d+)\.jsonl"
)
_REQUIRED_RECORD_KEYS = (
    "schema_version",
    "image_path",
    "label",
    "dataset",
    "source_id",
    "source_partition",
    "augmentation_index",
    "prompt",
    "mask",
    "fractal_alpha",
    "validation_split",
)


def _manifest_paths(
    manifest_path: str | Path,
) -> list[Path]:
    """Resolve a manifest file or a directory of generation shards."""
    path = Path(
        manifest_path,
    ).expanduser()

    if path.is_file():
        if _SHARD_MANIFEST_PATTERN.fullmatch(
            path.name,
        ):
            raise ValueError(
                "Pass the DiffuseMix artifact directory, not one sharded "
                f"manifest file ({path.name}), so the complete shard set can "
                "be validated."
            )
        return [
            path.resolve(),
        ]

    if not path.is_dir():
        raise FileNotFoundError(
            f"DiffuseMix manifest does not exist: {path}"
        )

    candidates = []
    single_manifest = path / "manifest.jsonl"
    if single_manifest.is_file():
        candidates.append(
            single_manifest.resolve(),
        )
    candidates.extend(
        candidate.resolve()
        for candidate in sorted(
            path.glob(
                "manifest-*-of-*.jsonl",
            )
        )
        if candidate.is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            "No DiffuseMix manifest files were found under: "
            f"{path}"
        )

    if single_manifest.is_file() and len(
        candidates,
    ) > 1:
        raise ValueError(
            "DiffuseMix artifact contains both manifest.jsonl and sharded "
            f"manifests: {path}."
        )

    if not single_manifest.is_file():
        shard_coordinates = []
        for candidate in candidates:
            match = _SHARD_MANIFEST_PATTERN.fullmatch(
                candidate.name,
            )
            if match is None:
                raise ValueError(
                    f"Invalid DiffuseMix shard manifest name: {candidate.name}"
                )
            shard_coordinates.append(
                (
                    int(
                        match.group(
                            1,
                        )
                    ),
                    int(
                        match.group(
                            2,
                        )
                    ),
                )
            )

        totals = {
            total
            for _index, total in shard_coordinates
        }
        if len(
            totals,
        ) != 1:
            raise ValueError(
                "DiffuseMix manifests disagree on their total shard count: "
                f"{sorted(totals)}."
            )
        total = next(
            iter(
                totals,
            )
        )
        indices = {
            index
            for index, _total in shard_coordinates
        }
        expected_indices = set(
            range(
                total,
            )
        )
        if indices != expected_indices:
            missing = sorted(
                expected_indices - indices,
            )
            raise ValueError(
                "DiffuseMix artifact has an incomplete manifest shard set. "
                f"Expected {total} shards; missing indices: {missing}."
            )

    return candidates


def _read_json_object(
    path: Path,
    description: str,
) -> dict[str, Any]:
    """Read one JSON object with an artifact-focused error message."""
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in DiffuseMix {description}: {path}."
        ) from error
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"DiffuseMix {description} must contain a JSON object: {path}."
        )

    return payload


def _native_run_fingerprint(
    manifest_files: list[Path],
) -> str | None:
    """Return the native producer fingerprint when run_config.json exists."""
    parents = {
        manifest_file.parent
        for manifest_file in manifest_files
    }
    if len(
        parents,
    ) != 1:
        raise ValueError(
            "DiffuseMix manifest shards must share one artifact directory."
        )
    artifact_dir = next(
        iter(
            parents,
        )
    )
    run_config_path = artifact_dir / "run_config.json"
    if not run_config_path.is_file():
        return None

    payload = _read_json_object(
        path=run_config_path,
        description="run configuration",
    )
    fingerprint = payload.get(
        "config_fingerprint",
    )
    if not isinstance(
        fingerprint,
        str,
    ) or not fingerprint.strip():
        raise ValueError(
            "DiffuseMix run_config.json is missing a nonempty "
            f"config_fingerprint: {run_config_path}."
        )

    return fingerprint


def _summary_path_for_manifest(
    manifest_file: Path,
) -> Path:
    """Return the producer summary paired with one manifest shard."""
    if manifest_file.name == "manifest.jsonl":
        return manifest_file.parent / "summary.json"

    match = _SHARD_MANIFEST_PATTERN.fullmatch(
        manifest_file.name,
    )
    if match is None:
        raise ValueError(
            f"Invalid DiffuseMix manifest filename: {manifest_file.name}."
        )

    return manifest_file.parent / (
        f"summary-{int(match.group(1)):05d}-of-"
        f"{int(match.group(2)):05d}.json"
    )


def _validate_native_summary(
    manifest_file: Path,
    expected_fingerprint: str,
    manifest_record_count: int,
) -> None:
    """Require a completed producer summary matching one manifest shard."""
    summary_path = _summary_path_for_manifest(
        manifest_file,
    )
    if not summary_path.is_file():
        raise ValueError(
            "DiffuseMix artifact is not complete: missing producer summary "
            f"for {manifest_file.name}: {summary_path}."
        )
    summary = _read_json_object(
        path=summary_path,
        description="generation summary",
    )
    if summary.get(
        "complete",
    ) is not True:
        raise ValueError(
            "DiffuseMix artifact is not complete according to its producer "
            f"summary: {summary_path}."
        )
    if summary.get(
        "manifest",
    ) != manifest_file.name:
        raise ValueError(
            "DiffuseMix summary references the wrong manifest: "
            f"{summary_path}."
        )
    if summary.get(
        "config_fingerprint",
    ) != expected_fingerprint:
        raise ValueError(
            "DiffuseMix summary fingerprint does not match run_config.json: "
            f"{summary_path}."
        )

    summary_count = summary.get(
        "manifest_record_count",
    )
    if (
        isinstance(
            summary_count,
            bool,
        )
        or not isinstance(
            summary_count,
            int,
        )
        or summary_count != manifest_record_count
    ):
        raise ValueError(
            "DiffuseMix summary manifest_record_count mismatch for "
            f"{manifest_file.name}: summary={summary_count!r}, "
            f"actual={manifest_record_count}."
        )

    match = _SHARD_MANIFEST_PATTERN.fullmatch(
        manifest_file.name,
    )
    expected_shard_index = 0 if match is None else int(
        match.group(
            1,
        )
    )
    expected_num_shards = 1 if match is None else int(
        match.group(
            2,
        )
    )
    if (
        summary.get(
            "shard_index",
        )
        != expected_shard_index
        or summary.get(
            "num_shards",
        )
        != expected_num_shards
    ):
        raise ValueError(
            "DiffuseMix summary shard coordinates do not match its manifest: "
            f"{summary_path}."
        )

    source_catalog_sha256 = summary.get(
        "source_catalog_sha256",
    )
    if not isinstance(
        source_catalog_sha256,
        str,
    ) or re.fullmatch(
        r"[0-9a-fA-F]{64}",
        source_catalog_sha256,
    ) is None:
        raise ValueError(
            "DiffuseMix summary is missing a valid source catalog digest: "
            f"{summary_path}."
        )


def _sha256_file(
    path: Path,
) -> str:
    """Hash one artifact file without importing the PyTorch generator."""
    digest = hashlib.sha256()
    with path.open(
        mode="rb",
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024,
            ),
            b"",
        ):
            digest.update(
                block,
            )

    return digest.hexdigest()


def _validate_record(
    record: dict[str, Any],
    manifest_file: Path,
    line_number: int,
    check_images: bool,
    require_output_sha256: bool = False,
) -> dict[str, Any]:
    """Validate and resolve one JSONL record."""
    missing = [
        key
        for key in _REQUIRED_RECORD_KEYS
        if key not in record
    ]
    if missing:
        raise ValueError(
            f"Missing DiffuseMix manifest keys {missing} at "
            f"{manifest_file}:{line_number}."
        )

    schema_version = record["schema_version"]
    if (
        isinstance(
            schema_version,
            bool,
        )
        or not isinstance(
            schema_version,
            int,
        )
        or schema_version != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported DiffuseMix manifest schema at "
            f"{manifest_file}:{line_number}: "
            f"{record['schema_version']}."
        )

    label = record["label"]
    if isinstance(
        label,
        bool,
    ) or not isinstance(
        label,
        int,
    ):
        raise ValueError(
            "DiffuseMix labels must be JSON integers at "
            f"{manifest_file}:{line_number}."
        )
    if label < 0:
        raise ValueError(
            "DiffuseMix labels must be nonnegative at "
            f"{manifest_file}:{line_number}."
        )

    raw_image_path = record["image_path"]
    if not isinstance(
        raw_image_path,
        str,
    ) or not raw_image_path.strip():
        raise ValueError(
            "DiffuseMix image_path must be a nonempty string at "
            f"{manifest_file}:{line_number}."
        )
    image_path = Path(
        raw_image_path,
    )
    if not image_path.is_absolute():
        image_path = manifest_file.parent / image_path
    image_path = image_path.resolve()

    if check_images and not image_path.is_file():
        raise FileNotFoundError(
            "DiffuseMix image referenced by the manifest is missing: "
            f"{image_path} ({manifest_file}:{line_number})"
        )

    output_sha256 = record.get(
        "output_png_sha256",
    )
    if require_output_sha256 and output_sha256 is None:
        raise ValueError(
            "Native DiffuseMix records must contain output_png_sha256 at "
            f"{manifest_file}:{line_number}."
        )
    if output_sha256 is not None:
        if not isinstance(
            output_sha256,
            str,
        ) or re.fullmatch(
            r"[0-9a-fA-F]{64}",
            output_sha256,
        ) is None:
            raise ValueError(
                "DiffuseMix output_png_sha256 must be a 64-character hex "
                f"digest at {manifest_file}:{line_number}."
            )
        output_sha256 = output_sha256.lower()
        if check_images:
            actual_sha256 = _sha256_file(
                image_path,
            )
            if actual_sha256 != output_sha256:
                raise ValueError(
                    "DiffuseMix image checksum mismatch at "
                    f"{manifest_file}:{line_number}: {image_path}."
                )

    source_id = record["source_id"]
    if not isinstance(
        source_id,
        str,
    ) or not source_id.strip():
        raise ValueError(
            "DiffuseMix source_id must be a nonempty string at "
            f"{manifest_file}:{line_number}."
        )

    augmentation_index = record["augmentation_index"]
    if (
        isinstance(
            augmentation_index,
            bool,
        )
        or not isinstance(
            augmentation_index,
            int,
        )
        or augmentation_index < 0
    ):
        raise ValueError(
            "DiffuseMix augmentation_index must be a nonnegative JSON "
            f"integer at {manifest_file}:{line_number}."
        )

    fractal_alpha = record["fractal_alpha"]
    if isinstance(
        fractal_alpha,
        bool,
    ) or not isinstance(
        fractal_alpha,
        (
            int,
            float,
        ),
    ):
        raise ValueError(
            "DiffuseMix fractal_alpha must be numeric at "
            f"{manifest_file}:{line_number}."
        )
    fractal_alpha = float(
        fractal_alpha,
    )
    if not math.isfinite(
        fractal_alpha,
    ) or not 0.0 <= fractal_alpha <= 1.0:
        raise ValueError(
            "DiffuseMix fractal_alpha must be finite and in [0, 1] at "
            f"{manifest_file}:{line_number}."
        )

    normalized = dict(
        record,
    )
    normalized["label"] = label
    normalized["augmentation_index"] = augmentation_index
    normalized["fractal_alpha"] = fractal_alpha
    if output_sha256 is not None:
        normalized["output_png_sha256"] = output_sha256
    normalized["resolved_image_path"] = str(
        image_path,
    )
    normalized["manifest_path"] = str(
        manifest_file,
    )

    return normalized


def iter_manifest_records(
    manifest_path: str | Path,
    check_images: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream validated, resolved records from one generation run."""
    manifest_files = _manifest_paths(
        manifest_path,
    )
    native_fingerprint = _native_run_fingerprint(
        manifest_files,
    )
    seen_outputs: set[str] = set()
    seen_jobs: set[tuple[str, int]] = set()
    seen_fingerprint: str | None = None
    fingerprint_presence: bool | None = None
    record_count = 0

    for manifest_file in manifest_files:
        manifest_record_count = 0
        with manifest_file.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    raw_record = json.loads(
                        line,
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSON in DiffuseMix manifest at "
                        f"{manifest_file}:{line_number}."
                    ) from error

                if not isinstance(
                    raw_record,
                    dict,
                ):
                    raise ValueError(
                        "Each DiffuseMix manifest line must be an object at "
                        f"{manifest_file}:{line_number}."
                    )

                raw_fingerprint = raw_record.get(
                    "config_fingerprint",
                )
                has_fingerprint = raw_fingerprint is not None
                if fingerprint_presence is None:
                    fingerprint_presence = has_fingerprint
                elif fingerprint_presence != has_fingerprint:
                    raise ValueError(
                        "DiffuseMix manifest records inconsistently include "
                        "config_fingerprint."
                    )
                if native_fingerprint is not None and not has_fingerprint:
                    raise ValueError(
                        "Native DiffuseMix records must contain "
                        f"config_fingerprint at {manifest_file}:{line_number}."
                    )
                if has_fingerprint:
                    if not isinstance(
                        raw_fingerprint,
                        str,
                    ) or not raw_fingerprint.strip():
                        raise ValueError(
                            "DiffuseMix config_fingerprint must be a nonempty "
                            f"string at {manifest_file}:{line_number}."
                        )
                    if seen_fingerprint is None:
                        seen_fingerprint = raw_fingerprint
                    elif raw_fingerprint != seen_fingerprint:
                        raise ValueError(
                            "DiffuseMix manifest records contain different "
                            "config_fingerprint values."
                        )
                    if (
                        native_fingerprint is not None
                        and raw_fingerprint != native_fingerprint
                    ):
                        raise ValueError(
                            "DiffuseMix record fingerprint does not match "
                            f"run_config.json at {manifest_file}:{line_number}."
                        )

                record = _validate_record(
                    record=raw_record,
                    manifest_file=manifest_file,
                    line_number=line_number,
                    check_images=check_images,
                    require_output_sha256=native_fingerprint is not None,
                )
                resolved_output = record["resolved_image_path"]
                if resolved_output in seen_outputs:
                    raise ValueError(
                        "Duplicate DiffuseMix image_path across manifests: "
                        f"{resolved_output}."
                    )
                seen_outputs.add(
                    resolved_output,
                )

                augmentation_index = int(
                    record.get(
                        "augmentation_index",
                        0,
                    )
                )
                job_key = (
                    str(
                        record["source_id"],
                    ),
                    augmentation_index,
                )
                if job_key in seen_jobs:
                    raise ValueError(
                        "Duplicate DiffuseMix source/augmentation job across "
                        f"manifests: {job_key}."
                    )
                seen_jobs.add(
                    job_key,
                )
                record_count += 1
                manifest_record_count += 1
                yield record

        if native_fingerprint is not None:
            _validate_native_summary(
                manifest_file=manifest_file,
                expected_fingerprint=native_fingerprint,
                manifest_record_count=manifest_record_count,
            )

    if record_count == 0:
        raise ValueError(
            f"DiffuseMix manifest contains no examples: {manifest_path}"
        )


def read_manifest_records(
    manifest_path: str | Path,
    check_images: bool = True,
) -> list[dict[str, Any]]:
    """Materialize validated records for inspection and small datasets."""
    return list(
        iter_manifest_records(
            manifest_path=manifest_path,
            check_images=check_images,
        )
    )


def validate_manifest_for_training(
    manifest_path: str | Path,
    dataset: str,
    num_classes: int,
    validation_split: float | None,
    check_images: bool = True,
) -> int:
    """Validate labels and source provenance before JAX sees the dataset.

    ``validation_split=None`` is used when model validation comes from the
    official evaluation split. In that protocol, any deterministic subset of
    the official training source is leakage-safe for appended generated data.
    """
    expected_dataset = dataset.strip().lower()
    if not expected_dataset:
        raise ValueError("Training dataset name must be nonempty.")
    if num_classes < 1:
        raise ValueError(
            f"num_classes must be positive. Got {num_classes}."
        )
    if validation_split is not None and (
        not math.isfinite(
            validation_split,
        )
        or not 0.0 <= validation_split < 1.0
    ):
        raise ValueError(
            "Training validation_split must be finite and in [0, 1): "
            f"got {validation_split!r}."
        )
    record_count = 0

    for record in iter_manifest_records(
        manifest_path=manifest_path,
        check_images=check_images,
    ):
        if record["label"] >= num_classes:
            raise ValueError(
                "DiffuseMix label is outside the classifier range: "
                f"label={record['label']}, num_classes={num_classes}, "
                f"image={record['resolved_image_path']}."
            )

        record_dataset = str(
            record["dataset"],
        ).strip().lower()
        if not record_dataset:
            raise ValueError(
                "DiffuseMix manifest dataset must be a nonempty name: "
                f"image={record['resolved_image_path']}."
            )
        if record_dataset != expected_dataset:
            raise ValueError(
                "DiffuseMix manifest dataset mismatch: generated for "
                f"'{record_dataset}', training requested '{expected_dataset}'."
            )

        if str(
            record["source_partition"],
        ).lower() != "train":
            raise ValueError(
                "DiffuseMix training manifests may contain only train-source "
                f"records. Got source_partition="
                f"{record['source_partition']!r}."
            )

        record_split = float(
            record["validation_split"],
        )
        if not math.isfinite(
            record_split,
        ) or not 0.0 <= record_split < 1.0:
            raise ValueError(
                "DiffuseMix validation_split must be finite and in [0, 1): "
                f"got {record_split!r} for "
                f"{record['resolved_image_path']}."
            )
        if (
            validation_split is not None
            and abs(record_split - validation_split) > 1.0e-9
        ):
            raise ValueError(
                "DiffuseMix generation-source split mismatch: generated with "
                f"{record_split}, training requested {validation_split}. "
                "Regenerate from the same training-source partition."
            )
        record_count += 1

    return record_count


def count_manifest_examples(
    manifest_path: str | Path,
) -> int:
    """Count valid manifest records without decoding image contents."""
    return sum(
        1
        for _record in iter_manifest_records(
            manifest_path=manifest_path,
            check_images=False,
        )
    )


def load_manifest_dataset(
    manifest_path: str | Path,
    image_size: int | None = None,
    check_images: bool = True,
):
    """Build a raw tf.data dataset from generated image paths and labels."""
    import tensorflow as tf

    if image_size is not None and image_size < 1:
        raise ValueError(
            f"image_size must be positive. Got {image_size}."
        )

    image_paths = []
    labels = []
    for record in iter_manifest_records(
        manifest_path=manifest_path,
        check_images=check_images,
    ):
        image_paths.append(
            record["resolved_image_path"],
        )
        labels.append(
            record["label"],
        )
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
        )
    )

    def decode_example(
        image_path,
        label,
    ):
        image_bytes = tf.io.read_file(
            image_path,
        )
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape(
            [
                None,
                None,
                3,
            ]
        )
        if image_size is not None:
            # Diffusion editing is normally performed at 256 or 512 pixels.
            # Resize back to the classifier's raw input resolution before
            # ordinary dataset augmentation and normalization (notably CIFAR,
            # whose preprocessor otherwise assumes an already-32px image).
            image = tf.image.resize(
                image,
                size=[
                    image_size,
                    image_size,
                ],
                method="bilinear",
                antialias=True,
            )
            image = tf.cast(
                tf.clip_by_value(
                    tf.round(
                        image,
                    ),
                    0.0,
                    255.0,
                ),
                tf.uint8,
            )
            image = tf.ensure_shape(
                image,
                [
                    image_size,
                    image_size,
                    3,
                ],
            )

        return {
            "image": image,
            "label": tf.cast(
                label,
                tf.int64,
            ),
        }

    return dataset.map(
        decode_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
