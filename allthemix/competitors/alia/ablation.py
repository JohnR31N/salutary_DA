"""Paired generated-versus-original ALIA quality ablations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from allthemix.competitors.alia.manifest import read_stage_records
from allthemix.competitors.generative.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    sha256_file,
    stable_seed,
)
from allthemix.competitors.generative.sources import (
    SourceExample,
    iter_allthemix_sources,
)

_DERIVED_RECORD_KEYS = {
    "manifest_path",
    "resolved_image_path",
}


def _select_records(
    records: list[dict[str, object]],
    max_records: int,
    seed: int,
) -> list[dict[str, object]]:
    """Select a deterministic subset without changing full-manifest order."""
    if max_records < 0 or max_records >= len(records):
        return records
    if max_records == 0:
        raise ValueError("max_records must be -1 or positive.")

    ranked = sorted(
        enumerate(records),
        key=lambda item: (
            stable_seed(seed, item[1]["record_id"]),
            str(item[1]["record_id"]),
        ),
    )[:max_records]
    selected_indices = {index for index, _record in ranked}

    return [
        record
        for index, record in enumerate(records)
        if index in selected_indices
    ]


def _save_png(image: Image.Image, path: Path) -> str:
    """Save one source image atomically and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    image.convert("RGB").save(temporary_path, format="PNG")
    temporary_path.replace(path)

    return sha256_file(path)


def _source_file_name(index: int, source_id: str) -> str:
    """Create a stable filesystem-safe name for one paired source image."""
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]

    return f"{index:05d}-{digest}.png"


def _publish_final_artifact(
    artifact_dir: Path,
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Publish a final-manifest-shaped artifact accepted by JAX training."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(artifact_dir / "manifest.jsonl", records)
    atomic_write_json(
        artifact_dir / "final_summary.json",
        {
            "complete": True,
            "kept_records": len(records),
            "stage": "final",
            **summary,
        },
    )


def resolve_paired_sources(
    source_ids: Iterable[str],
    dataset: str,
    data_dir: str,
    validation_split: float,
    sources: Iterable[SourceExample] | None = None,
) -> dict[str, SourceExample]:
    """Resolve manifest source IDs from the identical train partition."""
    required_ids = set(source_ids)
    matched_sources: dict[str, SourceExample] = {}
    source_iterator = sources
    if source_iterator is None:
        source_iterator = iter_allthemix_sources(
            dataset=dataset,
            data_dir=data_dir,
            validation_split=validation_split,
        )
    for source in source_iterator:
        if source.source_id in required_ids:
            matched_sources[source.source_id] = source
            if len(matched_sources) == len(required_ids):
                break

    missing = sorted(required_ids - matched_sources.keys())
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Could not resolve {len(missing)} ALIA source_id values from "
            f"the current training partition. First missing: {preview}"
        )

    return matched_sources


def build_paired_ablation_artifacts(
    artifact_dir: str | Path,
    output_dir: str | Path,
    dataset: str,
    data_dir: str,
    validation_split: float,
    max_records: int = 1000,
    seed: int = 0,
    sources: Iterable[SourceExample] | None = None,
) -> dict[str, object]:
    """Build exactly paired generated-only and original-only artifacts.

    Every generated record is paired with the raw training source identified by
    its ``source_id``. This makes the two subsets identical in size, class
    histogram, source identity, and ordering.
    """
    source_artifact = Path(artifact_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    records = read_stage_records(
        source_artifact,
        stage="final",
        check_images=True,
        require_complete=True,
    )
    selected = _select_records(records, max_records=max_records, seed=seed)

    source_ids = [str(record["source_id"]) for record in selected]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(
            "Paired ALIA ablation requires one generated edit per source. "
            "Rebuild the final manifest with --max-per-source 1."
        )

    matched_sources = resolve_paired_sources(
        source_ids=source_ids,
        dataset=dataset,
        data_dir=data_dir,
        validation_split=validation_split,
        sources=sources,
    )

    original_root = destination / "original"
    generated_root = destination / "generated"
    original_fingerprint = config_fingerprint(
        {
            "ablation": "alia_paired_original",
            "dataset": dataset,
            "seed": seed,
            "source_artifact": str(source_artifact),
            "validation_split": validation_split,
        }
    )
    generated_records: list[dict[str, object]] = []
    original_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []

    for pair_index, generated in enumerate(selected):
        source_id = str(generated["source_id"])
        source = matched_sources[source_id]
        generated_label = int(generated["label"])
        if source.label != generated_label:
            raise ValueError(
                "ALIA source label mismatch for "
                f"{source_id}: manifest={generated_label}, source={source.label}."
            )

        generated_record = {
            key: value
            for key, value in generated.items()
            if key not in _DERIVED_RECORD_KEYS
        }
        generated_record["image_path"] = str(
            Path(str(generated["resolved_image_path"])).resolve()
        )
        generated_records.append(generated_record)

        original_path = (
            original_root
            / "images"
            / _source_file_name(pair_index, source_id)
        )
        original_digest = _save_png(source.image, original_path)
        original_record_id = f"alia-ablation-original-{pair_index:05d}"
        original_records.append(
            {
                "schema_version": int(generated["schema_version"]),
                "method": "alia",
                "stage": "final",
                "record_id": original_record_id,
                "image_path": original_path.relative_to(original_root).as_posix(),
                "output_png_sha256": original_digest,
                "dataset": dataset,
                "source_id": source_id,
                "source_partition": "train",
                "augmentation_index": 0,
                "label": source.label,
                "prompt": "paired original source image",
                "validation_split": validation_split,
                "config_fingerprint": original_fingerprint,
                "filter_status": "keep",
            }
        )
        pair_records.append(
            {
                "pair_index": pair_index,
                "label": source.label,
                "source_id": source_id,
                "original_record_id": original_record_id,
                "generated_record_id": str(generated["record_id"]),
                "generated_image_path": str(
                    Path(str(generated["resolved_image_path"])).resolve()
                ),
                "original_image_path": str(original_path.resolve()),
            }
        )

    common_summary = {
        "ablation": "alia_paired_quality",
        "dataset": dataset,
        "pair_count": len(selected),
        "seed": seed,
        "source_artifact": str(source_artifact),
        "validation_split": validation_split,
    }
    _publish_final_artifact(
        generated_root,
        generated_records,
        {**common_summary, "subset": "generated"},
    )
    _publish_final_artifact(
        original_root,
        original_records,
        {**common_summary, "subset": "original"},
    )
    atomic_write_jsonl(destination / "pairs.jsonl", pair_records)
    summary = {
        **common_summary,
        "generated_artifact": str(generated_root),
        "original_artifact": str(original_root),
        "pairs_manifest": str(destination / "pairs.jsonl"),
    }
    atomic_write_json(destination / "summary.json", summary)

    return summary
