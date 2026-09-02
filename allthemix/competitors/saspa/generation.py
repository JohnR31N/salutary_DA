"""Deterministic, resumable SaSPA image generation."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from allthemix.competitors.generative.artifacts import (
    append_jsonl_record,
    atomic_write_json,
    config_fingerprint,
    sha256_file,
    stable_seed,
)
from allthemix.competitors.generative.sources import SourceExample
from allthemix.competitors.saspa.geometry import (
    make_canny_image,
    prepare_image,
    resolve_resize_mode,
)
from allthemix.competitors.saspa.manifest import (
    SASPA_SCHEMA_VERSION,
    stage_manifest_name,
)
from allthemix.competitors.saspa.prompts import (
    OFFICIAL_SASPA_REPOSITORY,
    format_scene_prompt,
)


def _plain_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove reader-only absolute paths before republishing a record."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"resolved_image_path", "manifest_path"}
    }


def _load_existing_records(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Read a possibly incomplete shard without requiring a summary."""
    if not manifest_path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    import json

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Cannot resume a truncated SaSPA manifest at "
                    f"{manifest_path}:{line_number}."
                ) from error
            record_id = str(record.get("record_id", ""))
            if not record_id or record_id in records:
                raise ValueError(
                    "Cannot resume SaSPA with a missing or duplicate "
                    f"record_id at {manifest_path}:{line_number}."
                )
            records[record_id] = record

    return records


def _save_png(image: Image.Image, path: Path, compact_size: int) -> None:
    """Save one RGB result atomically, optionally compacting after inference."""
    output = image.convert("RGB")
    if compact_size > 0:
        output = output.resize(
            (compact_size, compact_size),
            Image.Resampling.LANCZOS,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    output.save(temporary, format="PNG", optimize=True)
    temporary.replace(path)


def _choose(
    values: list[Any] | tuple[Any, ...],
    seed: int,
) -> Any:
    """Choose one value using a local RNG that cannot perturb other jobs."""
    if not values:
        raise ValueError("SaSPA cannot choose from an empty collection.")

    return values[random.Random(seed).randrange(len(values))]


def generate_images(
    editor,
    sources: Iterable[SourceExample],
    prompts: tuple[str, ...],
    output_dir: str | Path,
    dataset: str,
    validation_split: float,
    superclass: str,
    images_per_source: int = 2,
    generation_size: int = 512,
    source_resize: str = "auto",
    compact_size: int = 0,
    canny_low_threshold: int = 120,
    canny_high_threshold: int = 200,
    seed: int = 0,
    max_examples: int = -1,
    shard_index: int = 0,
    num_shards: int = 1,
    log_every: int = 50,
) -> dict[str, Any]:
    """Generate one official-style SaSPA shard and publish completion state."""
    if images_per_source < 1 or log_every < 1:
        raise ValueError("SaSPA generation and logging counts must be positive.")
    if max_examples == 0 or max_examples < -1:
        raise ValueError("SaSPA max_examples must be -1 or positive.")
    if compact_size < 0:
        raise ValueError("SaSPA compact_size must be nonnegative.")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid SaSPA generation shard coordinates.")
    if not prompts:
        raise ValueError("SaSPA generation requires at least one prompt.")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resize_mode = resolve_resize_mode(
        source_resize,
        editor.runtime.device_kind,
    )
    source_list = list(sources)
    if max_examples > 0:
        source_list = source_list[:max_examples]
    if not source_list:
        raise ValueError("SaSPA source partition is empty.")

    by_label: dict[int, list[SourceExample]] = defaultdict(list)
    for source in source_list:
        by_label[source.label].append(source)
    missing_references = [
        source.label for source in source_list if not by_label[source.label]
    ]
    if missing_references:
        raise ValueError(
            "SaSPA could not build same-class reference pools for labels "
            f"{sorted(set(missing_references))}."
        )

    config = {
        "schema_version": SASPA_SCHEMA_VERSION,
        "method": "saspa",
        "official_repository": OFFICIAL_SASPA_REPOSITORY,
        "dataset": dataset,
        "validation_split": float(validation_split),
        "superclass": superclass,
        "prompt_fingerprint": config_fingerprint(prompts),
        "model_id": editor.config.model_id,
        "model_commit": editor.resolved_model_commit,
        "guidance_scale": float(editor.config.guidance_scale),
        "num_inference_steps": int(editor.config.num_inference_steps),
        "negative_prompt": editor.config.negative_prompt,
        "images_per_source": images_per_source,
        "generation_size": generation_size,
        "source_resize": resize_mode,
        "compact_size": compact_size,
        "canny_low_threshold": canny_low_threshold,
        "canny_high_threshold": canny_high_threshold,
        "seed": seed,
        "max_examples": max_examples,
    }
    fingerprint = config_fingerprint(config)
    manifest_path = root / stage_manifest_name(
        "generated",
        shard_index=shard_index,
        num_shards=num_shards,
    )
    summary_name = (
        "generated_summary.json"
        if num_shards == 1
        else (
            f"generated_summary-{shard_index:05d}-of-"
            f"{num_shards:05d}.json"
        )
    )
    summary_path = root / summary_name
    existing = _load_existing_records(manifest_path)
    for record in existing.values():
        if record.get("config_fingerprint") != fingerprint:
            raise ValueError(
                "SaSPA output directory contains records from a different "
                f"configuration: {manifest_path}."
            )

    jobs = [
        (source, augmentation_index)
        for source in source_list
        if source.index % num_shards == shard_index
        for augmentation_index in range(images_per_source)
    ]
    atomic_write_json(
        summary_path,
        {
            "complete": False,
            "stage": "generated",
            "record_count": len(existing),
            "expected_record_count": len(jobs),
            "config_fingerprint": fingerprint,
            "shard_index": shard_index,
            "num_shards": num_shards,
        },
    )

    completed = 0
    skipped = 0
    for source, augmentation_index in jobs:
        record_id = (
            f"saspa:{dataset}:{source.index:09d}:{augmentation_index:03d}"
        )
        if record_id in existing:
            existing_record = existing[record_id]
            image_path = root / str(existing_record["image_path"])
            if (
                image_path.is_file()
                and sha256_file(image_path)
                == existing_record.get("output_png_sha256")
            ):
                skipped += 1
                continue

            raise ValueError(
                "SaSPA resume found a manifest row without its valid image: "
                f"{record_id}."
            )

        job_seed = stable_seed(
            seed,
            dataset,
            source.source_id,
            augmentation_index,
        )
        prompt_index = stable_seed(job_seed, "prompt") % len(prompts)
        scene_prompt = prompts[prompt_index]
        prompt = format_scene_prompt(
            scene_prompt,
            class_name=source.class_name,
            superclass=superclass,
        )
        reference = _choose(
            by_label[source.label],
            stable_seed(job_seed, "reference"),
        )
        prepared_source = prepare_image(
            source.image,
            generation_size=generation_size,
            mode=resize_mode,
        )
        prepared_reference = prepare_image(
            reference.image,
            generation_size=generation_size,
            mode=resize_mode,
        )
        conditioning = make_canny_image(
            prepared_source,
            low_threshold=canny_low_threshold,
            high_threshold=canny_high_threshold,
        )
        generated = editor.edit(
            reference_image=prepared_reference,
            conditioning_image=conditioning,
            prompt=prompt,
            superclass=superclass,
            seed=job_seed,
            width=prepared_source.width,
            height=prepared_source.height,
        )
        relative_path = Path("images") / (
            f"{source.label:04d}"
        ) / f"{source.index:09d}-{augmentation_index:03d}.png"
        image_path = root / relative_path
        _save_png(generated, image_path, compact_size=compact_size)
        record = {
            "schema_version": SASPA_SCHEMA_VERSION,
            "stage": "generated",
            "method": "saspa",
            "record_id": record_id,
            "image_path": relative_path.as_posix(),
            "output_png_sha256": sha256_file(image_path),
            "dataset": dataset,
            "source_id": source.source_id,
            "source_index": source.index,
            "source_ref": source.source_ref,
            "source_partition": "train",
            "reference_source_id": reference.source_id,
            "reference_source_index": reference.index,
            "reference_source_ref": reference.source_ref,
            "reference_same_class": reference.label == source.label,
            "augmentation_index": augmentation_index,
            "label": source.label,
            "class_name": source.class_name,
            "superclass": superclass,
            "scene_prompt": scene_prompt,
            "prompt": prompt,
            "prompt_index": prompt_index,
            "seed": job_seed,
            "validation_split": float(validation_split),
            "generation_width": prepared_source.width,
            "generation_height": prepared_source.height,
            "saved_width": generated.width if compact_size == 0 else compact_size,
            "saved_height": generated.height if compact_size == 0 else compact_size,
            "source_resize": resize_mode,
            "canny_low_threshold": canny_low_threshold,
            "canny_high_threshold": canny_high_threshold,
            "model_id": editor.config.model_id,
            "model_commit": editor.resolved_model_commit,
            "guidance_scale": float(editor.config.guidance_scale),
            "num_inference_steps": int(editor.config.num_inference_steps),
            "config_fingerprint": fingerprint,
        }
        append_jsonl_record(manifest_path, _plain_record(record))
        completed += 1
        if completed % log_every == 0 or completed + skipped == len(jobs):
            print(
                f"SaSPA generation shard {shard_index + 1}/{num_shards}: "
                f"{completed + skipped}/{len(jobs)} jobs complete"
            )

    editor.synchronize()
    final_count = len(existing) + completed
    if final_count != len(jobs):
        raise RuntimeError(
            "SaSPA generation did not publish every assigned job: "
            f"{final_count}/{len(jobs)}."
        )
    summary = {
        "complete": True,
        "stage": "generated",
        "record_count": final_count,
        "expected_record_count": len(jobs),
        "generated_records": completed,
        "resumed_records": skipped,
        "source_count": len(source_list),
        "config_fingerprint": fingerprint,
        "config": config,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    atomic_write_json(summary_path, summary)

    return summary
