"""Resumable offline stages for the ALIA competitor pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from allthemix.competitors.alia.captioning import BlipCaptioner
from allthemix.competitors.alia.clip_filter import ClipSemanticScorer
from allthemix.competitors.alia.editor import StableDiffusionImg2ImgEditor
from allthemix.competitors.alia.filtering import (
    filter_generated_records,
    strict_filter_generated_records,
)
from allthemix.competitors.alia.manifest import (
    ALIA_SCHEMA_VERSION,
    read_stage_records,
    stage_manifest_name,
)
from allthemix.competitors.alia.prompts import (
    build_prompt_request,
    format_prompt,
    generic_prompt_payload,
    paper_prompt_payload,
    parse_prompt_response,
    read_prompt_payload,
    release_prompt_payload,
    semantic_prompt_payload,
    write_prompt_payload,
)
from allthemix.competitors.generative.artifacts import (
    append_jsonl_record,
    append_jsonl_records,
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    sha256_file,
    stable_seed,
)
from allthemix.competitors.generative.sources import (
    SourceExample,
    iter_allthemix_sources,
    iter_class_folder_sources,
)


def iter_sources(
    dataset: str,
    data_dir: str,
    train_dir: str,
    validation_split: float,
    download: bool = True,
) -> Iterator[SourceExample]:
    """Select the shared AllTheMix or class-folder source adapter."""
    if bool(dataset) == bool(train_dir):
        raise ValueError(
            "Specify exactly one of dataset or train_dir for ALIA sources."
        )
    if dataset:
        return iter_allthemix_sources(
            dataset=dataset,
            data_dir=data_dir,
            validation_split=validation_split,
            download=download,
        )

    return iter_class_folder_sources(
        train_dir=train_dir,
        validation_split=validation_split,
    )


def _source_resize_mode(mode: str, device_kind: str) -> str:
    """Use native official geometry unless XLA needs a static square."""
    value = mode.lower()
    if value == "auto":
        return "center_crop" if device_kind == "xla" else "native"
    if value not in {"native", "center_crop", "letterbox"}:
        raise ValueError(
            "source_resize must be auto, native, center_crop, or letterbox."
        )

    return value


def prepare_source_image(
    image: Image.Image,
    generation_size: int,
    mode: str,
) -> Image.Image:
    """Prepare stable diffusion input while making geometry explicit."""
    if generation_size < 8 or generation_size % 8 != 0:
        raise ValueError("generation_size must be a positive multiple of 8.")
    rgb = image.convert("RGB")
    if mode == "native":
        width = max(8, rgb.width - rgb.width % 8)
        height = max(8, rgb.height - rgb.height % 8)
        if (width, height) != rgb.size:
            rgb = rgb.crop((0, 0, width, height))

        return rgb
    if mode == "center_crop":
        return ImageOps.fit(
            rgb,
            (generation_size, generation_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    contained = ImageOps.contain(
        rgb,
        (generation_size, generation_size),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (generation_size, generation_size), (127, 127, 127))
    offset = (
        (generation_size - contained.width) // 2,
        (generation_size - contained.height) // 2,
    )
    canvas.paste(contained, offset)

    return canvas


def _atomic_save_png(image: Image.Image, output_path: Path) -> str:
    """Save a complete RGB PNG before exposing its final path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}.png"
    )
    image.convert("RGB").save(temporary_path, format="PNG")
    temporary_path.replace(output_path)

    return sha256_file(output_path)


def _record_id(source_id: str, prompt_index: int, variant_index: int) -> str:
    """Build a compact stable identity for one generated edit."""
    digest = hashlib.sha256(
        f"{source_id}\0{prompt_index}\0{variant_index}".encode()
    ).hexdigest()

    return f"alia-{digest[:24]}"


def _read_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read one in-progress shard and reject malformed duplicate jobs."""
    if not path.is_file():
        return {}
    records = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid in-progress ALIA JSON at {path}:{line_number}."
            ) from error
        record_id = str(record.get("record_id", ""))
        if not record_id or record_id in records:
            raise ValueError(
                f"Duplicate or empty ALIA record_id at {path}:{line_number}."
            )
        records[record_id] = record

    return records


def _relative_output_path(output_dir: Path, record_id: str) -> Path:
    """Spread generated files across stable prefix directories."""
    return Path("images") / record_id[5:7] / f"{record_id}.png"


def _write_stage_summary(
    output_dir: Path,
    stage: str,
    complete: bool,
    record_count: int,
    **extra: Any,
) -> None:
    """Publish one atomic completion marker for an unsharded stage."""
    atomic_write_json(
        output_dir / f"{stage}_summary.json",
        {
            "complete": bool(complete),
            "stage": stage,
            "record_count": int(record_count),
            **extra,
        },
    )


def caption_dataset(
    captioner: BlipCaptioner,
    sources: Iterable[SourceExample],
    output_dir: str | Path,
    dataset: str,
    validation_split: float,
    max_examples: int = -1,
    log_every: int = 100,
) -> int:
    """Caption train-partition sources and durably resume by source ID."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "captions.jsonl"
    existing = _read_existing_records(manifest_path)
    # Caption records use source_id as their stage identity.
    completed = {
        str(record.get("source_id")): record
        for record in existing.values()
    }
    config = {
        "schema_version": 1,
        "stage": "caption",
        "dataset": dataset,
        "validation_split": validation_split,
        "model_id": captioner.config.model_id,
        "model_commit": captioner.resolved_model_commit,
        "max_new_tokens": captioner.config.max_new_tokens,
    }
    fingerprint = config_fingerprint(config)
    config["config_fingerprint"] = fingerprint
    config_path = root / "caption_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                "Cannot resume ALIA captions with changed settings. Choose "
                "a new output directory."
            )
    else:
        atomic_write_json(config_path, config)
    _write_stage_summary(root, "captions", False, len(completed))

    count = 0
    for source_position, source in enumerate(sources):
        if max_examples > 0 and source_position >= max_examples:
            break
        if source.source_id in completed:
            count += 1
            continue
        caption = captioner.caption(source.image)
        record = {
            "stage_id": source.source_id,
            "source_id": source.source_id,
            "source_ref": source.source_ref,
            "source_partition": "train",
            "dataset": dataset,
            "label": source.label,
            "class_name": source.class_name,
            "caption": caption,
            "validation_split": validation_split,
            "config_fingerprint": fingerprint,
        }
        # _read_existing_records expects record_id for generic resume parsing.
        record["record_id"] = source.source_id
        append_jsonl_record(manifest_path, record)
        completed[source.source_id] = record
        count += 1
        if count % log_every == 0:
            print(f"ALIA caption: {count} train sources complete")

    captioner.synchronize()
    _write_stage_summary(root, "captions", True, count)

    return count


def read_captions(path: str | Path) -> list[dict[str, Any]]:
    """Read caption records without importing BLIP or Torch."""
    caption_path = Path(path)
    if caption_path.is_dir():
        caption_path = caption_path / "captions.jsonl"
    records = list(_read_existing_records(caption_path).values())
    if not records:
        raise ValueError(f"ALIA caption artifact is empty: {caption_path}")

    return records


def create_prompt_artifact(
    dataset: str,
    output_path: str | Path,
    mode: str = "release",
    captions_path: str | Path = "",
    response_path: str | Path = "",
    request_path: str | Path = "",
    prefix: str = "a photo of a {class_name} bird",
    seed: int = 0,
) -> dict[str, object]:
    """Create a fixed prompt preset or parse an LLM response."""
    mode_name = mode.strip().lower()
    if mode_name == "release":
        payload = release_prompt_payload(dataset)
        write_prompt_payload(output_path, payload)

        return payload
    if mode_name == "paper":
        payload = paper_prompt_payload(dataset)
        write_prompt_payload(output_path, payload)

        return payload
    if mode_name == "generic":
        payload = generic_prompt_payload(dataset)
        write_prompt_payload(output_path, payload)

        return payload

    if mode_name not in {"request", "response"}:
        raise ValueError(
            "prompt mode must be release, paper, generic, request, or "
            "response."
        )
    if not captions_path:
        raise ValueError(f"prompt mode {mode_name!r} requires captions_path.")
    captions = [
        str(record["caption"])
        for record in read_captions(captions_path)
    ]
    request = build_prompt_request(
        captions=captions,
        prefix=prefix,
        seed=seed,
    )
    if mode_name == "request":
        destination = Path(request_path or output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(request + "\n", encoding="utf-8")

        return {
            "dataset": dataset,
            "prompt_source": "request",
            "request_path": str(destination.resolve()),
            "seed": seed,
        }

    if not response_path:
        raise ValueError("prompt mode 'response' requires response_path.")
    response = Path(response_path).read_text(encoding="utf-8")
    semantic_prompts, negative_prompts = semantic_prompt_payload(dataset)
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "alia",
        "dataset": dataset,
        "prompt_source": "llm-response",
        "prompts": list(parse_prompt_response(response, prefix=prefix)),
        "semantic_prompts": list(semantic_prompts),
        "negative_prompts": list(negative_prompts),
        "prompt_request": request,
        "prompt_discovery_seed": seed,
    }
    payload["prompt_fingerprint"] = config_fingerprint(payload)
    write_prompt_payload(output_path, payload)

    return payload


def generate_edits(
    editor: StableDiffusionImg2ImgEditor,
    sources: Iterable[SourceExample],
    prompt_path: str | Path,
    output_dir: str | Path,
    dataset: str,
    validation_split: float,
    images_per_prompt: int = 2,
    generation_size: int = 512,
    source_resize: str = "auto",
    compact_size: int = 0,
    seed: int = 0,
    max_examples: int = -1,
    shard_index: int = 0,
    num_shards: int = 1,
    log_every: int = 50,
) -> dict[str, int]:
    """Generate all prompt edits for assigned train sources with resume."""
    if images_per_prompt < 1:
        raise ValueError("images_per_prompt must be positive.")
    if compact_size < 0:
        raise ValueError("compact_size must be nonnegative.")
    prompt_payload = read_prompt_payload(prompt_path)
    if str(prompt_payload.get("dataset", "")).lower() != dataset.lower():
        raise ValueError(
            "ALIA prompt artifact dataset mismatch: "
            f"{prompt_payload.get('dataset')!r} vs {dataset!r}."
        )
    prompts = tuple(str(value) for value in prompt_payload["prompts"])
    resize_mode = _source_resize_mode(
        source_resize,
        editor.runtime.device_kind,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": ALIA_SCHEMA_VERSION,
        "method": "alia",
        "stage": "generated",
        "dataset": dataset,
        "validation_split": validation_split,
        "prompt_fingerprint": prompt_payload["prompt_fingerprint"],
        "images_per_prompt": images_per_prompt,
        "generation_size": generation_size,
        "source_resize": resize_mode,
        "compact_size": compact_size,
        "seed": seed,
        "editor_model_id": editor.config.model_id,
        "editor_model_commit": editor.resolved_model_commit,
        "strength": editor.config.strength,
        "guidance_scale": editor.config.guidance_scale,
        "num_inference_steps": editor.config.num_inference_steps,
    }
    fingerprint = config_fingerprint(config)
    run_config = {**config, "config_fingerprint": fingerprint}
    run_config_path = root / "run_config.json"
    if run_config_path.is_file():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise ValueError(
                "Cannot resume ALIA editing with changed settings. Choose a "
                "new output directory."
            )
    else:
        atomic_write_json(run_config_path, run_config)

    manifest_path = root / stage_manifest_name(
        "generated",
        shard_index=shard_index,
        num_shards=num_shards,
    )
    existing = _read_existing_records(manifest_path)
    valid_existing = {}
    for record_id, record in existing.items():
        image_path = root / str(record.get("image_path", ""))
        if (
            record.get("config_fingerprint") == fingerprint
            and image_path.is_file()
            and sha256_file(image_path) == record.get("output_png_sha256")
        ):
            valid_existing[record_id] = record
    if len(valid_existing) != len(existing):
        atomic_write_jsonl(
            manifest_path,
            sorted(valid_existing.values(), key=lambda item: item["record_id"]),
        )
    completed = valid_existing
    summary_name = (
        "generated_summary.json"
        if num_shards == 1
        else f"generated_summary-{shard_index:05d}-of-{num_shards:05d}.json"
    )
    atomic_write_json(
        root / summary_name,
        {
            "complete": False,
            "stage": "generated",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "record_count": len(completed),
            "config_fingerprint": fingerprint,
        },
    )

    counters = {"generated": 0, "resumed": 0, "sources": 0}
    expected_ids = set()
    for source_position, source in enumerate(sources):
        if max_examples > 0 and source_position >= max_examples:
            break
        if source.index % num_shards != shard_index:
            continue
        counters["sources"] += 1
        prepared = prepare_source_image(
            source.image,
            generation_size=generation_size,
            mode=resize_mode,
        )
        for prompt_index, template in enumerate(prompts):
            prompt = format_prompt(template, source.class_name)
            for variant_index in range(images_per_prompt):
                record_id = _record_id(
                    source.source_id,
                    prompt_index,
                    variant_index,
                )
                expected_ids.add(record_id)
                if record_id in completed:
                    counters["resumed"] += 1
                    continue
                job_seed = stable_seed(
                    seed,
                    "alia-edit",
                    source.source_id,
                    prompt_index,
                    variant_index,
                )
                generated = editor.edit(prepared, prompt=prompt, seed=job_seed)
                if compact_size > 0:
                    generated = generated.resize(
                        (compact_size, compact_size),
                        Image.Resampling.BILINEAR,
                    )
                relative_path = _relative_output_path(root, record_id)
                output_path = root / relative_path
                digest = _atomic_save_png(generated, output_path)
                record = {
                    "schema_version": ALIA_SCHEMA_VERSION,
                    "method": "alia",
                    "stage": "generated",
                    "record_id": record_id,
                    "image_path": relative_path.as_posix(),
                    "output_png_sha256": digest,
                    "dataset": dataset,
                    "source_id": source.source_id,
                    "source_ref": source.source_ref,
                    "source_partition": "train",
                    "source_index": source.index,
                    "source_width": source.image.width,
                    "source_height": source.image.height,
                    "label": source.label,
                    "class_name": source.class_name,
                    "augmentation_index": (
                        prompt_index * images_per_prompt + variant_index
                    ),
                    "prompt_index": prompt_index,
                    "variant_index": variant_index,
                    "prompt_template": template,
                    "prompt": prompt,
                    "seed": job_seed,
                    "validation_split": validation_split,
                    "editor_model_id": editor.config.model_id,
                    "editor_model_commit": editor.resolved_model_commit,
                    "strength": editor.config.strength,
                    "guidance_scale": editor.config.guidance_scale,
                    "num_inference_steps": editor.config.num_inference_steps,
                    "source_resize": resize_mode,
                    "config_fingerprint": fingerprint,
                }
                append_jsonl_record(manifest_path, record)
                completed[record_id] = record
                counters["generated"] += 1
                done = counters["generated"] + counters["resumed"]
                if done % log_every == 0:
                    print(
                        f"ALIA edit shard {shard_index + 1}/{num_shards}: "
                        f"{done} jobs complete"
                    )

    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise ValueError(
            "ALIA output contains jobs no longer present in the source or "
            f"prompt catalog: {unexpected[:5]}."
        )
    editor.synchronize()
    atomic_write_json(
        root / summary_name,
        {
            "complete": True,
            "stage": "generated",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "record_count": len(completed),
            "config_fingerprint": fingerprint,
            **counters,
        },
    )

    return counters


def score_clip_stage(
    scorer: ClipSemanticScorer,
    artifact_dir: str | Path,
    batch_size: int,
) -> dict[str, int]:
    """Attach CLIP decisions with durable batch-level resume support."""
    if batch_size < 1:
        raise ValueError("ALIA CLIP batch_size must be positive.")
    root = Path(artifact_dir).expanduser().resolve()
    records = read_stage_records(
        root,
        stage="generated",
        check_images=True,
        require_complete=True,
    )
    output_path = root / stage_manifest_name("clip")
    in_progress_path = root / ".clip.inprogress.jsonl"
    config_path = root / "clip_config.json"
    input_digest = hashlib.sha256()
    for record in records:
        input_digest.update(str(record["record_id"]).encode("utf-8"))
        input_digest.update(b"\0")
        input_digest.update(str(record["output_png_sha256"]).encode("ascii"))
        input_digest.update(b"\n")
    clip_config = {
        "schema_version": 1,
        "stage": "clip",
        "input_record_count": len(records),
        "input_fingerprint": input_digest.hexdigest(),
        "model_id": scorer.config.model_id,
        "model_commit": scorer.resolved_model_commit,
        "model_revision": getattr(scorer.config, "model_revision", ""),
        "logit_scale": float(getattr(scorer.config, "logit_scale", 100.0)),
        "batch_size": int(batch_size),
        "positive_prompts": list(scorer.positive_prompts),
        "negative_prompts": list(scorer.negative_prompts),
    }
    clip_config["config_fingerprint"] = config_fingerprint(clip_config)
    if config_path.is_file():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != clip_config:
            raise ValueError(
                "Cannot resume ALIA CLIP scoring with changed inputs or "
                "settings. Remove the incomplete CLIP stage or choose a new "
                "artifact directory."
            )
    else:
        atomic_write_json(config_path, clip_config)

    expected_ids = [str(record["record_id"]) for record in records]
    if output_path.is_file():
        completed = _read_existing_records(output_path)
        completed_ids = list(completed)
        if completed_ids != expected_ids:
            raise ValueError(
                "Completed ALIA CLIP manifest does not match the generated "
                "record order."
            )
        kept = sum(
            int(bool(record.get("semantic_pass")))
            for record in completed.values()
        )
        _write_stage_summary(
            root,
            "clip",
            True,
            len(completed),
            semantic_pass=kept,
            semantic_reject=len(completed) - kept,
            clip_model_id=scorer.config.model_id,
            clip_model_commit=scorer.resolved_model_commit,
        )
        print(f"ALIA CLIP: {len(completed)}/{len(records)} images already scored")
        return {
            "records": len(completed),
            "semantic_pass": kept,
            "semantic_reject": len(completed) - kept,
        }

    completed = _read_existing_records(in_progress_path)
    completed_ids = list(completed)
    if completed_ids != expected_ids[: len(completed_ids)]:
        raise ValueError(
            "Incomplete ALIA CLIP manifest is not a prefix of the generated "
            "record order."
        )
    kept = sum(
        int(bool(record.get("semantic_pass")))
        for record in completed.values()
    )
    _write_stage_summary(
        root,
        "clip",
        False,
        len(completed),
        semantic_pass=kept,
        semantic_reject=len(completed) - kept,
    )
    if completed:
        print(
            f"ALIA CLIP: resuming from {len(completed)}/{len(records)} "
            "durably scored images"
        )

    for start in range(len(completed), len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = []
        for record in batch_records:
            with Image.open(record["resolved_image_path"]) as image:
                images.append(image.convert("RGB").copy())
        scores = scorer.score(images)
        if len(scores) != len(batch_records):
            raise ValueError(
                "ALIA CLIP scorer returned a different number of scores "
                "than input images."
            )
        output_records = []
        for record, score in zip(batch_records, scores):
            updated = {
                key: value
                for key, value in record.items()
                if key not in {"resolved_image_path", "manifest_path"}
            }
            updated.update(score)
            updated["stage"] = "clip"
            kept += int(bool(score["semantic_pass"]))
            output_records.append(updated)
        append_jsonl_records(in_progress_path, output_records)
        completed_count = start + len(output_records)
        _write_stage_summary(
            root,
            "clip",
            False,
            completed_count,
            semantic_pass=kept,
            semantic_reject=completed_count - kept,
        )
        print(
            f"ALIA CLIP: {min(start + batch_size, len(records))}/"
            f"{len(records)} images scored"
        )
    scorer.synchronize()
    in_progress_path.replace(output_path)
    _write_stage_summary(
        root,
        "clip",
        True,
        len(records),
        semantic_pass=kept,
        semantic_reject=len(records) - kept,
        clip_model_id=scorer.config.model_id,
        clip_model_commit=scorer.resolved_model_commit,
    )

    return {
        "records": len(records),
        "semantic_pass": kept,
        "semantic_reject": len(records) - kept,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a nonempty JSONL stage artifact with precise diagnostics."""
    if not path.is_file():
        raise FileNotFoundError(f"ALIA stage artifact does not exist: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid ALIA JSON at {path}:{line_number}."
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"ALIA JSONL rows must be objects at {path}:{line_number}."
            )
        records.append(record)
    if not records:
        raise ValueError(f"ALIA stage artifact is empty: {path}")

    return records


def filter_scored_stage(
    artifact_dir: str | Path,
    num_classes: int,
    extra_ratio: float = 1.0,
    seed: int = 0,
    require_semantic_pass: bool = True,
    max_per_source: int = 1,
) -> dict[str, Any]:
    """Publish the immutable ALIA training manifest after all filters."""
    root = Path(artifact_dir).expanduser().resolve()
    classifier_records = read_stage_records(
        root,
        stage="classifier",
        check_images=True,
        require_complete=True,
    )
    base_scores = _read_jsonl(root / "base_scores.jsonl")
    accepted, summary = filter_generated_records(
        records=classifier_records,
        base_scores=base_scores,
        num_classes=num_classes,
        extra_ratio=extra_ratio,
        seed=seed,
        require_semantic_pass=require_semantic_pass,
        max_per_source=max_per_source,
    )
    if not accepted:
        raise ValueError(
            "ALIA filtering rejected every generated image; no training "
            "manifest was published."
        )

    final_records = []
    for record in accepted:
        final_record = {
            key: value
            for key, value in record.items()
            if key not in {"resolved_image_path", "manifest_path"}
        }
        final_record["stage"] = "final"
        final_record["filter_status"] = "keep"
        final_records.append(final_record)
    final_records.sort(key=lambda record: str(record["record_id"]))

    manifest_path = root / stage_manifest_name("final")
    summary_path = root / "final_summary.json"
    atomic_write_json(
        summary_path,
        {
            "complete": False,
            "stage": "final",
            "kept_records": 0,
        },
    )
    atomic_write_jsonl(manifest_path, final_records)
    final_summary = {
        "complete": True,
        "stage": "final",
        **summary,
        "kept_records": len(final_records),
        "num_classes": num_classes,
        "seed": seed,
        "require_semantic_pass": require_semantic_pass,
        "max_per_source": max_per_source,
    }
    atomic_write_json(summary_path, final_summary)

    return final_summary


def publish_strict_filtered_stage(
    artifact_dir: str | Path,
    output_dir: str | Path,
    num_classes: int,
    min_assigned_probability: float = 0.2,
    per_class: int = 5,
    max_per_source: int = 1,
    max_records: int = -1,
    seed: int = 0,
    require_semantic_pass: bool = True,
    exclude_too_easy: bool = True,
) -> dict[str, Any]:
    """Publish a separate class-fidelity-first ALIA training artifact."""
    root = Path(artifact_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == root:
        raise ValueError(
            "Strict ALIA output_dir must differ from artifact_dir so the "
            "official-style manifest remains immutable."
        )
    classifier_records = read_stage_records(
        root,
        stage="classifier",
        check_images=True,
        require_complete=True,
    )
    base_scores = _read_jsonl(root / "base_scores.jsonl")
    accepted, summary = strict_filter_generated_records(
        records=classifier_records,
        base_scores=base_scores,
        num_classes=num_classes,
        min_assigned_probability=min_assigned_probability,
        per_class=per_class,
        max_per_source=max_per_source,
        max_records=max_records,
        seed=seed,
        require_semantic_pass=require_semantic_pass,
        exclude_too_easy=exclude_too_easy,
    )
    if not accepted:
        raise ValueError(
            "Strict ALIA filtering rejected every generated image; no "
            "training manifest was published."
        )

    strict_config = {
        "exclude_too_easy": exclude_too_easy,
        "max_per_source": max_per_source,
        "min_assigned_probability": min_assigned_probability,
        "max_records": max_records,
        "num_classes": num_classes,
        "per_class": per_class,
        "require_semantic_pass": require_semantic_pass,
        "seed": seed,
        "source_artifact": str(root),
        "stage": "strict_filter",
    }
    strict_fingerprint = config_fingerprint(strict_config)
    final_records = []
    for record in accepted:
        final_record = {
            key: value
            for key, value in record.items()
            if key not in {"resolved_image_path", "manifest_path"}
        }
        final_record["image_path"] = str(
            Path(str(record["resolved_image_path"])).resolve()
        )
        final_record["stage"] = "final"
        final_record["filter_status"] = "keep"
        final_record["filter_reason"] = "strict_keep"
        final_record["strict_filter_fingerprint"] = strict_fingerprint
        final_records.append(final_record)
    final_records.sort(key=lambda record: str(record["record_id"]))

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / stage_manifest_name("final")
    summary_path = destination / "final_summary.json"
    atomic_write_json(
        summary_path,
        {
            "complete": False,
            "stage": "final",
            "kept_records": 0,
        },
    )
    atomic_write_jsonl(manifest_path, final_records)
    final_summary = {
        "complete": True,
        "stage": "final",
        **summary,
        "kept_records": len(final_records),
        "num_classes": num_classes,
        "source_artifact": str(root),
        "strict_filter": True,
        "strict_filter_fingerprint": strict_fingerprint,
    }
    atomic_write_json(summary_path, final_summary)

    return final_summary
