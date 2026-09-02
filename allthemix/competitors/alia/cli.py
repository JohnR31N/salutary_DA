"""Standalone staged CLI for the ALIA offline competitor."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive AllTheMix and ImageFolder source options."""
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", type=str, default="")
    sources.add_argument("--train-dir", type=str, default="")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Dataset identity recorded when --train-dir is used.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help=(
            "Optional fraction excluded from this offline source. Formal "
            "val_source=test runs use 0 and generate from full official train."
        ),
    )


def _add_torch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add backend options shared by foundation-model inference stages."""
    parser.add_argument(
        "--device",
        choices=("auto", "xla", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--xla-cache-dir", type=str, default="")


def build_parser() -> argparse.ArgumentParser:
    """Build the ALIA command tree without importing Torch or TensorFlow."""
    parser = argparse.ArgumentParser(
        description=(
            "Run official-style ALIA caption, prompt, edit, filter, and JAX "
            "training-artifact stages."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prompts = commands.add_parser("prompts", help="Build a prompt artifact.")
    prompts.add_argument("--dataset", type=str, required=True)
    prompts.add_argument(
        "--mode",
        choices=("release", "paper", "generic", "request", "response"),
        default="release",
    )
    prompts.add_argument("--output", type=str, required=True)
    prompts.add_argument("--captions", type=str, default="")
    prompts.add_argument("--response", type=str, default="")
    prompts.add_argument("--request-output", type=str, default="")
    prompts.add_argument(
        "--prefix",
        type=str,
        default="a photo of a {class_name} bird",
    )
    prompts.add_argument("--seed", type=int, default=0)

    caption = commands.add_parser("caption", help="Caption train sources.")
    _add_source_arguments(caption)
    _add_torch_arguments(caption)
    caption.add_argument("--output-dir", type=str, required=True)
    caption.add_argument(
        "--model-id",
        type=str,
        default="Salesforce/blip-image-captioning-large",
    )
    caption.add_argument("--model-revision", type=str, default="")
    caption.add_argument("--max-new-tokens", type=int, default=64)
    caption.add_argument("--max-examples", type=int, default=-1)
    caption.add_argument("--log-every", type=int, default=100)

    edit = commands.add_parser("edit", help="Generate Stable Diffusion edits.")
    _add_source_arguments(edit)
    _add_torch_arguments(edit)
    edit.add_argument("--prompts", type=str, required=True)
    edit.add_argument("--output-dir", type=str, required=True)
    edit.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    edit.add_argument("--model-revision", type=str, default="")
    edit.add_argument("--strength", type=float, default=0.6)
    edit.add_argument("--guidance-scale", type=float, default=7.5)
    edit.add_argument("--num-inference-steps", type=int, default=50)
    edit.add_argument("--images-per-prompt", type=int, default=2)
    edit.add_argument("--generation-size", type=int, default=512)
    edit.add_argument(
        "--source-resize",
        choices=("auto", "native", "center_crop", "letterbox"),
        default="auto",
    )
    edit.add_argument(
        "--compact-size",
        type=int,
        default=0,
        help="Optional saved PNG size; generation still occurs at full size.",
    )
    edit.add_argument("--seed", type=int, default=0)
    edit.add_argument("--max-examples", type=int, default=-1)
    edit.add_argument("--num-shards", type=int, default=1)
    edit.add_argument("--shard-index", type=int, default=0)
    edit.add_argument("--xla-launch", action="store_true")
    edit.add_argument("--attention-slicing", action="store_true")
    edit.add_argument("--keep-safety-checker", action="store_true")
    edit.add_argument("--log-every", type=int, default=50)

    clip = commands.add_parser("clip", help="Run the semantic filter.")
    _add_torch_arguments(clip)
    clip.add_argument("--artifact-dir", type=str, required=True)
    clip.add_argument("--prompts", type=str, required=True)
    clip.add_argument(
        "--model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
    )
    clip.add_argument("--model-revision", type=str, default="")
    clip.add_argument("--batch-size", type=int, default=16)

    score = commands.add_parser(
        "score",
        help="Score originals and edits with a JAX baseline checkpoint.",
    )
    score.add_argument("--config", type=str, required=True)
    score.add_argument("--checkpoint", type=str, required=True)
    score.add_argument("--artifact-dir", type=str, required=True)
    score.add_argument("--batch-size", type=int, default=64)
    score.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=True)
    score.add_argument(
        "--input-stage",
        choices=("generated", "clip"),
        default="clip",
    )

    filter_parser = commands.add_parser(
        "filter",
        help="Publish the accepted JAX training manifest.",
    )
    filter_parser.add_argument("--artifact-dir", type=str, required=True)
    filter_parser.add_argument("--num-classes", type=int, required=True)
    filter_parser.add_argument("--extra-ratio", type=float, default=1.0)
    filter_parser.add_argument(
        "--max-per-source",
        type=int,
        default=1,
        help="Keep one accepted edit per source as in the official loader.",
    )
    filter_parser.add_argument("--seed", type=int, default=0)
    filter_parser.add_argument(
        "--require-semantic-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    strict_filter = commands.add_parser(
        "strict-filter",
        help="Publish a separate class-fidelity-first ALIA artifact.",
    )
    strict_filter.add_argument("--artifact-dir", type=str, required=True)
    strict_filter.add_argument("--output-dir", type=str, required=True)
    strict_filter.add_argument("--num-classes", type=int, required=True)
    strict_filter.add_argument(
        "--min-assigned-probability",
        type=float,
        default=0.2,
    )
    strict_filter.add_argument("--per-class", type=int, default=5)
    strict_filter.add_argument("--max-per-source", type=int, default=1)
    strict_filter.add_argument("--max-records", type=int, default=-1)
    strict_filter.add_argument("--seed", type=int, default=0)
    strict_filter.add_argument(
        "--require-semantic-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    strict_filter.add_argument(
        "--exclude-too-easy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    import_official = commands.add_parser(
        "import-official",
        help="Import the public ALIA CUB W&B artifact without source leakage.",
    )
    import_official.add_argument("--artifact-dir", type=str, required=True)
    import_official.add_argument("--output-dir", type=str, required=True)
    import_official.add_argument(
        "--dataset",
        type=str,
        default="caltech_birds2011",
    )
    import_official.add_argument("--data-dir", type=str, default="./data")
    import_official.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
    )
    import_official.add_argument(
        "--artifact-ref",
        type=str,
        default="clipinvariance/ALIA/cub_generic:v0",
    )
    import_official.add_argument(
        "--prompt",
        type=str,
        default="see official W&B artifact metadata",
    )
    import_official.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    verify = commands.add_parser("verify", help="Validate a final manifest.")
    verify.add_argument("--artifact-dir", type=str, required=True)
    verify.add_argument("--dataset", type=str, required=True)
    verify.add_argument("--num-classes", type=int, required=True)
    verify.add_argument("--validation-split", type=float, default=None)

    paired_ablation = commands.add_parser(
        "paired-ablation",
        help=(
            "Build matched generated-only and source-original-only training "
            "artifacts."
        ),
    )
    paired_ablation.add_argument("--artifact-dir", type=str, required=True)
    paired_ablation.add_argument("--output-dir", type=str, required=True)
    paired_ablation.add_argument("--dataset", type=str, required=True)
    paired_ablation.add_argument("--data-dir", type=str, default="./data")
    paired_ablation.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
    )
    paired_ablation.add_argument("--max-records", type=int, default=1000)
    paired_ablation.add_argument("--seed", type=int, default=0)

    visualize = commands.add_parser(
        "visualize",
        help="Visualize ranked filtered edits beside their source images.",
    )
    visualize.add_argument("--artifact-dir", type=str, required=True)
    visualize.add_argument("--output", type=str, required=True)
    visualize.add_argument("--dataset", type=str, required=True)
    visualize.add_argument("--data-dir", type=str, default="./data")
    visualize.add_argument("--validation-split", type=float, default=0.0)
    visualize.add_argument("--num-samples", type=int, default=24)
    visualize.add_argument(
        "--ranking",
        choices=("best", "worst"),
        default="best",
    )
    visualize.add_argument("--max-per-class", type=int, default=1)
    visualize.add_argument("--pairs-per-row", type=int, default=3)
    visualize.add_argument("--tile-size", type=int, default=224)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject ambiguous source, sharding, and numerical settings early."""
    if hasattr(args, "train_dir") and args.train_dir and not args.dataset_name:
        parser.error("--dataset-name is required with --train-dir.")
    if (
        hasattr(args, "validation_split")
        and args.validation_split is not None
        and (
            not math.isfinite(args.validation_split)
            or not 0.0 <= args.validation_split < 1.0
        )
    ):
        parser.error("--validation-split must be finite and in [0, 1).")
    if args.command == "caption":
        if args.max_new_tokens < 1 or args.log_every < 1:
            parser.error("caption token and logging counts must be positive.")
        if args.max_examples == 0 or args.max_examples < -1:
            parser.error("--max-examples must be -1 or positive.")
    if args.command == "edit":
        if args.images_per_prompt < 1 or args.num_inference_steps < 1:
            parser.error("edit image and inference-step counts must be positive.")
        if args.generation_size < 8 or args.generation_size % 8:
            parser.error("--generation-size must be a positive multiple of 8.")
        if args.compact_size < 0:
            parser.error("--compact-size must be nonnegative.")
        if args.num_shards < 0:
            parser.error("--num-shards must be nonnegative.")
        if args.num_shards > 0 and not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, num-shards).")
        if args.xla_launch and args.device not in {"auto", "xla"}:
            parser.error("--xla-launch requires --device auto or xla.")
    if args.command == "clip" and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.command == "filter":
        if args.num_classes < 1:
            parser.error("--num-classes must be positive.")
        if not math.isfinite(args.extra_ratio):
            parser.error("--extra-ratio must be finite.")
        if args.max_per_source == 0 or args.max_per_source < -1:
            parser.error(
                "--max-per-source must be positive or -1 for no cap."
            )
    if args.command == "strict-filter":
        if args.num_classes < 1:
            parser.error("--num-classes must be positive.")
        if (
            not math.isfinite(args.min_assigned_probability)
            or not 0.0 <= args.min_assigned_probability <= 1.0
        ):
            parser.error("--min-assigned-probability must be in [0, 1].")
        if args.per_class == 0 or args.per_class < -1:
            parser.error("--per-class must be positive or -1 for no cap.")
        if args.max_per_source == 0 or args.max_per_source < -1:
            parser.error("--max-per-source must be positive or -1 for no cap.")
        if args.max_records == 0 or args.max_records < -1:
            parser.error("--max-records must be positive or -1 for no cap.")
    if args.command == "import-official":
        if args.dataset.strip().lower() != "caltech_birds2011":
            parser.error(
                "cub_generic:v0 can only be imported as "
                "caltech_birds2011."
            )
        if not args.artifact_ref.strip():
            parser.error("--artifact-ref must be nonempty.")
    if args.command == "paired-ablation":
        if args.max_records == 0 or args.max_records < -1:
            parser.error("--max-records must be positive or -1 for all.")
    if args.command == "visualize":
        if args.num_samples < 1 or args.pairs_per_row < 1:
            parser.error("visualization sample and row counts must be positive.")
        if args.max_per_class == 0 or args.max_per_class < -1:
            parser.error("--max-per-class must be positive or -1 for no cap.")
        if args.tile_size < 64:
            parser.error("--tile-size must be at least 64.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate one ALIA stage command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    return args


def _dataset_name(args: argparse.Namespace) -> str:
    """Return the artifact dataset identity for either source adapter."""
    return args.dataset or args.dataset_name


def _sources(args: argparse.Namespace):
    """Construct a lazy source iterator after model initialization."""
    from allthemix.competitors.alia.stages import iter_sources

    return iter_sources(
        dataset=args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        validation_split=args.validation_split,
        download=bool(getattr(args, "source_download", True)),
    )


def _run_prompts(args: argparse.Namespace) -> dict[str, object]:
    """Create the requested immutable prompt artifact."""
    from allthemix.competitors.alia.stages import create_prompt_artifact

    return create_prompt_artifact(
        dataset=args.dataset,
        output_path=args.output,
        mode=args.mode,
        captions_path=args.captions,
        response_path=args.response,
        request_path=args.request_output,
        prefix=args.prefix,
        seed=args.seed,
    )


def _run_caption(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize BLIP and caption only the classifier train partition."""
    from allthemix.competitors.alia.captioning import BlipCaptioner, CaptionerConfig
    from allthemix.competitors.alia.stages import caption_dataset

    captioner = BlipCaptioner(
        CaptionerConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            xla_cache_dir=args.xla_cache_dir,
        )
    )
    count = caption_dataset(
        captioner=captioner,
        sources=_sources(args),
        output_dir=args.output_dir,
        dataset=_dataset_name(args),
        validation_split=args.validation_split,
        max_examples=args.max_examples,
        log_every=args.log_every,
    )

    return {"captioned_sources": count}


def _resolve_runtime_shard(
    args: argparse.Namespace,
    editor,
) -> tuple[int, int]:
    """Resolve manual coordinates or the launched XLA worker ordinal."""
    if args.num_shards > 0:
        return args.shard_index, args.num_shards
    if editor.runtime.device_kind != "xla":
        raise ValueError("num_shards=0 requires an XLA runtime.")
    import torch_xla.runtime as xr

    return int(xr.global_ordinal()), int(xr.world_size())


def _run_edit(
    args: argparse.Namespace,
    worker_index: int | None = None,
) -> dict[str, int]:
    """Initialize Stable Diffusion and generate one deterministic shard."""
    from allthemix.competitors.alia.editor import (
        EditorConfig,
        StableDiffusionImg2ImgEditor,
    )
    from allthemix.competitors.alia.stages import generate_edits

    xla_cache_dir = args.xla_cache_dir
    if xla_cache_dir and worker_index is not None:
        xla_cache_dir = str(
            Path(xla_cache_dir).expanduser() / f"rank-{worker_index:05d}"
        )
    editor = StableDiffusionImg2ImgEditor(
        EditorConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            disable_safety_checker=not args.keep_safety_checker,
            attention_slicing=args.attention_slicing,
            xla_cache_dir=xla_cache_dir,
        )
    )
    shard_index, num_shards = _resolve_runtime_shard(args, editor)
    print(
        f"[alia] worker={worker_index} uses shard="
        f"{shard_index + 1}/{num_shards}; source_download="
        f"{bool(getattr(args, 'source_download', True))}",
        flush=True,
    )

    return generate_edits(
        editor=editor,
        sources=_sources(args),
        prompt_path=args.prompts,
        output_dir=args.output_dir,
        dataset=_dataset_name(args),
        validation_split=args.validation_split,
        images_per_prompt=args.images_per_prompt,
        generation_size=args.generation_size,
        source_resize=args.source_resize,
        compact_size=args.compact_size,
        seed=args.seed,
        max_examples=args.max_examples,
        shard_index=shard_index,
        num_shards=num_shards,
        log_every=args.log_every,
    )


def _xla_edit_worker(worker_index: int, args: argparse.Namespace) -> None:
    """Run one edit shard created by torch_xla.launch."""
    print(
        f"[alia] XLA worker={worker_index} pid={os.getpid()} starting",
        flush=True,
    )
    _run_edit(args, worker_index=worker_index)
    print(
        f"[alia] XLA worker={worker_index} pid={os.getpid()} completed",
        flush=True,
    )


def _run_clip(args: argparse.Namespace) -> dict[str, int]:
    """Score all generated images using the prompt artifact semantics."""
    from allthemix.competitors.alia.clip_filter import ClipConfig, ClipSemanticScorer
    from allthemix.competitors.alia.prompts import read_prompt_payload
    from allthemix.competitors.alia.stages import score_clip_stage

    payload = read_prompt_payload(args.prompts)
    scorer = ClipSemanticScorer(
        config=ClipConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            xla_cache_dir=args.xla_cache_dir,
        ),
        positive_prompts=tuple(payload["semantic_prompts"]),
        negative_prompts=tuple(payload["negative_prompts"]),
    )

    return score_clip_stage(
        scorer=scorer,
        artifact_dir=args.artifact_dir,
        batch_size=args.batch_size,
    )


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch non-launched ALIA stages with lazy heavy imports."""
    if args.command == "prompts":
        return _run_prompts(args)
    if args.command == "caption":
        return _run_caption(args)
    if args.command == "edit":
        return _run_edit(args)
    if args.command == "clip":
        return _run_clip(args)
    if args.command == "score":
        from allthemix.competitors.alia.scoring import score_checkpoint_stage

        return score_checkpoint_stage(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            artifact_dir=args.artifact_dir,
            batch_size=args.batch_size,
            distributed=args.distributed,
            input_stage=args.input_stage,
        )
    if args.command == "filter":
        from allthemix.competitors.alia.stages import filter_scored_stage

        return filter_scored_stage(
            artifact_dir=args.artifact_dir,
            num_classes=args.num_classes,
            extra_ratio=args.extra_ratio,
            seed=args.seed,
            require_semantic_pass=args.require_semantic_pass,
            max_per_source=args.max_per_source,
        )
    if args.command == "strict-filter":
        from allthemix.competitors.alia.stages import (
            publish_strict_filtered_stage,
        )

        return publish_strict_filtered_stage(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            min_assigned_probability=args.min_assigned_probability,
            per_class=args.per_class,
            max_per_source=args.max_per_source,
            max_records=args.max_records,
            seed=args.seed,
            require_semantic_pass=args.require_semantic_pass,
            exclude_too_easy=args.exclude_too_easy,
        )
    if args.command == "import-official":
        from allthemix.competitors.alia.official_artifact import (
            import_official_cub_artifact,
        )

        return import_official_cub_artifact(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            data_dir=args.data_dir,
            validation_split=args.validation_split,
            artifact_ref=args.artifact_ref,
            prompt=args.prompt,
            overwrite=args.overwrite,
        )
    if args.command == "verify":
        from allthemix.competitors.alia.manifest import validate_manifest_for_training

        count = validate_manifest_for_training(
            manifest_path=args.artifact_dir,
            dataset=args.dataset,
            num_classes=args.num_classes,
            validation_split=args.validation_split,
            check_images=True,
        )

        return {"valid_training_records": count}
    if args.command == "paired-ablation":
        from allthemix.competitors.alia.ablation import (
            build_paired_ablation_artifacts,
        )

        return build_paired_ablation_artifacts(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            data_dir=args.data_dir,
            validation_split=args.validation_split,
            max_records=args.max_records,
            seed=args.seed,
        )
    if args.command == "visualize":
        from allthemix.competitors.alia.visualize import (
            visualize_filtered_quality,
        )

        return visualize_filtered_quality(
            artifact_dir=args.artifact_dir,
            output_path=args.output,
            dataset=args.dataset,
            data_dir=args.data_dir,
            validation_split=args.validation_split,
            num_samples=args.num_samples,
            ranking=args.ranking,
            max_per_class=args.max_per_class,
            pairs_per_row=args.pairs_per_row,
            tile_size=args.tile_size,
        )

    raise ValueError(f"Unsupported ALIA command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    """Run one offline stage and print a machine-readable summary."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    args = parse_args(argv)
    if args.command in {"caption", "edit"}:
        from allthemix.competitors.generative.source_preflight import (
            prepare_generation_dataset,
        )

        prepare_generation_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
        )
        args.source_download = False
    if args.command == "edit" and args.xla_launch:
        from allthemix.competitors.generative.torch_xla import import_torch_xla

        args.device = "xla"
        args.num_shards = 0
        import_torch_xla().launch(_xla_edit_worker, args=(args,))
        summary: dict[str, Any] = {"launched_xla_workers": True}
    else:
        summary = _run_command(args)
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
