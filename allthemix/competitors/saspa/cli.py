"""Standalone official-style SaSPA generation and filtering CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from allthemix.competitors.generative.artifacts import (
    atomic_write_json,
    config_fingerprint,
)
from allthemix.competitors.saspa.prompts import (
    PROMPT_SOURCES,
    SEMANTIC_PROMPTS,
    SUPERCLASSES,
    load_official_prompts,
    normalize_dataset_name,
    semantic_negative_prompts,
)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Require the canonical loader whose indices JAX training reuses."""
    parser.add_argument("--dataset", type=str, required=True)
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
    """Add foundation-model runtime options."""
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
    """Build the command tree without importing Torch, TensorFlow, or JAX."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SaSPA's official BLIP-ControlNet generation, CLIP semantic "
            "filter, classifier top-k filter, and JAX training artifact."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prompts = commands.add_parser(
        "prompts",
        help="Export the official committed scene prompts.",
    )
    prompts.add_argument("--dataset", type=str, required=True)
    prompts.add_argument("--prompt-file", type=str, default="")
    prompts.add_argument("--output", type=str, required=True)

    generate = commands.add_parser(
        "generate",
        help="Generate subject- and structure-preserving images.",
    )
    _add_source_arguments(generate)
    _add_torch_arguments(generate)
    generate.add_argument("--output-dir", type=str, required=True)
    generate.add_argument("--prompt-file", type=str, default="")
    generate.add_argument(
        "--model-id",
        type=str,
        default="Salesforce/blipdiffusion-controlnet",
    )
    generate.add_argument("--model-revision", type=str, default="")
    generate.add_argument("--guidance-scale", type=float, default=7.5)
    generate.add_argument(
        "--num-inference-steps",
        type=int,
        default=0,
        help="0 selects official defaults: CUB=30, Cars=50.",
    )
    generate.add_argument("--images-per-source", type=int, default=2)
    generate.add_argument("--generation-size", type=int, default=512)
    generate.add_argument(
        "--source-resize",
        choices=("auto", "official", "center_crop", "letterbox"),
        default="auto",
    )
    generate.add_argument("--compact-size", type=int, default=0)
    generate.add_argument("--canny-low-threshold", type=int, default=120)
    generate.add_argument("--canny-high-threshold", type=int, default=200)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--max-examples", type=int, default=-1)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--xla-launch", action="store_true")
    generate.add_argument("--log-every", type=int, default=50)

    semantic = commands.add_parser(
        "semantic",
        help="Apply the official positive-vs-negative CLIP filter.",
    )
    _add_torch_arguments(semantic)
    semantic.add_argument("--artifact-dir", type=str, required=True)
    semantic.add_argument("--dataset", type=str, required=True)
    semantic.add_argument("--model-name", type=str, default="RN50")
    semantic.add_argument("--clip-download-root", type=str, default="")
    semantic.add_argument("--batch-size", type=int, default=16)

    filter_parser = commands.add_parser(
        "filter",
        help="Apply the official baseline-classifier top-k filter.",
    )
    filter_parser.add_argument("--config", type=str, required=True)
    filter_parser.add_argument("--checkpoint", type=str, required=True)
    filter_parser.add_argument("--artifact-dir", type=str, required=True)
    filter_parser.add_argument("--top-k", type=int, default=10)
    filter_parser.add_argument("--batch-size", type=int, default=64)
    filter_parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    filter_parser.add_argument(
        "--require-semantic-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    verify = commands.add_parser(
        "verify",
        help="Validate the final filtered JAX training artifact.",
    )
    verify.add_argument("--artifact-dir", type=str, required=True)
    verify.add_argument("--dataset", type=str, required=True)
    verify.add_argument("--num-classes", type=int, required=True)
    verify.add_argument("--validation-split", type=float, default=None)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject ambiguous or paper-incompatible settings before model load."""
    if hasattr(args, "dataset") and args.dataset:
        normalize_dataset_name(args.dataset)
    if (
        hasattr(args, "validation_split")
        and args.validation_split is not None
        and (
            not math.isfinite(args.validation_split)
            or not 0.0 <= args.validation_split < 1.0
        )
    ):
        parser.error("--validation-split must be finite and in [0, 1).")
    if args.command == "generate":
        if args.images_per_source < 1 or args.log_every < 1:
            parser.error("generation and logging counts must be positive.")
        if args.num_inference_steps < 0:
            parser.error("--num-inference-steps must be nonnegative.")
        if args.generation_size < 64 or args.generation_size % 64:
            parser.error("--generation-size must be a multiple of 64.")
        if args.compact_size < 0:
            parser.error("--compact-size must be nonnegative.")
        if args.max_examples == 0 or args.max_examples < -1:
            parser.error("--max-examples must be -1 or positive.")
        if args.num_shards < 0:
            parser.error("--num-shards must be nonnegative.")
        if args.num_shards > 0 and not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, num-shards).")
        if args.xla_launch and args.device not in {"auto", "xla"}:
            parser.error("--xla-launch requires --device auto or xla.")
        if not (
            0
            <= args.canny_low_threshold
            < args.canny_high_threshold
            <= 255
        ):
            parser.error("Canny thresholds must satisfy 0 <= low < high <= 255.")
    if args.command == "semantic" and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.command == "filter" and (
        args.top_k < 1 or args.batch_size < 1
    ):
        parser.error("--top-k and --batch-size must be positive.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one validated SaSPA stage invocation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    return args


def _dataset_name(args: argparse.Namespace) -> str:
    """Return the canonical dataset identity."""
    return normalize_dataset_name(args.dataset)


def _sources(args: argparse.Namespace):
    """Create the lazy train-side source iterator after model setup."""
    from allthemix.competitors.generative.sources import iter_allthemix_sources

    return iter_allthemix_sources(
        dataset=args.dataset,
        data_dir=args.data_dir,
        validation_split=args.validation_split,
        download=bool(getattr(args, "source_download", True)),
    )


def _run_prompts(args: argparse.Namespace) -> dict[str, Any]:
    """Publish a self-describing copy of the exact official prompts."""
    dataset = normalize_dataset_name(args.dataset)
    prompts = load_official_prompts(dataset, args.prompt_file)
    payload = {
        "method": "saspa",
        "dataset": dataset,
        "source": PROMPT_SOURCES[dataset],
        "superclass": SUPERCLASSES[dataset],
        "prompts": list(prompts),
        "semantic_prompts": list(SEMANTIC_PROMPTS[dataset]),
        "negative_prompts": list(semantic_negative_prompts(dataset)),
        "prompt_fingerprint": config_fingerprint(prompts),
    }
    atomic_write_json(args.output, payload)

    return payload


def _runtime_shard(args: argparse.Namespace, editor) -> tuple[int, int]:
    """Resolve manual shard coordinates or launched XLA process ordinals."""
    if args.num_shards > 0:
        return args.shard_index, args.num_shards
    if editor.runtime.device_kind != "xla":
        raise ValueError("num_shards=0 requires an XLA runtime.")
    import torch_xla.runtime as xr

    return int(xr.global_ordinal()), int(xr.world_size())


def _run_generate(
    args: argparse.Namespace,
    worker_index: int | None = None,
) -> dict[str, Any]:
    """Initialize BLIP-ControlNet and generate one deterministic shard."""
    from allthemix.competitors.saspa.editor import EditorConfig, SaSPAEditor
    from allthemix.competitors.saspa.generation import generate_images

    dataset = _dataset_name(args)
    xla_cache_dir = args.xla_cache_dir
    if xla_cache_dir and worker_index is not None:
        xla_cache_dir = str(
            Path(xla_cache_dir).expanduser() / f"rank-{worker_index:05d}"
        )
    steps = args.num_inference_steps
    if steps == 0:
        steps = 50 if dataset == "cars196" else 30
    editor = SaSPAEditor(
        EditorConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            guidance_scale=args.guidance_scale,
            num_inference_steps=steps,
            xla_cache_dir=xla_cache_dir,
        )
    )
    shard_index, num_shards = _runtime_shard(args, editor)
    print(
        f"[saspa] worker={worker_index} uses shard="
        f"{shard_index + 1}/{num_shards}; source_download="
        f"{bool(getattr(args, 'source_download', True))}",
        flush=True,
    )

    return generate_images(
        editor=editor,
        sources=_sources(args),
        prompts=load_official_prompts(dataset, args.prompt_file),
        output_dir=args.output_dir,
        dataset=dataset,
        validation_split=args.validation_split,
        superclass=SUPERCLASSES[dataset],
        images_per_source=args.images_per_source,
        generation_size=args.generation_size,
        source_resize=args.source_resize,
        compact_size=args.compact_size,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        seed=args.seed,
        max_examples=args.max_examples,
        shard_index=shard_index,
        num_shards=num_shards,
        log_every=args.log_every,
    )


def _xla_generate_worker(worker_index: int, args: argparse.Namespace) -> None:
    """Run one generation shard under ``torch_xla.launch``."""
    print(
        f"[saspa] XLA worker={worker_index} pid={os.getpid()} starting",
        flush=True,
    )
    _run_generate(args, worker_index=worker_index)
    print(
        f"[saspa] XLA worker={worker_index} pid={os.getpid()} completed",
        flush=True,
    )


def _run_semantic(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize CLIP and apply SaSPA's official semantic filter."""
    from allthemix.competitors.saspa.clip_filter import (
        SaSPAClipConfig,
        SaSPAClipSemanticScorer,
    )
    from allthemix.competitors.saspa.filtering import score_semantic_stage

    dataset = normalize_dataset_name(args.dataset)
    scorer = SaSPAClipSemanticScorer(
        config=SaSPAClipConfig(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            download_root=args.clip_download_root,
            xla_cache_dir=args.xla_cache_dir,
        ),
        positive_prompts=SEMANTIC_PROMPTS[dataset],
        negative_prompts=semantic_negative_prompts(dataset),
    )

    return score_semantic_stage(
        scorer=scorer,
        artifact_dir=args.artifact_dir,
        batch_size=args.batch_size,
    )


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch one non-launched SaSPA stage."""
    if args.command == "prompts":
        return _run_prompts(args)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "semantic":
        return _run_semantic(args)
    if args.command == "filter":
        from allthemix.competitors.saspa.filtering import (
            filter_classifier_stage,
        )

        return filter_classifier_stage(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            artifact_dir=args.artifact_dir,
            top_k=args.top_k,
            batch_size=args.batch_size,
            distributed=args.distributed,
            require_semantic_pass=args.require_semantic_pass,
        )
    if args.command == "verify":
        from allthemix.competitors.saspa.manifest import (
            validate_manifest_for_training,
        )

        count = validate_manifest_for_training(
            manifest_path=args.artifact_dir,
            dataset=args.dataset,
            num_classes=args.num_classes,
            validation_split=args.validation_split,
            check_images=True,
        )

        return {"valid_training_records": count}

    raise ValueError(f"Unsupported SaSPA command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    """Run one stage and print its machine-readable summary."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    args = parse_args(argv)
    if args.command == "generate":
        from allthemix.competitors.generative.source_preflight import (
            prepare_generation_dataset,
        )

        prepare_generation_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
        )
        args.source_download = False
    if args.command == "generate" and args.xla_launch:
        from allthemix.competitors.generative.torch_xla import import_torch_xla

        args.device = "xla"
        args.num_shards = 0
        import_torch_xla().launch(_xla_generate_worker, args=(args,))
        summary: dict[str, Any] = {"launched_xla_workers": True}
    else:
        summary = _run_command(args)
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
