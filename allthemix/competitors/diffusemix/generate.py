"""Generate a DiffuseMix training set with PyTorch, including PyTorch/XLA."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from allthemix.competitors.diffusemix.compat import import_torch_xla
from allthemix.competitors.diffusemix.compose import (
    DEFAULT_PROMPTS,
    OFFICIAL_RELEASE_PROMPTS,
    build_instruction,
    compose_diffusemix,
    is_near_black,
    masks_for_mode,
)
from allthemix.competitors.diffusemix.editor import (
    DEFAULT_MODEL_ID,
    EditorConfig,
    InstructPix2PixEditor,
)
from allthemix.competitors.diffusemix.manifest import MANIFEST_SCHEMA_VERSION
from allthemix.competitors.diffusemix.sources import (
    SourceExample,
    iter_allthemix_sources,
    iter_class_folder_sources,
)

_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}
_COMPACT_OUTPUT_SIZES = {
    "cifar10": 32,
    "cifar100": 32,
    "svhn_cropped": 32,
    "tiny_imagenet": 64,
    "stl10": 96,
    "caltech_birds2011": 224,
    "cars196": 224,
    "imagenet100": 224,
    "oxford_iiit_pet": 224,
}
_RESUME_PROVENANCE_FIELDS = (
    "class_name",
    "job_seed",
    "label",
    "prompt",
    "source_id",
    "source_image_sha256",
    "source_index",
    "source_ref",
)


def _resolve_output_size(
    dataset_name: str,
    generation_size: int,
    compact_output: bool,
) -> int:
    """Resolve the square PNG size without importing TensorFlow metadata."""
    if not compact_output:
        return generation_size

    normalized_name = dataset_name.strip().lower()
    output_size = _COMPACT_OUTPUT_SIZES.get(
        normalized_name,
    )
    if output_size is None:
        supported = ", ".join(
            sorted(
                _COMPACT_OUTPUT_SIZES,
            )
        )
        raise ValueError(
            "--compact-output cannot infer the classifier input size for "
            f"dataset {dataset_name!r}. Supported datasets: {supported}."
        )
    if output_size > generation_size:
        raise ValueError(
            "--compact-output cannot enlarge generated images: dataset "
            f"{normalized_name!r} needs {output_size}px, but "
            f"--generation-size is {generation_size}px."
        )

    return output_size


def _parse_prompts(
    value: str,
) -> tuple[str, ...]:
    prompts = tuple(
        prompt.strip()
        for prompt in value.split(
            ",",
        )
        if prompt.strip()
    )
    if not prompts:
        raise argparse.ArgumentTypeError(
            "prompts must contain at least one comma-separated value."
        )

    return prompts


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse the standalone generation command."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline DiffuseMix dataset for AllTheMix JAX "
            "training."
        )
    )
    parser.add_argument(
        "--preset",
        choices=(
            "paper",
            "official_release",
        ),
        default="paper",
        help=(
            "paper uses four hard masks and templated random prompts; "
            "official_release uses the 512px release-code behavior."
        ),
    )
    source_group = parser.add_mutually_exclusive_group(
        required=True,
    )
    source_group.add_argument(
        "--dataset",
        type=str,
        default="",
        help="AllTheMix dataset name (supports TFDS and local loaders).",
    )
    source_group.add_argument(
        "--train-dir",
        type=str,
        default="",
        help="ImageFolder-style train/<class> image directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help=(
            "AllTheMix dataset name recorded for --train-dir sources; "
            "required when using --train-dir."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--fractal-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help=(
            "Optional fraction excluded from the generation source itself. "
            "Formal val_source=test runs leave this at 0 so every official "
            "training image remains eligible; this value is artifact "
            "provenance and is independent of the classifier's official-"
            "evaluation validation fraction."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=_parse_prompts,
        default=None,
        help="Comma-separated filter-like prompts.",
    )
    parser.add_argument(
        "--prompt-policy",
        choices=(
            "random",
            "all",
        ),
        default=None,
        help=(
            "random follows the paper algorithm; all follows the released "
            "code's loop over every prompt."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Use '{prompt}' to insert each filter-like prompt.",
    )
    parser.add_argument(
        "--images-per-source",
        type=int,
        default=1,
        help=(
            "Number of random-prompt images per source, or number per prompt "
            "when prompt-policy=all."
        ),
    )
    parser.add_argument(
        "--generation-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help=(
            "Generate and compose at --generation-size, then resize each "
            "saved PNG to the dataset's classifier input size."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=(
            "paper",
            "official_code",
        ),
        default=None,
    )
    parser.add_argument(
        "--seam-width",
        type=int,
        default=-1,
        help=(
            "Center transition width. Defaults to 0 for paper mode and 20 "
            "for official_code mode."
        ),
    )
    parser.add_argument(
        "--fractal-alpha",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default="",
        help="Optional Hugging Face model revision/commit to pin.",
    )
    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "xla",
            "cuda",
            "cpu",
        ),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=(
            "auto",
            "float32",
            "float16",
            "bfloat16",
        ),
        default="auto",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--keep-safety-checker",
        action="store_true",
        help="Keep the model's safety checker (the released code disables it).",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
    )
    parser.add_argument(
        "--xla-cache-dir",
        type=str,
        default="",
        help="Optional persistent XLA compilation-cache root.",
    )
    parser.add_argument(
        "--xla-launch",
        action="store_true",
        help=(
            "Launch one generator process per local XLA device with "
            "torch_xla.launch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--black-pixel-fraction",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--allow-generation-failures",
        action="store_true",
        help=(
            "Return success even when near-black outputs remain after all "
            "retries."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=-1,
        help="Limit source examples for a smoke run; -1 means all.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Manual generation shard count. Set 0 to read the XLA runtime "
            "world size and ordinal."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    args = parser.parse_args(
        argv,
    )

    preset_defaults = {
        "paper": {
            "prompts": DEFAULT_PROMPTS,
            "prompt_policy": "random",
            "prompt_template": (
                "A transformed version of image into {prompt}"
            ),
            "generation_size": 512,
            "mask_mode": "paper",
        },
        "official_release": {
            "prompts": OFFICIAL_RELEASE_PROMPTS,
            "prompt_policy": "all",
            "prompt_template": "{prompt}",
            "generation_size": 512,
            "mask_mode": "official_code",
        },
    }[args.preset]
    for name, value in preset_defaults.items():
        if getattr(
            args,
            name,
        ) is None:
            setattr(
                args,
                name,
                value,
            )

    if args.train_dir and not args.dataset_name:
        parser.error("--dataset-name is required with --train-dir.")
    if not math.isfinite(
        args.validation_split,
    ) or not 0.0 <= args.validation_split < 1.0:
        parser.error("--validation-split must be finite and in [0, 1).")
    if args.images_per_source < 1:
        parser.error("--images-per-source must be >= 1.")
    if args.generation_size < 8 or args.generation_size % 8 != 0:
        parser.error("--generation-size must be a positive multiple of 8.")
    try:
        args.output_size = _resolve_output_size(
            dataset_name=args.dataset or args.dataset_name,
            generation_size=args.generation_size,
            compact_output=args.compact_output,
        )
    except ValueError as error:
        parser.error(
            str(
                error,
            )
        )
    if args.seam_width < -1:
        parser.error("--seam-width must be -1 or >= 0.")
    if not 0.0 <= args.fractal_alpha <= 1.0:
        parser.error("--fractal-alpha must be in [0, 1].")
    if not math.isfinite(
        args.guidance_scale,
    ) or args.guidance_scale <= 0.0:
        parser.error("--guidance-scale must be finite and > 0.")
    if (
        not math.isfinite(
            args.image_guidance_scale,
        )
        or args.image_guidance_scale < 1.0
    ):
        parser.error("--image-guidance-scale must be finite and >= 1.")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be >= 1.")
    if args.max_generation_attempts < 1:
        parser.error("--max-generation-attempts must be >= 1.")
    if not 0 <= args.black_threshold <= 255:
        parser.error("--black-threshold must be in [0, 255].")
    if not math.isfinite(
        args.black_pixel_fraction,
    ) or not 0.0 <= args.black_pixel_fraction <= 1.0:
        parser.error("--black-pixel-fraction must be finite and in [0, 1].")
    if args.max_examples == 0 or args.max_examples < -1:
        parser.error("--max-examples must be -1 or >= 1.")
    if args.num_shards < 0:
        parser.error("--num-shards must be >= 0.")
    if args.num_shards > 0 and not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards).")
    if args.log_every < 1:
        parser.error("--log-every must be >= 1.")
    if args.xla_launch and args.device not in {
        "auto",
        "xla",
    }:
        parser.error("--xla-launch requires --device auto or xla.")
    try:
        for prompt in args.prompts:
            build_instruction(
                prompt=prompt,
                template=args.prompt_template,
            )
    except (
        IndexError,
        KeyError,
        ValueError,
    ) as error:
        parser.error(
            f"invalid prompt/template configuration: {error}"
        )

    return args


def _fractal_paths(
    fractal_dir: str,
) -> list[Path]:
    root = Path(
        fractal_dir,
    ).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"DiffuseMix fractal_dir does not exist: {root}"
        )
    paths = sorted(
        path
        for path in root.rglob(
            "*",
        )
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(
            f"No fractal images were found under: {root}"
        )

    return paths


def _stable_seed(
    root_seed: int,
    source_id: str,
    augmentation_index: int,
    attempt: int = 0,
    domain: str = "job",
) -> int:
    value = (
        f"{domain}|{root_seed}|{source_id}|{augmentation_index}|{attempt}"
    ).encode()
    digest = hashlib.blake2b(
        value,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)


def _slug(
    value: str,
) -> str:
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        value.lower(),
    ).strip(
        "-",
    )

    return slug[:40] or "prompt"


def _package_versions() -> dict[str, str]:
    """Record generation-library versions without importing their modules."""
    versions = {}
    for package in (
        "torch",
        "torch-xla",
        "diffusers",
        "huggingface-hub",
        "transformers",
        "accelerate",
        "numpy",
        "Pillow",
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-datasets",
    ):
        try:
            versions[package] = importlib.metadata.version(
                package,
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    return versions


def _sha256_file(
    path: Path,
) -> str:
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


def _sha256_rgb_image(
    image: Image.Image,
) -> str:
    rgb = image.convert(
        "RGB",
    )
    digest = hashlib.sha256()
    digest.update(
        f"RGB:{rgb.width}x{rgb.height}:".encode(
            "ascii",
        )
    )
    digest.update(
        rgb.tobytes(),
    )

    return digest.hexdigest()


def _fractal_catalog(
    fractal_paths: list[Path],
    fractal_root: Path,
) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(
                fractal_root,
            ).as_posix(),
            "sha256": _sha256_file(
                path,
            ),
        }
        for path in fractal_paths
    ]


def _catalog_sha256(
    catalog: list[dict[str, str]],
) -> str:
    return hashlib.sha256(
        json.dumps(
            catalog,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8",
        )
    ).hexdigest()


def _config_dict(
    args: argparse.Namespace,
    seam_width: int,
    fractal_catalog_sha256: str,
) -> dict[str, Any]:
    config = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generation_contract_version": 2,
        "rng_contract": "domain-separated-v1",
        "preset": args.preset,
        "dataset": args.dataset or args.dataset_name,
        "source_kind": "allthemix" if args.dataset else "class_folder",
        "source_split_contract": "class-stratified-index-v1",
        "source": args.dataset or str(
            Path(
                args.train_dir,
            ).expanduser().resolve()
        ),
        "data_dir": args.data_dir if args.dataset else "",
        "validation_split": args.validation_split,
        "prompts": list(
            args.prompts,
        ),
        "prompt_policy": args.prompt_policy,
        "prompt_template": args.prompt_template,
        "images_per_source": args.images_per_source,
        "generation_size": args.generation_size,
        "mask_mode": args.mask_mode,
        "seam_width": seam_width,
        "fractal_alpha": args.fractal_alpha,
        "fractal_catalog_sha256": fractal_catalog_sha256,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "device": args.device,
        "dtype": args.dtype,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "safety_checker_enabled": args.keep_safety_checker,
        "attention_slicing": args.attention_slicing,
        "black_threshold": args.black_threshold,
        "black_pixel_fraction": args.black_pixel_fraction,
        "max_generation_attempts": args.max_generation_attempts,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "package_versions": _package_versions(),
    }
    if args.compact_output:
        config.update(
            {
                "compact_output": True,
                "output_size": args.output_size,
                "output_resize": "pillow-bilinear",
            }
        )

    return config


def _fingerprint(
    config: dict[str, Any],
) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8",
    )

    return hashlib.sha256(
        encoded,
    ).hexdigest()


def _stable_json_value(
    value: Any,
) -> Any:
    """Normalize nested config values without nondeterministic set reprs."""
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _stable_json_value(
                nested_value,
            )
            for key, nested_value in value.items()
        }
    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _stable_json_value(
                nested_value,
            )
            for nested_value in value
        ]
    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        normalized = [
            _stable_json_value(
                nested_value,
            )
            for nested_value in value
        ]

        return sorted(
            normalized,
            key=lambda nested_value: json.dumps(
                nested_value,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
                default=str,
            ),
        )
    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
        ),
    ):
        return value

    return str(
        value,
    )


def _scheduler_behavior_config(
    value: Any,
) -> dict[str, Any]:
    """Return scheduler fields that affect generation semantics."""
    normalized = _stable_json_value(
        value,
    )
    if not isinstance(
        normalized,
        dict,
    ):
        raise TypeError(
            "Scheduler config must normalize to a dictionary."
        )

    # Diffusers records which constructor values came from defaults in this
    # private field. Its contents may differ between concurrent loaders, but
    # it does not change the resolved scheduler values or generated samples.
    normalized.pop(
        "_use_default_values",
        None,
    )

    return normalized


def _normalize_legacy_scheduler_config(
    value: Any,
) -> Any:
    """Recover sets stringified by the pre-stable scheduler serializer."""
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _normalize_legacy_scheduler_config(
                nested_value,
            )
            for key, nested_value in value.items()
        }
    if isinstance(
        value,
        list,
    ):
        return [
            _normalize_legacy_scheduler_config(
                nested_value,
            )
            for nested_value in value
        ]
    if isinstance(
        value,
        str,
    ) and (
        value.startswith(
            "{",
        )
        or value.startswith(
            "set(",
        )
    ):
        try:
            parsed = ast.literal_eval(
                value,
            )
        except (
            SyntaxError,
            ValueError,
        ):
            return value
        if isinstance(
            parsed,
            set,
        ):
            return _stable_json_value(
                parsed,
            )

    return value


def _resolve_config_fingerprint(
    output_dir: Path,
    config: dict[str, Any],
    scheduler_config: dict[str, Any],
) -> str:
    """Keep pre-stable scheduler fingerprints resumable when equivalent."""
    current_fingerprint = _fingerprint(
        config,
    )
    run_config_path = output_dir / "run_config.json"
    if not run_config_path.is_file():
        return current_fingerprint

    try:
        existing = json.loads(
            run_config_path.read_text(
                encoding="utf-8",
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ):
        return current_fingerprint
    if not isinstance(
        existing,
        dict,
    ):
        return current_fingerprint
    existing_fingerprint = existing.get(
        "config_fingerprint",
    )
    existing_config = existing.get(
        "config",
    )
    if (
        not isinstance(
            existing_fingerprint,
            str,
        )
        or not isinstance(
            existing_config,
            dict,
        )
        or _fingerprint(
            existing_config,
        )
        != existing_fingerprint
    ):
        return current_fingerprint
    if existing_fingerprint == current_fingerprint:
        return current_fingerprint

    scheduler_key = "scheduler_config_sha256"
    existing_without_scheduler = dict(
        existing_config,
    )
    current_without_scheduler = dict(
        config,
    )
    legacy_scheduler_fingerprint = existing_without_scheduler.pop(
        scheduler_key,
        None,
    )
    current_without_scheduler.pop(
        scheduler_key,
        None,
    )
    if existing_without_scheduler != current_without_scheduler:
        return current_fingerprint
    if not isinstance(
        legacy_scheduler_fingerprint,
        str,
    ) or re.fullmatch(
        r"[0-9a-fA-F]{64}",
        legacy_scheduler_fingerprint,
    ) is None:
        return current_fingerprint

    runtime = existing.get(
        "runtime",
    )
    if not isinstance(
        runtime,
        dict,
    ):
        return current_fingerprint
    legacy_scheduler_config = runtime.get(
        "scheduler_config",
    )
    if not isinstance(
        legacy_scheduler_config,
        dict,
    ):
        return current_fingerprint
    if _scheduler_behavior_config(
        _normalize_legacy_scheduler_config(
            legacy_scheduler_config,
        )
    ) != _scheduler_behavior_config(
        scheduler_config,
    ):
        return current_fingerprint

    compatible_config = dict(
        config,
    )
    compatible_config[scheduler_key] = legacy_scheduler_fingerprint.lower()
    if _fingerprint(
        compatible_config,
    ) != existing_fingerprint:
        return current_fingerprint

    config[scheduler_key] = legacy_scheduler_fingerprint.lower()
    print(
        "Resuming a DiffuseMix artifact with its equivalent legacy "
        "scheduler fingerprint."
    )

    return existing_fingerprint


def _resolve_shard(
    args: argparse.Namespace,
    editor: InstructPix2PixEditor,
) -> tuple[int, int]:
    if args.num_shards > 0:
        return args.shard_index, args.num_shards
    if editor.device_kind != "xla":
        raise ValueError(
            "num_shards=0 requires device=xla so the runtime ordinal can be "
            "discovered."
        )

    import torch_xla.runtime as xr

    num_shards = int(
        xr.world_size(),
    )
    shard_index = int(
        xr.global_ordinal(),
    )
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise RuntimeError(
            "Invalid PyTorch/XLA runtime shard coordinates: "
            f"ordinal={shard_index}, world_size={num_shards}."
        )

    return shard_index, num_shards


def _manifest_path(
    output_dir: Path,
    shard_index: int,
    num_shards: int,
) -> Path:
    if num_shards == 1:
        return output_dir / "manifest.jsonl"

    return output_dir / (
        f"manifest-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    )


def _guard_manifest_layout(
    output_dir: Path,
    num_shards: int,
) -> None:
    """Avoid duplicate records left by changing the shard topology."""
    single = output_dir / "manifest.jsonl"
    sharded = sorted(
        output_dir.glob(
            "manifest-*-of-*.jsonl",
        )
    )
    if num_shards == 1 and sharded:
        raise ValueError(
            "Output directory already contains sharded manifests. Reuse the "
            "same num_shards or choose a new output directory."
        )
    if num_shards > 1 and single.exists():
        raise ValueError(
            "Output directory already contains an unsharded manifest. Reuse "
            "num_shards=1 or choose a new output directory."
        )

    suffix = f"-of-{num_shards:05d}.jsonl"
    incompatible = [
        path
        for path in sharded
        if not path.name.endswith(
            suffix,
        )
    ]
    if incompatible:
        raise ValueError(
            "Output directory contains manifests from a different shard "
            f"topology: {incompatible[0].name}."
        )


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Durably replace one JSON object without exposing a partial file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(
        path.suffix + f".{os.getpid()}.tmp",
    )
    with temporary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        file.flush()
        os.fsync(
            file.fileno(),
        )
    temporary_path.replace(
        path,
    )


def _count_manifest_records(
    manifest_path: Path,
) -> int:
    """Count nonempty JSONL records for the completion contract."""
    with manifest_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return sum(
            1
            for line in file
            if line.strip()
        )


def _summary_path(
    output_dir: Path,
    shard_index: int,
    num_shards: int,
) -> Path:
    """Return the completion marker for one generation shard."""
    if num_shards == 1:
        return output_dir / "summary.json"

    return output_dir / (
        f"summary-{shard_index:05d}-of-{num_shards:05d}.json"
    )


def _write_generation_summary(
    summary_path: Path,
    manifest_path: Path,
    shard_index: int,
    num_shards: int,
    config_fingerprint: str,
    complete: bool,
    manifest_record_count: int,
    counters: dict[str, int] | None = None,
    accepted_incomplete: bool = False,
    source_catalog_sha256: str = "",
) -> None:
    """Publish or invalidate one shard's atomic completion marker."""
    payload: dict[str, Any] = {
        "complete": complete,
        "accepted_incomplete": accepted_incomplete,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "manifest": manifest_path.name,
        "manifest_record_count": manifest_record_count,
        "config_fingerprint": config_fingerprint,
    }
    if source_catalog_sha256:
        payload["source_catalog_sha256"] = source_catalog_sha256
    if counters is not None:
        payload.update(
            counters,
        )
    _write_json_atomic(
        path=summary_path,
        payload=payload,
    )


def _previous_source_catalog_sha256(
    summary_path: Path,
    config_fingerprint: str,
) -> str:
    """Read the persistent source baseline before invalidating a shard."""
    if not summary_path.is_file():
        return ""
    try:
        payload = json.loads(
            summary_path.read_text(
                encoding="utf-8",
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ) as error:
        raise ValueError(
            "Cannot safely resume because the DiffuseMix summary is "
            f"unreadable: {summary_path}."
        ) from error
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Cannot safely resume because the DiffuseMix summary is not a "
            f"JSON object: {summary_path}."
        )
    if (
        payload.get(
            "config_fingerprint",
        )
        != config_fingerprint
    ):
        raise ValueError(
            "Cannot safely resume because the DiffuseMix summary fingerprint "
            f"does not match this run: {summary_path}."
        )

    catalog_sha256 = payload.get(
        "source_catalog_sha256",
        "",
    )
    if not catalog_sha256:
        if payload.get(
            "complete",
        ) is True:
            raise ValueError(
                "Cannot safely resume because a completed DiffuseMix summary "
                f"has no source catalog baseline: {summary_path}."
            )
        return ""
    if not isinstance(
        catalog_sha256,
        str,
    ) or re.fullmatch(
        r"[0-9a-fA-F]{64}",
        catalog_sha256,
    ) is None:
        raise ValueError(
            "Cannot safely resume because the DiffuseMix source catalog "
            f"digest is invalid: {summary_path}."
        )

    return catalog_sha256.lower()


def _load_completed(
    manifest_path: Path,
    output_dir: Path,
    config_fingerprint: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load complete jobs and repair a trailing partial JSONL write."""
    if not manifest_path.exists():
        return {}

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    temporary_path = manifest_path.with_suffix(
        manifest_path.suffix + f".{os.getpid()}.repair.tmp",
    )
    try:
        with manifest_path.open(
            mode="r",
            encoding="utf-8",
        ) as source_file, temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as repaired_file:
            for line_number, line in enumerate(
                source_file,
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line,
                    )
                except json.JSONDecodeError as error:
                    has_later_record = any(
                        later_line.strip()
                        for later_line in source_file
                    )
                    if has_later_record:
                        raise ValueError(
                            "Invalid JSON in existing manifest: "
                            f"{manifest_path}:{line_number}"
                        ) from error
                    print(
                        "Ignoring interrupted final manifest line: "
                        f"{manifest_path}"
                    )
                    break

                if not isinstance(
                    record,
                    dict,
                ):
                    raise ValueError(
                        "Existing DiffuseMix manifest records must be JSON "
                        f"objects: {manifest_path}:{line_number}."
                    )

                existing_fingerprint = str(
                    record.get(
                        "config_fingerprint",
                        "",
                    )
                )
                if existing_fingerprint != config_fingerprint:
                    raise ValueError(
                        "Existing DiffuseMix manifest was generated with "
                        "different settings. Choose a new output directory."
                    )
                image_path = Path(
                    str(
                        record["image_path"],
                    )
                )
                if not image_path.is_absolute():
                    image_path = output_dir / image_path
                if not image_path.is_file():
                    continue

                expected_output_sha256 = record.get(
                    "output_png_sha256",
                )
                if (
                    not isinstance(
                        expected_output_sha256,
                        str,
                    )
                    or re.fullmatch(
                        r"[0-9a-fA-F]{64}",
                        expected_output_sha256,
                    )
                    is None
                    or _sha256_file(
                        image_path,
                    )
                    != expected_output_sha256.lower()
                ):
                    print(
                        "Regenerating DiffuseMix image with a missing or "
                        f"mismatched checksum: {image_path}"
                    )
                    continue

                key = (
                    str(
                        record["source_id"],
                    ),
                    int(
                        record["augmentation_index"],
                    ),
                )
                if key in completed:
                    raise ValueError(
                        "Duplicate source/augmentation job in existing "
                        f"DiffuseMix manifest: {key}."
                    )
                completed[key] = {
                    field: record.get(
                        field,
                    )
                    for field in _RESUME_PROVENANCE_FIELDS
                }
                repaired_file.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                    )
                    + "\n"
                )

            repaired_file.flush()
            os.fsync(
                repaired_file.fileno(),
            )
        temporary_path.replace(
            manifest_path,
        )
    except Exception:
        temporary_path.unlink(
            missing_ok=True,
        )
        raise

    return completed


def _source_iterator(
    args: argparse.Namespace,
) -> Iterator[SourceExample]:
    if args.dataset:
        return iter_allthemix_sources(
            dataset=args.dataset,
            data_dir=args.data_dir,
            validation_split=args.validation_split,
            download=bool(getattr(args, "source_download", True)),
        )

    return iter_class_folder_sources(
        train_dir=args.train_dir,
        validation_split=args.validation_split,
    )


def _jobs_for_source(
    source: SourceExample,
    args: argparse.Namespace,
) -> Iterator[tuple[int, str, int]]:
    """Yield augmentation index, prompt, and deterministic job seed."""
    if args.prompt_policy == "all":
        for prompt_index, prompt in enumerate(
            args.prompts,
        ):
            for image_index in range(
                args.images_per_source,
            ):
                augmentation_index = (
                    prompt_index * args.images_per_source + image_index
                )
                yield (
                    augmentation_index,
                    prompt,
                    _stable_seed(
                        root_seed=args.seed,
                        source_id=source.source_id,
                        augmentation_index=augmentation_index,
                        domain="composition",
                    ),
                )
        return

    for augmentation_index in range(
        args.images_per_source,
    ):
        job_seed = _stable_seed(
            root_seed=args.seed,
            source_id=source.source_id,
            augmentation_index=augmentation_index,
            domain="composition",
        )
        prompt_seed = _stable_seed(
            root_seed=args.seed,
            source_id=source.source_id,
            augmentation_index=augmentation_index,
            domain="prompt",
        )
        prompt = random.Random(
            prompt_seed,
        ).choice(
            args.prompts,
        )
        yield augmentation_index, prompt, job_seed


def _output_path(
    output_dir: Path,
    source: SourceExample,
    prompt: str,
    augmentation_index: int,
    config_fingerprint: str,
) -> Path:
    source_hash = hashlib.sha1(
        source.source_id.encode(
            "utf-8",
        )
    ).hexdigest()[:20]
    filename = (
        f"{source_hash}__a-{augmentation_index:03d}__p-{_slug(prompt)}"
        f"__c-{config_fingerprint[:8]}.png"
    )

    return output_dir / "images" / f"{source.label:06d}" / filename


def _save_png_atomic(
    image: Image.Image,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_suffix(
        output_path.suffix + f".{os.getpid()}.tmp",
    )
    image.save(
        temporary_path,
        format="PNG",
    )
    temporary_path.replace(
        output_path,
    )


def _write_run_config(
    output_dir: Path,
    config: dict[str, Any],
    config_fingerprint: str,
    fractal_catalog: list[dict[str, str]],
    scheduler_config: dict[str, Any],
    editor: InstructPix2PixEditor,
) -> str:
    path = output_dir / "run_config.json"
    payload = {
        "config_fingerprint": config_fingerprint,
        "config": config,
        "runtime": {
            "device_kind": editor.device_kind,
            "device": str(
                editor.device,
            ),
            "dtype": str(
                editor.dtype,
            ).removeprefix(
                "torch.",
            ),
            "scheduler_config": scheduler_config,
        },
        "fractal_catalog": fractal_catalog,
        "paper": "https://arxiv.org/abs/2405.14881",
        "paper_release_code_commit": (
            "b215f336036a075ca4ec442ef1a9fee8592ac240"
        ),
        "current_reference_code_commit": (
            "e58418d15d2ea5179dbedc2fcac80fe393dc6ec0"
        ),
    }
    payload = json.loads(
        json.dumps(
            payload,
            default=str,
        )
    )

    def resolve_existing() -> str:
        existing = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
        existing_fingerprint = existing.get(
            "config_fingerprint"
        )
        if existing_fingerprint == config_fingerprint:
            return config_fingerprint

        compatible_fingerprint = _resolve_config_fingerprint(
            output_dir=output_dir,
            config=config,
            scheduler_config=scheduler_config,
        )
        if compatible_fingerprint == existing_fingerprint:
            return compatible_fingerprint

        if existing_fingerprint != config_fingerprint:
            existing_config = existing.get(
                "config",
                {},
            )
            changed_fields = []
            if isinstance(
                existing_config,
                dict,
            ):
                changed_fields = sorted(
                    key
                    for key in set(
                        existing_config,
                    )
                    | set(
                        config,
                    )
                    if existing_config.get(
                        key,
                    )
                    != config.get(
                        key,
                    )
                )
            raise ValueError(
                "Output directory contains a DiffuseMix run with different "
                "settings. Choose a new output directory. Changed config "
                f"fields: {changed_fields}."
            )

        return config_fingerprint

    if path.exists():
        return resolve_existing()

    temporary_path = path.with_suffix(
        f".{os.getpid()}.tmp",
    )
    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        # Linking a fully written temporary file is atomic and fails when
        # another XLA worker already claimed the shared run config. This
        # prevents a check-then-replace race between generation ranks.
        os.link(
            temporary_path,
            path,
        )
    except FileExistsError:
        return resolve_existing()
    finally:
        temporary_path.unlink(
            missing_ok=True,
        )

    return config_fingerprint


def generate(
    args: argparse.Namespace,
) -> dict[str, int]:
    """Run one generation shard and return compact counters."""
    seam_width = args.seam_width
    if seam_width == -1:
        seam_width = 0 if args.mask_mode == "paper" else 20
    fractal_root = Path(
        args.fractal_dir,
    ).expanduser().resolve()
    fractal_paths = _fractal_paths(
        args.fractal_dir,
    )
    fractal_catalog = _fractal_catalog(
        fractal_paths=fractal_paths,
        fractal_root=fractal_root,
    )
    fractal_sha256 = {
        entry["path"]: entry["sha256"]
        for entry in fractal_catalog
    }
    config = _config_dict(
        args=args,
        seam_width=seam_width,
        fractal_catalog_sha256=_catalog_sha256(
            fractal_catalog,
        ),
    )
    output_dir = Path(
        args.output_dir,
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    editor = InstructPix2PixEditor(
        EditorConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            num_inference_steps=args.num_inference_steps,
            disable_safety_checker=not args.keep_safety_checker,
            attention_slicing=args.attention_slicing,
            xla_cache_dir=args.xla_cache_dir,
        )
    )
    scheduler_config = _stable_json_value(
        dict(
            editor.pipeline.scheduler.config,
        )
    )
    config["resolved_device"] = editor.device_kind
    config["resolved_dtype"] = str(
        editor.dtype,
    ).removeprefix(
        "torch.",
    )
    pipeline_config = editor.pipeline.config
    resolved_model_commit = getattr(
        editor,
        "resolved_model_commit",
        "",
    ) or getattr(
        pipeline_config,
        "_commit_hash",
        "",
    )
    if not resolved_model_commit and hasattr(
        pipeline_config,
        "get",
    ):
        resolved_model_commit = pipeline_config.get(
            "_commit_hash",
            "",
        )
    config["resolved_model_commit"] = str(
        resolved_model_commit or ""
    )
    config["scheduler_config_sha256"] = _fingerprint(
        _scheduler_behavior_config(
            scheduler_config,
        ),
    )
    config_fingerprint = _resolve_config_fingerprint(
        output_dir=output_dir,
        config=config,
        scheduler_config=scheduler_config,
    )
    shard_index, num_shards = _resolve_shard(
        args=args,
        editor=editor,
    )
    print(
        f"[diffusemix] worker shard={shard_index + 1}/{num_shards}; "
        "source_download="
        f"{bool(getattr(args, 'source_download', True))}",
        flush=True,
    )
    _guard_manifest_layout(
        output_dir=output_dir,
        num_shards=num_shards,
    )
    config_fingerprint = _write_run_config(
        output_dir=output_dir,
        config=config,
        config_fingerprint=config_fingerprint,
        fractal_catalog=fractal_catalog,
        scheduler_config=scheduler_config,
        editor=editor,
    )
    manifest_path = _manifest_path(
        output_dir=output_dir,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    summary_path = _summary_path(
        output_dir=output_dir,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if manifest_path.exists() and not summary_path.is_file():
        raise ValueError(
            "Cannot safely resume a DiffuseMix manifest whose generation "
            f"summary is missing: {summary_path}. Choose a new output "
            "directory or restore the summary."
        )
    previous_source_catalog_sha256 = _previous_source_catalog_sha256(
        summary_path=summary_path,
        config_fingerprint=config_fingerprint,
    )
    # Invalidate a prior successful marker before touching the manifest. If
    # this process is interrupted, JAX must not consume the old partial state.
    _write_generation_summary(
        summary_path=summary_path,
        manifest_path=manifest_path,
        shard_index=shard_index,
        num_shards=num_shards,
        config_fingerprint=config_fingerprint,
        complete=False,
        manifest_record_count=0,
        source_catalog_sha256=previous_source_catalog_sha256,
    )
    completed = _load_completed(
        manifest_path=manifest_path,
        output_dir=output_dir,
        config_fingerprint=config_fingerprint,
    )
    failure_path = output_dir / (
        "failures.jsonl"
        if num_shards == 1
        else f"failures-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    )

    mask_names = masks_for_mode(
        args.mask_mode,
    )
    release_semantics = args.mask_mode == "official_code"
    resize_resample = (
        Image.Resampling.BICUBIC
        if release_semantics
        else Image.Resampling.LANCZOS
    )
    dataset_name = args.dataset or args.dataset_name
    counters = {
        "source_examples": 0,
        "generated": 0,
        "resumed": 0,
        "failed": 0,
    }
    seen_source_ids: set[str] = set()
    source_catalog_digest = hashlib.sha256()
    with manifest_path.open(
        mode="a",
        encoding="utf-8",
    ) as manifest_file, failure_path.open(
        mode="w",
        encoding="utf-8",
    ) as failure_file:
        for train_position, source in enumerate(
            _source_iterator(
                args,
            )
        ):
            if (
                args.max_examples > 0
                and train_position >= args.max_examples
            ):
                break
            if source.index % num_shards != shard_index:
                continue
            if source.source_id in seen_source_ids:
                raise ValueError(
                    "Source adapter produced a duplicate source_id on one "
                    f"generation shard: {source.source_id}."
                )
            seen_source_ids.add(
                source.source_id,
            )

            counters["source_examples"] += 1
            original = source.image.convert(
                "RGB",
            ).resize(
                (
                    args.generation_size,
                    args.generation_size,
                ),
                resize_resample,
            )
            source_image_sha256 = _sha256_rgb_image(
                source.image,
            )
            source_descriptor = {
                "class_name": source.class_name,
                "label": source.label,
                "source_id": source.source_id,
                "source_image_sha256": source_image_sha256,
                "source_index": source.index,
                "source_ref": source.source_ref,
            }
            source_catalog_digest.update(
                json.dumps(
                    source_descriptor,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                ).encode(
                    "utf-8",
                )
            )
            source_catalog_digest.update(
                b"\n",
            )
            for augmentation_index, prompt, job_seed in _jobs_for_source(
                source=source,
                args=args,
            ):
                job_key = (
                    source.source_id,
                    augmentation_index,
                )
                if job_key in completed:
                    existing_record = completed.pop(
                        job_key,
                    )
                    expected_provenance = {
                        **source_descriptor,
                        "job_seed": job_seed,
                        "prompt": prompt,
                    }
                    mismatched_fields = [
                        field
                        for field, expected_value in (
                            expected_provenance.items()
                        )
                        if existing_record.get(
                            field,
                        )
                        != expected_value
                    ]
                    if mismatched_fields:
                        raise ValueError(
                            "Source provenance changed for an existing "
                            f"DiffuseMix job {job_key}; mismatched fields: "
                            f"{mismatched_fields}. Choose a new output "
                            "directory or restore the original source."
                        )
                    counters["resumed"] += 1
                    continue

                rng = random.Random(
                    job_seed,
                )
                mask_name = rng.choice(
                    mask_names,
                )
                fractal_path = rng.choice(
                    fractal_paths,
                )
                instruction = build_instruction(
                    prompt=prompt,
                    template=args.prompt_template,
                )
                generated = None
                accepted_attempt = -1
                accepted_seed = -1
                for attempt in range(
                    args.max_generation_attempts,
                ):
                    attempt_seed = _stable_seed(
                        root_seed=args.seed,
                        source_id=source.source_id,
                        augmentation_index=augmentation_index,
                        attempt=attempt,
                        domain="diffusion",
                    )
                    candidate = editor.edit(
                        image=original,
                        instruction=instruction,
                        seed=attempt_seed,
                    ).resize(
                        original.size,
                        resize_resample,
                    )
                    if not is_near_black(
                        image=candidate,
                        channel_threshold=args.black_threshold,
                        pixel_fraction=args.black_pixel_fraction,
                    ):
                        generated = candidate
                        accepted_attempt = attempt
                        accepted_seed = attempt_seed
                        break

                if generated is None:
                    counters["failed"] += 1
                    failure_file.write(
                        json.dumps(
                            {
                                "source_id": source.source_id,
                                "augmentation_index": augmentation_index,
                                "prompt": prompt,
                                "reason": "near_black_after_retries",
                                "attempts": args.max_generation_attempts,
                                "config_fingerprint": config_fingerprint,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    failure_file.flush()
                    continue

                with Image.open(
                    fractal_path,
                ) as fractal_image:
                    output_image = compose_diffusemix(
                        original=original,
                        generated=generated,
                        fractal=fractal_image,
                        mask_name=mask_name,
                        fractal_alpha=args.fractal_alpha,
                        seam_width=seam_width,
                        quantization=(
                            "truncate" if release_semantics else "round"
                        ),
                        resize_resample=resize_resample,
                    )
                if args.compact_output:
                    # Match the JAX manifest loader's bilinear classifier-size
                    # conversion as closely as Pillow permits, while keeping
                    # diffusion editing and composition at generation_size.
                    output_image = output_image.resize(
                        (
                            args.output_size,
                            args.output_size,
                        ),
                        Image.Resampling.BILINEAR,
                    )
                output_path = _output_path(
                    output_dir=output_dir,
                    source=source,
                    prompt=prompt,
                    augmentation_index=augmentation_index,
                    config_fingerprint=config_fingerprint,
                )
                _save_png_atomic(
                    image=output_image,
                    output_path=output_path,
                )
                relative_output = output_path.relative_to(
                    output_dir,
                ).as_posix()
                relative_fractal = fractal_path.relative_to(
                    fractal_root,
                ).as_posix()
                record = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "config_fingerprint": config_fingerprint,
                    "dataset": dataset_name,
                    "image_path": relative_output,
                    "label": source.label,
                    "class_name": source.class_name,
                    "source_id": source.source_id,
                    "source_ref": source.source_ref,
                    "source_index": source.index,
                    "source_image_sha256": source_image_sha256,
                    "source_partition": "train",
                    "validation_split": args.validation_split,
                    "augmentation_index": augmentation_index,
                    "prompt": prompt,
                    "instruction": instruction,
                    "mask": mask_name,
                    "seam_width": seam_width,
                    "fractal": relative_fractal,
                    "fractal_sha256": fractal_sha256[relative_fractal],
                    "fractal_alpha": args.fractal_alpha,
                    "job_seed": job_seed,
                    "diffusion_seed": accepted_seed,
                    "generation_attempt": accepted_attempt,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "device": editor.device_kind,
                    "dtype": str(
                        editor.dtype,
                    ).removeprefix(
                        "torch.",
                    ),
                    "guidance_scale": args.guidance_scale,
                    "image_guidance_scale": args.image_guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "generation_size": args.generation_size,
                    "compact_output": args.compact_output,
                    "output_size": args.output_size,
                    "output_resize": (
                        "pillow-bilinear"
                        if args.compact_output
                        else "none"
                    ),
                    "generated_rgb_sha256": _sha256_rgb_image(
                        generated,
                    ),
                    "output_png_sha256": _sha256_file(
                        output_path,
                    ),
                }
                manifest_file.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                    )
                    + "\n"
                )
                manifest_file.flush()
                counters["generated"] += 1
                completed_count = (
                    counters["generated"] + counters["resumed"]
                )
                if completed_count % args.log_every == 0:
                    print(
                        f"DiffuseMix shard {shard_index + 1}/{num_shards}: "
                        f"{completed_count} jobs complete, "
                        f"{counters['failed']} failed"
                    )

    unexpected_completed_jobs = sorted(
        completed,
    )
    if unexpected_completed_jobs:
        preview = unexpected_completed_jobs[:5]
        raise ValueError(
            "Existing DiffuseMix records no longer belong to the current "
            f"training-source snapshot: {preview}. Choose a new output "
            "directory or restore the original source dataset."
        )

    source_catalog_sha256 = source_catalog_digest.hexdigest()
    if (
        previous_source_catalog_sha256
        and previous_source_catalog_sha256 != source_catalog_sha256
    ):
        raise ValueError(
            "The DiffuseMix training-source catalog changed since this "
            "artifact was completed. Choose a new output directory; this "
            "artifact remains locked to its previous source snapshot."
        )

    editor.synchronize()
    manifest_record_count = _count_manifest_records(
        manifest_path,
    )
    accepted_incomplete = bool(
        counters["failed"] > 0 and args.allow_generation_failures
    )
    _write_generation_summary(
        summary_path=summary_path,
        manifest_path=manifest_path,
        shard_index=shard_index,
        num_shards=num_shards,
        config_fingerprint=config_fingerprint,
        complete=(
            counters["failed"] == 0 or args.allow_generation_failures
        ),
        manifest_record_count=manifest_record_count,
        counters=counters,
        accepted_incomplete=accepted_incomplete,
        source_catalog_sha256=source_catalog_sha256,
    )
    print(
        f"DiffuseMix generation complete on shard {shard_index + 1}/"
        f"{num_shards}: {counters}"
    )
    print(
        f"Manifest: {manifest_path}"
    )

    if counters["failed"] > 0 and not args.allow_generation_failures:
        raise RuntimeError(
            f"DiffuseMix generation left {counters['failed']} jobs without "
            f"an image on shard {shard_index + 1}/{num_shards}. Details: "
            f"{failure_path}. The attempt seeds are deterministic; choose a "
            "new seed/output directory, or pass --allow-generation-failures "
            "to accept an incomplete artifact."
        )

    return counters


def main(
    argv: list[str] | None = None,
) -> None:
    """Run the command-line entry point."""
    # The source adapter may import TensorFlow. Keep it away from CUDA while
    # PyTorch owns generation; JAX training happens in a later process.
    os.environ.setdefault(
        "TF_CPP_MIN_LOG_LEVEL",
        "3",
    )
    args = parse_args(
        argv,
    )
    from allthemix.competitors.generative.source_preflight import (
        prepare_generation_dataset,
    )

    prepare_generation_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
    )
    args.source_download = False
    if args.xla_launch:
        torch_xla = import_torch_xla()

        args.device = "xla"
        args.num_shards = 0
        torch_xla.launch(
            _xla_worker,
            args=(
                args,
            ),
        )
    else:
        generate(
            args,
        )


def _xla_worker(
    worker_index: int,
    args: argparse.Namespace,
) -> None:
    """Run one shard created by ``torch_xla.launch``."""
    print(
        f"[diffusemix] XLA worker={worker_index} pid={os.getpid()} starting",
        flush=True,
    )
    if args.xla_cache_dir:
        args.xla_cache_dir = str(
            Path(
                args.xla_cache_dir,
            ).expanduser()
            / f"rank-{worker_index:05d}"
        )
    generate(
        args,
    )
    print(
        f"[diffusemix] XLA worker={worker_index} pid={os.getpid()} completed",
        flush=True,
    )


if __name__ == "__main__":
    main()
