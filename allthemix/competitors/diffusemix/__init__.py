"""Offline DiffuseMix generation and JAX dataset hand-off."""

from allthemix.competitors.diffusemix.compose import (
    DEFAULT_PROMPTS,
    OFFICIAL_RELEASE_PROMPTS,
    PAPER_MASKS,
    PAPER_PROMPTS,
    blend_fractal,
    build_instruction,
    compose_diffusemix,
    is_near_black,
    merge_original_and_generated,
)
from allthemix.competitors.diffusemix.manifest import (
    count_manifest_examples,
    load_manifest_dataset,
    read_manifest_records,
    validate_manifest_for_training,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "OFFICIAL_RELEASE_PROMPTS",
    "PAPER_MASKS",
    "PAPER_PROMPTS",
    "blend_fractal",
    "build_instruction",
    "compose_diffusemix",
    "count_manifest_examples",
    "is_near_black",
    "load_manifest_dataset",
    "merge_original_and_generated",
    "read_manifest_records",
    "validate_manifest_for_training",
]
