"""Offline Automatic Language-guided Image Augmentation competitor."""

from allthemix.competitors.alia.ablation import (
    build_paired_ablation_artifacts,
)
from allthemix.competitors.alia.filtering import (
    compute_confident_thresholds,
    filter_generated_records,
    strict_filter_generated_records,
)
from allthemix.competitors.alia.manifest import (
    count_manifest_examples,
    load_manifest_dataset,
    read_stage_records,
    validate_manifest_for_training,
)
from allthemix.competitors.alia.official_artifact import (
    OFFICIAL_CUB_ARTIFACT_REF,
    import_official_cub_artifact,
    scan_official_cub_artifact,
)
from allthemix.competitors.alia.prompts import (
    CUB_PAPER_PROMPTS,
    CUB_RELEASE_PROMPTS,
    build_prompt_request,
    format_prompt,
    parse_prompt_response,
    release_prompt_payload,
)
from allthemix.competitors.alia.visualize import (
    rank_quality_records,
    visualize_filtered_quality,
)

__all__ = [
    "CUB_PAPER_PROMPTS",
    "CUB_RELEASE_PROMPTS",
    "OFFICIAL_CUB_ARTIFACT_REF",
    "build_paired_ablation_artifacts",
    "build_prompt_request",
    "compute_confident_thresholds",
    "count_manifest_examples",
    "filter_generated_records",
    "format_prompt",
    "import_official_cub_artifact",
    "load_manifest_dataset",
    "parse_prompt_response",
    "rank_quality_records",
    "read_stage_records",
    "release_prompt_payload",
    "scan_official_cub_artifact",
    "strict_filter_generated_records",
    "validate_manifest_for_training",
    "visualize_filtered_quality",
]
