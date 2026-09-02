"""Prompt discovery plus paper and released-code presets for ALIA."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

from allthemix.competitors.generative.artifacts import (
    atomic_write_json,
    config_fingerprint,
)
from allthemix.competitors.generative.sources import clean_class_name

CUB_DATASET_NAMES = {
    "caltech_birds2011",
    "cub200",
    "cub_200_2011",
}
CUB_PAPER_PROMPTS = (
    "a photo of a {class_name} bird interacting with flowers.",
    "a photo of a {class_name} bird standing by the waters edge.",
    "a photo of a {class_name} bird perched on a fence.",
    "a photo of a {class_name} bird standing on a rock.",
    "a photo of a {class_name} bird perched on a branch.",
    "a photo of a {class_name} bird flying near a tree, sky as the backdrop.",
    "a photo of a {class_name} bird perched on a birdfeeder.",
)
CUB_RELEASE_PROMPTS = (
    "a photo of a {class_name} bird flying.",
    "a photo of a {class_name} bird interacting with flowers.",
    "a photo of a {class_name} bird in the water.",
    "a photo of a {class_name} bird on a branch.",
    "a photo of a {class_name} bird on rocks.",
    "a photo of a {class_name} bird perched on a birdfeeder.",
    "a photo of a {class_name} bird perched on a fence.",
)
CUB_SEMANTIC_PROMPTS = ("a photo of a bird",)
CUB_NEGATIVE_PROMPTS = (
    "a photo of an object",
    "a photo of geometric shapes",
    "a photo",
    "an image",
    "a painting",
    "a photo of a person",
    "a cartoon of a bird",
)
STL10_DATASET_NAMES = {
    "stl10",
}
STL10_GENERIC_PROMPTS = (
    "a photo showing {class_name} outdoors during daytime.",
    "a close-up photo showing {class_name}.",
    "a photo showing {class_name} viewed from the side.",
    "a photo showing {class_name} viewed from the front.",
    "a photo showing {class_name} in motion.",
    "a photo showing {class_name} at a distance.",
    "a photo showing {class_name} with a softly blurred background.",
)
STL10_SEMANTIC_PROMPTS = (
    "a photo of an animal",
    "a photo of a vehicle",
)
STL10_NEGATIVE_PROMPTS = (
    "a photo of geometric shapes",
    "an abstract image",
    "a painting",
    "a drawing",
    "a photo of a person",
    "a black image",
)
CARS196_DATASET_NAMES = {
    "cars196",
}
CARS196_GENERIC_PROMPTS = (
    "a photo of the {class_name} parked outdoors.",
    "a photo of the {class_name} driving on a road.",
    "a front view photo of the {class_name}.",
    "a rear view photo of the {class_name}.",
    "a side view photo of the {class_name}.",
    "a photo of the {class_name} in a parking lot.",
    "a photo of the {class_name} at an automobile show.",
)
CARS196_SEMANTIC_PROMPTS = (
    "a photo of a car",
    "a photo of an automobile",
    "a photo of a road vehicle",
)
CARS196_NEGATIVE_PROMPTS = (
    "a photo of an animal",
    "a photo of a person",
    "a photo of a motorcycle",
    "a painting",
    "a drawing",
    "an abstract image",
    "a black image",
)


def format_prompt(template: str, class_name: str) -> str:
    """Render one class-aware prompt with a cleaned dataset class name."""
    if "{class_name}" not in template:
        raise ValueError(
            "ALIA prompt templates must contain '{class_name}': "
            f"{template!r}."
        )

    return template.format(class_name=clean_class_name(class_name))


def _caption_is_usable(caption: str) -> bool:
    """Match the official repeated-word rejection with stable tokenization."""
    words = re.findall(r"[A-Za-z0-9'-]+", caption.lower())
    if not words:
        return False
    counts = {word: words.count(word) for word in set(words)}

    return max(counts.values()) <= 5


def build_prompt_request(
    captions: Iterable[str],
    prefix: str,
    max_captions: int = 20,
    seed: int = 0,
) -> str:
    """Build the official random-20-caption request reproducibly."""
    deduplicated = []
    seen = set()
    for caption in captions:
        caption = str(caption).strip()
        if caption and caption not in seen and _caption_is_usable(caption):
            deduplicated.append(caption)
            seen.add(caption)
    if not deduplicated:
        raise ValueError("No usable captions remain for ALIA prompt discovery.")

    sample_size = min(max_captions, len(deduplicated))
    selected = random.Random(seed).sample(deduplicated, sample_size)
    examples = (
        f'- "{prefix} standing on a branch."\n'
        f'- "{prefix} flying in the sky with a city skyline in the '
        'background."\n'
        f'- "{prefix} playing in a river at night."'
    )

    return (
        "I have a set of image captions to summarize into objective "
        "descriptions of scenes, actions, camera pose, zoom, weather, and "
        "other visible image qualities.\n\n"
        "Captions:\n- "
        + "\n- ".join(selected)
        + "\n\nReturn at most 10 unique prompt templates. Each line must "
        f"start with '- ' and must begin with '{prefix}'. Do not add "
        "explanations. Examples:\n"
        + examples
    )


def parse_prompt_response(
    response: str,
    prefix: str,
) -> tuple[str, ...]:
    """Parse bullet-style LLM output into class-aware prompt templates."""
    prompts = []
    seen = set()
    normalized_prefix = prefix.strip().rstrip(".")
    for raw_line in response.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
        line = line.strip('"\' ')
        if not line:
            continue
        line = line.replace("{}", "{class_name}")
        if "{class_name}" not in line:
            if line.lower().startswith(normalized_prefix.lower()):
                suffix = line[len(normalized_prefix):].lstrip(" .,:-")
                line = f"{normalized_prefix} {suffix}".strip()
            elif line.lower().startswith("a photo of"):
                line = re.sub(
                    r"^a photo of (?:a |an )?",
                    "a photo of a {class_name} ",
                    line,
                    flags=re.IGNORECASE,
                )
            else:
                line = f"{normalized_prefix} {line}"
        if "{class_name}" not in line:
            line = line.replace(
                normalized_prefix,
                normalized_prefix.replace("{}", "{class_name}"),
                1,
            )
        line = re.sub(r"\s+", " ", line).strip()
        if not line.endswith((".", "!", "?")):
            line += "."
        if line not in seen:
            prompts.append(line)
            seen.add(line)
        if len(prompts) == 10:
            break

    for prompt in prompts:
        if "{class_name}" not in prompt:
            raise ValueError(
                "The LLM response could not be converted to class-aware "
                f"templates: {prompt!r}."
            )
    if not prompts:
        raise ValueError("The LLM response contained no prompt templates.")

    return tuple(prompts)


def paper_prompt_payload(dataset: str) -> dict[str, object]:
    """Return checked-in prompts explicitly reported by the ALIA paper."""
    if dataset.lower() not in CUB_DATASET_NAMES:
        raise ValueError(
            "No paper prompt preset is available for dataset "
            f"{dataset!r}. Supply an LLM response instead."
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "alia",
        "dataset": dataset,
        "prompt_source": "paper",
        "prompts": list(CUB_PAPER_PROMPTS),
        "semantic_prompts": list(CUB_SEMANTIC_PROMPTS),
        "negative_prompts": list(CUB_NEGATIVE_PROMPTS),
    }
    payload["prompt_fingerprint"] = config_fingerprint(payload)

    return payload


def release_prompt_payload(dataset: str) -> dict[str, object]:
    """Return the seven CUB prompts named by the official GitHub config."""
    if dataset.lower() not in CUB_DATASET_NAMES:
        raise ValueError(
            "No release prompt preset is available for dataset "
            f"{dataset!r}. Supply an LLM response instead."
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "alia",
        "dataset": dataset,
        "prompt_source": "official-github-release",
        "prompts": list(CUB_RELEASE_PROMPTS),
        "semantic_prompts": list(CUB_SEMANTIC_PROMPTS),
        "negative_prompts": list(CUB_NEGATIVE_PROMPTS),
    }
    payload["prompt_fingerprint"] = config_fingerprint(payload)

    return payload


def generic_prompt_payload(dataset: str) -> dict[str, object]:
    """Return a reproducible AllTheMix prompt extension."""
    dataset_name = dataset.lower()
    if dataset_name in STL10_DATASET_NAMES:
        prompt_source = "allthemix-generic-stl10-v1"
        prompts = STL10_GENERIC_PROMPTS
        semantic_prompts = STL10_SEMANTIC_PROMPTS
        negative_prompts = STL10_NEGATIVE_PROMPTS
    elif dataset_name in CARS196_DATASET_NAMES:
        prompt_source = "allthemix-generic-cars196-v1"
        prompts = CARS196_GENERIC_PROMPTS
        semantic_prompts = CARS196_SEMANTIC_PROMPTS
        negative_prompts = CARS196_NEGATIVE_PROMPTS
    else:
        raise ValueError(
            "No generic ALIA prompt preset is available for dataset "
            f"{dataset!r}. Supply an LLM response instead."
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "alia",
        "dataset": dataset,
        "prompt_source": prompt_source,
        "prompts": list(prompts),
        "semantic_prompts": list(semantic_prompts),
        "negative_prompts": list(negative_prompts),
    }
    payload["prompt_fingerprint"] = config_fingerprint(payload)

    return payload


def semantic_prompt_payload(dataset: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return dataset-aware CLIP prompts for discovered ALIA prompts."""
    dataset_name = dataset.lower()
    if dataset_name in CUB_DATASET_NAMES:
        return CUB_SEMANTIC_PROMPTS, CUB_NEGATIVE_PROMPTS
    if dataset_name in STL10_DATASET_NAMES:
        return STL10_SEMANTIC_PROMPTS, STL10_NEGATIVE_PROMPTS
    if dataset_name in CARS196_DATASET_NAMES:
        return CARS196_SEMANTIC_PROMPTS, CARS196_NEGATIVE_PROMPTS

    raise ValueError(
        "No built-in ALIA semantic prompts are available for dataset "
        f"{dataset!r}."
    )


def write_prompt_payload(path: str | Path, payload: dict[str, object]) -> None:
    """Validate and atomically persist prompt discovery output."""
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("ALIA prompt payload must contain a nonempty list.")
    for prompt in prompts:
        if not isinstance(prompt, str) or "{class_name}" not in prompt:
            raise ValueError(
                "Every ALIA prompt must be a string containing "
                "'{class_name}'."
            )
    atomic_write_json(path, payload)


def read_prompt_payload(path: str | Path) -> dict[str, object]:
    """Read and validate an immutable ALIA prompt artifact."""
    prompt_path = Path(path)
    try:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid ALIA prompt JSON: {prompt_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"ALIA prompt artifact must be an object: {path}")
    fingerprint = payload.pop("prompt_fingerprint", None)
    actual = config_fingerprint(payload)
    payload["prompt_fingerprint"] = fingerprint
    if fingerprint != actual:
        raise ValueError(
            f"ALIA prompt fingerprint mismatch: {prompt_path}."
        )
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"ALIA prompt artifact has no prompts: {path}")

    return payload
