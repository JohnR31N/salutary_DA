"""Official SaSPA scene prompts and fine-grained prompt formatting."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from allthemix.competitors.generative.sources import clean_class_name

OFFICIAL_SASPA_REPOSITORY = "https://github.com/EyalMichaeli/SaSPA-Aug"
OFFICIAL_PROMPT_FILES = {
    "caltech_birds2011": "cub-100-gpt_v1.txt",
    "cars196": "cars-100-gpt_v1.txt",
}
EXTENSION_PROMPT_FILES = {
    "stl10": "stl10-generic-v1.txt",
}
PROMPT_FILES = {
    **OFFICIAL_PROMPT_FILES,
    **EXTENSION_PROMPT_FILES,
}
PROMPT_SOURCES = {
    "caltech_birds2011": OFFICIAL_SASPA_REPOSITORY,
    "cars196": OFFICIAL_SASPA_REPOSITORY,
    "stl10": "allthemix-stl10-extension-v1",
}
SUPERCLASSES = {
    "caltech_birds2011": "bird",
    "cars196": "car",
    "stl10": "object",
}
SEMANTIC_PROMPTS = {
    "caltech_birds2011": ("a photo of a bird",),
    "cars196": ("a photo of a car",),
    "stl10": (
        "a photo of an animal",
        "a photo of a vehicle",
    ),
}
SEMANTIC_NEGATIVE_PROMPTS = (
    "a photo of an object",
    "a photo of a scene",
    "a photo of geometric shapes",
    "a photo",
    "an image",
    "a black photo",
)
STL10_SEMANTIC_NEGATIVE_PROMPTS = (
    "a photo of geometric shapes",
    "an abstract image",
    "a painting",
    "a drawing",
    "a photo of a person",
    "a black photo",
)


def normalize_dataset_name(dataset: str) -> str:
    """Validate one dataset supported by SaSPA or its STL extension."""
    name = dataset.strip().lower()
    if name not in PROMPT_FILES:
        raise ValueError(
            "SaSPA currently supports caltech_birds2011, cars196, and "
            "stl10. "
            f"Got {dataset!r}."
        )

    return name


def load_official_prompts(
    dataset: str,
    prompt_file: str | Path = "",
) -> tuple[str, ...]:
    """Load the dataset's immutable scene-prompt artifact."""
    dataset_name = normalize_dataset_name(dataset)
    if prompt_file:
        path = Path(prompt_file).expanduser()
        text = path.read_text(encoding="utf-8")
    else:
        asset = resources.files(f"{__package__}.assets").joinpath(
            PROMPT_FILES[dataset_name],
        )
        text = asset.read_text(encoding="utf-8")
    prompts = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not prompts:
        raise ValueError("SaSPA prompt file contains no nonempty prompts.")

    return prompts


def format_scene_prompt(
    scene_prompt: str,
    class_name: str,
    superclass: str,
) -> str:
    """Inject the fine-grained class before the superclass token."""
    prompt = scene_prompt.strip().rstrip(".")
    fine_class = clean_class_name(class_name)
    if not fine_class:
        raise ValueError("SaSPA class_name must contain prompt text.")
    if "{class_name}" in prompt:
        return prompt.replace("{class_name}", fine_class)
    pattern = re.compile(rf"\b{re.escape(superclass)}\b", re.IGNORECASE)
    if pattern.search(prompt) is None:
        raise ValueError(
            f"SaSPA scene prompt must contain {superclass!r}: {scene_prompt!r}."
        )

    return pattern.sub(
        f"{fine_class} {superclass}",
        prompt,
        count=1,
    )


def semantic_negative_prompts(dataset: str) -> tuple[str, ...]:
    """Return official or extension-specific semantic negatives."""
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "stl10":
        return STL10_SEMANTIC_NEGATIVE_PROMPTS

    return SEMANTIC_NEGATIVE_PROMPTS
