"""Import the public ALIA CUB artifact into the staged manifest contract."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from allthemix.competitors.alia.manifest import (
    ALIA_SCHEMA_VERSION,
    stage_manifest_name,
)
from allthemix.competitors.generative.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    config_fingerprint,
    sha256_file,
)
from allthemix.competitors.generative.sources import (
    is_validation_class_index,
    validate_validation_split,
)

OFFICIAL_CUB_ARTIFACT_REF = "clipinvariance/ALIA/cub_generic:v0"
_CUB_DATASET = "caltech_birds2011"
_CUB_NUM_CLASSES = 200
_CUB_TRAIN_EXAMPLES = 5_994
_CLASS_DIRECTORY_PATTERN = re.compile(r"^(\d{3})\.(.+)$")
_EDIT_FILE_PATTERN = re.compile(r"^(\d+)-(\d+)\.png$", re.IGNORECASE)
_SAMPLE_PREVIEW_FILE_PATTERN = re.compile(r"^\d+\.png$", re.IGNORECASE)
_OFFICIAL_INVERTED_CLASSES = frozenset(
    {
        8,
        20,
        22,
        24,
        25,
        28,
        41,
        42,
        50,
        56,
        58,
        62,
        64,
        67,
        71,
        81,
        82,
        85,
        87,
        88,
        89,
        97,
        100,
        103,
        104,
        112,
        113,
        123,
        125,
        127,
        130,
        131,
        133,
        134,
        135,
        140,
        142,
        143,
        145,
        158,
        161,
        162,
        163,
        168,
        169,
        175,
        178,
        187,
        191,
        196,
    }
)


@dataclass(frozen=True)
class OfficialCubEdit:
    """One PNG from the public W&B artifact."""

    path: Path
    relative_path: str
    label: int
    class_directory: str
    source_index: int
    variant_index: int


@dataclass(frozen=True)
class CubTrainSource:
    """One official CUB train image in AllTheMix's deterministic order."""

    global_index: int
    label: int
    class_index: int
    filename: str
    class_name: str
    is_validation: bool


def _normalized_class_name(value: str) -> str:
    """Normalize formatting without applying prompt-oriented name aliases."""
    value = re.sub(r"^\d+[._ -]*", "", value.strip())

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _decode_filename(value: object) -> str:
    """Decode the TFDS image filename into a stable POSIX-style string."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    filename = str(value).replace("\\", "/").strip()
    if not filename:
        raise ValueError("CUB TFDS example has an empty image/filename field.")

    return filename


def _decode_class_name(value: object, fallback: str) -> str:
    """Prefer the record-level CUB label name over numeric TFDS metadata."""
    if value is None:
        return fallback
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    class_name = str(value).strip()

    return class_name or fallback


def _is_numeric_class_placeholder(value: str, label: int) -> bool:
    """Identify unnamed TFDS ClassLabel values such as ``"0"``."""
    return value.strip() == str(label)


def _find_class_root(artifact_dir: Path) -> Path:
    """Find the directory whose immediate children are all 200 CUB classes."""
    candidates = []
    nested_directories = (
        path for path in artifact_dir.rglob("*") if path.is_dir()
    )
    for candidate in (artifact_dir, *nested_directories):
        class_numbers = {
            int(match.group(1))
            for child in candidate.iterdir()
            if child.is_dir()
            for match in [_CLASS_DIRECTORY_PATTERN.fullmatch(child.name)]
            if match is not None
        }
        if class_numbers == set(range(1, _CUB_NUM_CLASSES + 1)):
            candidates.append(candidate.resolve())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one directory containing CUB class folders "
            f"001..200 under {artifact_dir}, found {candidates}."
        )

    return candidates[0]


def scan_official_cub_artifact(
    artifact_dir: str | Path,
) -> tuple[Path, list[OfficialCubEdit], int]:
    """Parse and validate the public artifact's class-folder layout."""
    root = Path(artifact_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Official ALIA artifact is missing: {root}")
    class_root = _find_class_root(root)
    class_directories: dict[int, Path] = {}
    for child in class_root.iterdir():
        if not child.is_dir():
            continue
        match = _CLASS_DIRECTORY_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        label = int(match.group(1)) - 1
        if label in class_directories:
            raise ValueError(f"Duplicate official CUB class label: {label}")
        class_directories[label] = child

    edits = []
    ignored_preview_pngs = 0
    seen_coordinates: set[tuple[int, int]] = set()
    source_labels: dict[int, int] = {}
    for label in range(_CUB_NUM_CLASSES):
        class_dir = class_directories[label]
        image_paths = sorted(class_dir.glob("*.png"))
        if not image_paths:
            raise ValueError(f"Official ALIA class has no PNG files: {class_dir}")
        unexpected_nested_pngs = []
        for path in class_dir.rglob("*.png"):
            if path.parent == class_dir:
                continue
            relative_path = path.relative_to(class_dir)
            is_official_preview = (
                len(relative_path.parts) == 2
                and relative_path.parts[0] == "samples"
                and _SAMPLE_PREVIEW_FILE_PATTERN.fullmatch(
                    relative_path.parts[1]
                )
                is not None
            )
            if is_official_preview:
                ignored_preview_pngs += 1
            else:
                unexpected_nested_pngs.append(path)
        if unexpected_nested_pngs:
            unexpected_nested_pngs.sort()
            raise ValueError(
                "Official ALIA artifact contains unexpected nested PNGs, "
                f"starting with {unexpected_nested_pngs[0]}."
            )
        for image_path in image_paths:
            match = _EDIT_FILE_PATTERN.fullmatch(image_path.name)
            if match is None:
                raise ValueError(
                    "Official ALIA PNG must use source-variant naming: "
                    f"{image_path}"
                )
            source_index = int(match.group(1))
            variant_index = int(match.group(2))
            coordinate = (source_index, variant_index)
            if coordinate in seen_coordinates:
                raise ValueError(
                    "Duplicate official ALIA source/variant coordinate: "
                    f"{coordinate}."
                )
            previous_label = source_labels.setdefault(source_index, label)
            if previous_label != label:
                raise ValueError(
                    f"Official source {source_index} occurs in labels "
                    f"{previous_label} and {label}."
                )
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as error:
                raise ValueError(
                    f"Official ALIA image is not a valid PNG: {image_path}"
                ) from error
            seen_coordinates.add(coordinate)
            edits.append(
                OfficialCubEdit(
                    path=image_path.resolve(),
                    relative_path=image_path.relative_to(root).as_posix(),
                    label=label,
                    class_directory=class_dir.name,
                    source_index=source_index,
                    variant_index=variant_index,
                )
            )
    edits.sort(
        key=lambda edit: (
            edit.source_index,
            edit.variant_index,
            edit.relative_path,
        )
    )

    return class_root, edits, ignored_preview_pngs


def _tfds_class_names(dataset: str, data_dir: str) -> tuple[str, ...]:
    """Load the CUB label vocabulary without relying on folder ordering."""
    import tensorflow_datasets as tfds

    builder = tfds.builder(dataset, data_dir=data_dir)
    names = tuple(builder.info.features["label"].names)
    if len(names) != _CUB_NUM_CLASSES:
        raise ValueError(
            f"CUB TFDS should expose {_CUB_NUM_CLASSES} classes, got "
            f"{len(names)}."
        )

    return names


def _load_cub_train_sources(
    dataset: str,
    data_dir: str,
    validation_split: float,
) -> list[CubTrainSource]:
    """Read source filenames and reproduce the exact AllTheMix val split."""
    from allthemix.data.datasets.loader import load_train_dataset

    class_names = _tfds_class_names(dataset=dataset, data_dir=data_dir)
    raw_dataset = load_train_dataset(
        name=dataset,
        data_dir=data_dir,
        shuffle_files=False,
    )
    class_counts: dict[int, int] = defaultdict(int)
    observed_class_names: dict[int, str] = {}
    sources = []
    for global_index, example in enumerate(raw_dataset.as_numpy_iterator()):
        if "label" not in example or "image/filename" not in example:
            raise ValueError(
                "CUB TFDS records must contain label and image/filename."
            )
        label = int(example["label"])
        if not 0 <= label < _CUB_NUM_CLASSES:
            raise ValueError(f"CUB TFDS label is out of range: {label}")
        class_name = _decode_class_name(
            example.get("label_name"),
            fallback=class_names[label],
        )
        previous_class_name = observed_class_names.setdefault(label, class_name)
        if _normalized_class_name(previous_class_name) != _normalized_class_name(
            class_name
        ):
            raise ValueError(
                "CUB TFDS label_name changed within one class: "
                f"label={label}, first={previous_class_name!r}, "
                f"current={class_name!r}."
            )
        class_index = class_counts[label]
        class_counts[label] += 1
        sources.append(
            CubTrainSource(
                global_index=global_index,
                label=label,
                class_index=class_index,
                filename=_decode_filename(example["image/filename"]),
                class_name=class_name,
                is_validation=is_validation_class_index(
                    class_index=class_index,
                    validation_split=validation_split,
                ),
            )
        )
    if len(sources) != _CUB_TRAIN_EXAMPLES:
        raise ValueError(
            f"CUB TFDS train split should contain {_CUB_TRAIN_EXAMPLES} "
            f"examples, got {len(sources)}."
        )
    if set(class_counts) != set(range(_CUB_NUM_CLASSES)):
        raise ValueError("CUB TFDS train split does not cover all 200 labels.")

    return sources


def _sources_by_label(
    sources: Iterable[CubTrainSource],
) -> dict[int, list[CubTrainSource]]:
    """Return each class in the original metadata filename order."""
    grouped: dict[int, list[CubTrainSource]] = defaultdict(list)
    for source in sources:
        grouped[source.label].append(source)
    for label in range(_CUB_NUM_CLASSES):
        grouped[label].sort(key=lambda source: source.filename)

    return grouped


def _candidate_source_layouts(
    sources: Iterable[CubTrainSource],
) -> dict[str, list[CubTrainSource]]:
    """Reconstruct source orders used by released ALIA CUB dataset classes."""
    grouped = _sources_by_label(sources)

    def flatten(per_label: dict[int, list[CubTrainSource]]) -> list[CubTrainSource]:
        return [
            source
            for label in range(_CUB_NUM_CLASSES)
            for source in per_label[label]
        ]

    new_cub = {
        label: class_sources[:-15]
        for label, class_sources in grouped.items()
    }
    cub_subset = {
        label: (
            class_sources[:-20]
            if label + 1 in _OFFICIAL_INVERTED_CLASSES
            else class_sources
        )
        for label, class_sources in grouped.items()
    }

    return {
        "newCub2011_train_minus_15": flatten(new_cub),
        "Cub2011_full_train": flatten(grouped),
        "Cub2011_inverted_class_holdout": flatten(cub_subset),
    }


def match_official_cub_source_layout(
    edits: Iterable[OfficialCubEdit],
    sources: Iterable[CubTrainSource],
) -> tuple[str, dict[int, CubTrainSource]]:
    """Identify the released source-index convention without guessing."""
    source_labels = {
        edit.source_index: edit.label
        for edit in edits
    }
    if not source_labels:
        raise ValueError("Official ALIA artifact contains no source images.")

    matches = []
    diagnostics = {}
    for name, ordered_sources in _candidate_source_layouts(sources).items():
        mismatched = 0
        for source_index, label in source_labels.items():
            if (
                source_index >= len(ordered_sources)
                or ordered_sources[source_index].label != label
            ):
                mismatched += 1
        diagnostics[name] = mismatched
        if mismatched == 0:
            matches.append(
                (
                    abs(len(ordered_sources) - len(source_labels)),
                    name,
                    ordered_sources,
                )
            )
    if not matches:
        raise ValueError(
            "The official ALIA source indices do not match any released CUB "
            f"dataset layout: {diagnostics}."
        )
    matches.sort(key=lambda item: (item[0], item[1]))
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError(
            "The official ALIA source layout is ambiguous: "
            f"{[(item[1], item[0]) for item in matches]}."
        )
    _distance, layout_name, ordered_sources = matches[0]
    mapping = {
        source_index: ordered_sources[source_index]
        for source_index in source_labels
    }

    return layout_name, mapping


def _inventory_fingerprint(edits: Iterable[OfficialCubEdit]) -> str:
    """Fingerprint artifact paths and sizes before publishing provenance."""
    digest = hashlib.sha256()
    for edit in edits:
        digest.update(edit.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(edit.path.stat().st_size).encode("ascii"))
        digest.update(b"\n")

    return digest.hexdigest()


def import_official_cub_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    data_dir: str = "./data",
    dataset: str = _CUB_DATASET,
    validation_split: float = 0.1,
    artifact_ref: str = OFFICIAL_CUB_ARTIFACT_REF,
    prompt: str = "see official W&B artifact metadata",
    overwrite: bool = False,
) -> dict[str, object]:
    """Publish leakage-free official PNGs as an ALIA generated stage."""
    dataset = dataset.strip().lower()
    if dataset != _CUB_DATASET:
        raise ValueError(
            "The public cub_generic artifact is valid only for "
            f"{_CUB_DATASET}, got {dataset!r}."
        )
    validate_validation_split(validation_split)
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / stage_manifest_name("generated")
    summary_path = destination / "generated_summary.json"
    if not overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError(
            "Official ALIA import output already exists. Use a new output "
            f"directory or --overwrite: {destination}"
        )

    class_root, edits, ignored_preview_pngs = scan_official_cub_artifact(
        artifact_dir
    )
    sources = _load_cub_train_sources(
        dataset=dataset,
        data_dir=data_dir,
        validation_split=validation_split,
    )
    layout_name, source_mapping = match_official_cub_source_layout(
        edits=edits,
        sources=sources,
    )
    for edit in edits:
        source = source_mapping[edit.source_index]
        has_meaningful_tfds_name = not _is_numeric_class_placeholder(
            source.class_name,
            source.label,
        )
        if (
            has_meaningful_tfds_name
            and _normalized_class_name(edit.class_directory)
            != _normalized_class_name(source.class_name)
        ):
            raise ValueError(
                "Official artifact class name does not match TFDS: "
                f"artifact={edit.class_directory!r}, "
                f"tfds={source.class_name!r}, label={edit.label}."
            )

    inventory_fingerprint = _inventory_fingerprint(edits)
    import_config = {
        "artifact_ref": artifact_ref,
        "artifact_root": str(Path(artifact_dir).expanduser().resolve()),
        "dataset": dataset,
        "importer": "official_alia_cub_v1",
        "inventory_fingerprint": inventory_fingerprint,
        "prompt": prompt,
        "source_layout": layout_name,
        "validation_split": validation_split,
    }
    fingerprint = config_fingerprint(import_config)
    records = []
    excluded_source_ids: set[int] = set()
    kept_source_ids: set[int] = set()
    class_counts: Counter[int] = Counter()
    for edit in edits:
        source = source_mapping[edit.source_index]
        if source.is_validation:
            excluded_source_ids.add(edit.source_index)
            continue
        kept_source_ids.add(edit.source_index)
        class_counts[edit.label] += 1
        identity = (
            f"{artifact_ref}\0{edit.relative_path}\0{inventory_fingerprint}"
        )
        record_id = "official-alia-" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]
        records.append(
            {
                "schema_version": ALIA_SCHEMA_VERSION,
                "method": "alia",
                "stage": "generated",
                "record_id": record_id,
                "image_path": str(edit.path),
                "output_png_sha256": sha256_file(edit.path),
                "dataset": dataset,
                "source_id": f"{dataset}:{source.global_index:09d}",
                "source_ref": source.filename,
                "source_partition": "train",
                "augmentation_index": edit.variant_index,
                "label": edit.label,
                "class_name": source.class_name,
                "prompt": prompt,
                "validation_split": validation_split,
                "config_fingerprint": fingerprint,
                "official_artifact_ref": artifact_ref,
                "official_relative_path": edit.relative_path,
                "official_source_index": edit.source_index,
                "official_source_layout": layout_name,
                "official_variant_index": edit.variant_index,
            }
        )
    if not records:
        raise ValueError("Every official ALIA edit was excluded as validation leakage.")
    missing_classes = [
        label for label in range(_CUB_NUM_CLASSES) if label not in class_counts
    ]
    if missing_classes:
        raise ValueError(
            "Leakage-safe official ALIA import is missing classes: "
            f"{missing_classes}."
        )
    records.sort(key=lambda record: str(record["record_id"]))

    atomic_write_json(
        summary_path,
        {"complete": False, "stage": "generated", "record_count": 0},
    )
    atomic_write_jsonl(manifest_path, records)
    summary: dict[str, object] = {
        "complete": True,
        "stage": "generated",
        "record_count": len(records),
        "artifact_records": len(edits),
        "artifact_ref": artifact_ref,
        "artifact_root": str(Path(artifact_dir).expanduser().resolve()),
        "class_root": str(class_root),
        "config_fingerprint": fingerprint,
        "dataset": dataset,
        "excluded_validation_records": len(edits) - len(records),
        "excluded_validation_sources": len(excluded_source_ids),
        "imported_sources": len(kept_source_ids),
        "inventory_fingerprint": inventory_fingerprint,
        "ignored_preview_pngs": ignored_preview_pngs,
        "num_classes": len(class_counts),
        "source_layout": layout_name,
        "validation_split": validation_split,
    }
    atomic_write_json(summary_path, summary)

    return summary


__all__ = [
    "OFFICIAL_CUB_ARTIFACT_REF",
    "CubTrainSource",
    "OfficialCubEdit",
    "import_official_cub_artifact",
    "match_official_cub_source_layout",
    "scan_official_cub_artifact",
]
