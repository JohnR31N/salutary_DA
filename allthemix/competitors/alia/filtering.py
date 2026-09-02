"""Official ALIA semantic and baseline-confidence filtering semantics."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

_CONFIDENT_THRESHOLD_LOWER_BOUND = 1.0e-6


def compute_confident_thresholds(
    base_scores: Iterable[dict[str, Any]],
    num_classes: int,
) -> np.ndarray:
    """Match Cleanlab's average per-class self-confidence thresholds."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive. Got {num_classes}.")
    values: list[list[float]] = [[] for _ in range(num_classes)]
    for record in base_scores:
        label = int(record["label"])
        probability = float(record["label_probability"])
        if not 0 <= label < num_classes:
            raise ValueError(f"Base score label is out of range: {label}.")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Base label_probability must be in [0, 1]: {probability}."
            )
        values[label].append(probability)

    missing = [index for index, class_values in enumerate(values) if not class_values]
    if missing:
        raise ValueError(
            "Base scores are missing classes needed for ALIA confidence "
            f"thresholds: {missing}."
        )

    return np.asarray(
        [
            max(_CONFIDENT_THRESHOLD_LOWER_BOUND, float(np.mean(class_values)))
            for class_values in values
        ],
        dtype=np.float64,
    )


def _deterministic_rank(record: dict[str, Any], seed: int) -> bytes:
    """Rank accepted records reproducibly without depending on input order."""
    return hashlib.sha256(
        f"{seed}\0{record['record_id']}".encode()
    ).digest()


def _cap_by_source(
    records: list[dict[str, Any]],
    max_per_source: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep a deterministic number of accepted edits for each source image."""
    if max_per_source < 0:
        return records, 0
    if max_per_source == 0:
        raise ValueError("max_per_source must be positive or negative for no cap.")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_id = str(record.get("source_id", ""))
        if not source_id:
            raise ValueError("Every ALIA record must identify its source image.")
        by_source[source_id].append(record)

    selected = []
    removed = 0
    for source_records in by_source.values():
        ranked = sorted(
            source_records,
            key=lambda record: _deterministic_rank(record, seed),
        )
        selected.extend(ranked[:max_per_source])
        removed += max(0, len(ranked) - max_per_source)

    selected.sort(key=lambda record: str(record["record_id"]))

    return selected, removed


def _cap_by_class(
    records: list[dict[str, Any]],
    base_scores: list[dict[str, Any]],
    extra_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select a class-balanced generated subset relative to train counts."""
    if extra_ratio < 0.0:
        return records, 0
    if not math.isfinite(extra_ratio):
        raise ValueError("extra_ratio must be finite or negative for no cap.")

    base_counts = Counter(int(record["label"]) for record in base_scores)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[int(record["label"])].append(record)

    selected = []
    removed = 0
    for label, class_records in sorted(by_class.items()):
        limit = int(round(base_counts[label] * extra_ratio))
        ranked = sorted(
            class_records,
            key=lambda record: _deterministic_rank(record, seed),
        )
        selected.extend(ranked[:limit])
        removed += max(0, len(ranked) - limit)

    selected.sort(key=lambda record: str(record["record_id"]))

    return selected, removed


def filter_generated_records(
    records: Iterable[dict[str, Any]],
    base_scores: Iterable[dict[str, Any]],
    num_classes: int,
    extra_ratio: float = 1.0,
    seed: int = 0,
    require_semantic_pass: bool = True,
    max_per_source: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the official union of semantic, easy, and mislabeled filters."""
    base_records = list(base_scores)
    thresholds = compute_confident_thresholds(
        base_scores=base_records,
        num_classes=num_classes,
    )
    accepted = []
    reasons = Counter()
    total = 0
    for raw_record in records:
        total += 1
        record = dict(raw_record)
        label = int(record["label"])
        predicted_label = int(record["classifier_predicted_label"])
        confidence = float(record["classifier_max_probability"])
        if not 0 <= label < num_classes or not 0 <= predicted_label < num_classes:
            raise ValueError(
                f"ALIA classifier score has an out-of-range label: "
                f"assigned={label}, predicted={predicted_label}."
            )
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid ALIA classifier confidence: {confidence}.")

        reason = "keep"
        if require_semantic_pass and record.get("semantic_pass") is not True:
            reason = "semantic"
        elif confidence > thresholds[label]:
            reason = "too_easy" if predicted_label == label else "mislabeled"

        record["class_confident_threshold"] = float(thresholds[label])
        record["filter_status"] = "keep" if reason == "keep" else "reject"
        record["filter_reason"] = reason
        reasons[reason] += 1
        if reason == "keep":
            accepted.append(record)

    accepted, source_capped = _cap_by_source(
        records=accepted,
        max_per_source=max_per_source,
        seed=seed,
    )
    accepted, capped = _cap_by_class(
        records=accepted,
        base_scores=base_records,
        extra_ratio=extra_ratio,
        seed=seed,
    )
    reasons["class_balance_cap"] = capped
    summary = {
        "total_records": total,
        "kept_before_cap": reasons["keep"],
        "kept_records": len(accepted),
        "rejected_semantic": reasons["semantic"],
        "rejected_too_easy": reasons["too_easy"],
        "rejected_mislabeled": reasons["mislabeled"],
        "removed_by_source_cap": source_capped,
        "removed_by_class_balance_cap": capped,
        "extra_ratio": extra_ratio,
        "max_per_source": max_per_source,
        "confident_thresholds": thresholds.tolist(),
    }

    return accepted, summary


def _strict_quality_rank(record: dict[str, Any]) -> tuple[float, float, str]:
    """Rank class-preserving edits by classifier support and semantics."""
    return (
        -float(record["classifier_assigned_label_probability"]),
        -float(record.get("clip_positive_probability", 0.0)),
        str(record["record_id"]),
    )


def _strict_cap_by_source(
    records: list[dict[str, Any]],
    max_per_source: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep the strongest strict edit for each source image."""
    if max_per_source < 0:
        return records, 0
    if max_per_source == 0:
        raise ValueError("max_per_source must be positive or -1 for no cap.")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_id = str(record.get("source_id", ""))
        if not source_id:
            raise ValueError("Every strict ALIA record must identify its source.")
        by_source[source_id].append(record)

    selected = []
    for source_records in by_source.values():
        selected.extend(
            sorted(source_records, key=_strict_quality_rank)[:max_per_source]
        )
    selected.sort(key=lambda record: str(record["record_id"]))

    return selected, len(records) - len(selected)


def _strict_cap_by_class(
    records: list[dict[str, Any]],
    per_class: int,
) -> tuple[list[dict[str, Any]], int]:
    """Take the strongest available edits without filling weak classes."""
    if per_class < 0:
        return records, 0
    if per_class == 0:
        raise ValueError("per_class must be positive or -1 for no cap.")

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[int(record["label"])].append(record)

    selected = []
    for class_records in by_class.values():
        selected.extend(sorted(class_records, key=_strict_quality_rank)[:per_class])
    selected.sort(key=lambda record: str(record["record_id"]))

    return selected, len(records) - len(selected)


def _strict_cap_total(
    records: list[dict[str, Any]],
    max_records: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Apply an exact, deterministic total budget with balanced classes."""
    if max_records < 0 or len(records) <= max_records:
        return records, 0
    if max_records == 0:
        raise ValueError("max_records must be positive or -1 for no cap.")

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[int(record["label"])].append(record)
    for class_records in by_class.values():
        class_records.sort(key=_strict_quality_rank)

    def class_order_key(label: int) -> tuple[bytes, int]:
        digest = hashlib.sha256(f"{seed}\0{label}".encode("ascii")).digest()
        return digest, label

    labels = sorted(by_class, key=class_order_key)
    selected = []
    rank = 0
    while len(selected) < max_records:
        added = False
        for label in labels:
            if rank >= len(by_class[label]):
                continue
            selected.append(by_class[label][rank])
            added = True
            if len(selected) == max_records:
                break
        if not added:
            break
        rank += 1
    selected.sort(key=lambda record: str(record["record_id"]))

    return selected, len(records) - len(selected)


def strict_filter_generated_records(
    records: Iterable[dict[str, Any]],
    base_scores: Iterable[dict[str, Any]],
    num_classes: int,
    min_assigned_probability: float = 0.2,
    per_class: int = 5,
    max_per_source: int = 1,
    max_records: int = -1,
    seed: int = 0,
    require_semantic_pass: bool = True,
    exclude_too_easy: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply an adapted class-fidelity-first ALIA filter.

    Unlike the released filter, this variant never retains a classifier label
    mismatch. It is intentionally an adapted ablation rather than a claim of
    reproducing the official ALIA selection rule.
    """
    if (
        not math.isfinite(min_assigned_probability)
        or not 0.0 <= min_assigned_probability <= 1.0
    ):
        raise ValueError("min_assigned_probability must be in [0, 1].")
    if per_class == 0 or per_class < -1:
        raise ValueError("per_class must be positive or -1 for no cap.")
    if max_per_source == 0 or max_per_source < -1:
        raise ValueError("max_per_source must be positive or -1 for no cap.")
    if max_records == 0 or max_records < -1:
        raise ValueError("max_records must be positive or -1 for no cap.")

    base_records = list(base_scores)
    thresholds = compute_confident_thresholds(
        base_scores=base_records,
        num_classes=num_classes,
    )
    eligible = []
    reasons = Counter()
    total = 0
    for raw_record in records:
        total += 1
        record = dict(raw_record)
        label = int(record["label"])
        predicted_label = int(record["classifier_predicted_label"])
        assigned_probability = float(
            record["classifier_assigned_label_probability"]
        )
        clip_probability = float(record.get("clip_positive_probability", 0.0))
        if not 0 <= label < num_classes or not 0 <= predicted_label < num_classes:
            raise ValueError(
                "Strict ALIA classifier score has an out-of-range label: "
                f"assigned={label}, predicted={predicted_label}."
            )
        for name, probability in (
            ("assigned", assigned_probability),
            ("CLIP", clip_probability),
        ):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Invalid strict ALIA {name} probability: {probability}."
                )

        reason = "strict_keep"
        if require_semantic_pass and record.get("semantic_pass") is not True:
            reason = "semantic"
        elif predicted_label != label:
            reason = "label_mismatch"
        elif assigned_probability < min_assigned_probability:
            reason = "low_assigned_probability"
        elif exclude_too_easy and assigned_probability > thresholds[label]:
            reason = "too_easy"

        record["class_confident_threshold"] = float(thresholds[label])
        record["filter_status"] = "keep" if reason == "strict_keep" else "reject"
        record["filter_reason"] = reason
        record["strict_filter"] = True
        reasons[reason] += 1
        if reason == "strict_keep":
            eligible.append(record)

    eligible_before_cap = len(eligible)
    accepted, source_capped = _strict_cap_by_source(
        records=eligible,
        max_per_source=max_per_source,
    )
    accepted, class_capped = _strict_cap_by_class(
        records=accepted,
        per_class=per_class,
    )
    accepted, total_capped = _strict_cap_total(
        records=accepted,
        max_records=max_records,
        seed=seed,
    )
    class_counts = Counter(int(record["label"]) for record in accepted)
    summary = {
        "total_records": total,
        "eligible_before_cap": eligible_before_cap,
        "kept_records": len(accepted),
        "rejected_semantic": reasons["semantic"],
        "rejected_label_mismatch": reasons["label_mismatch"],
        "rejected_low_assigned_probability": reasons[
            "low_assigned_probability"
        ],
        "rejected_too_easy": reasons["too_easy"],
        "removed_by_source_cap": source_capped,
        "removed_by_class_cap": class_capped,
        "removed_by_total_cap": total_capped,
        "classes_covered": len(class_counts),
        "missing_classes": [
            label for label in range(num_classes) if label not in class_counts
        ],
        "class_counts": {
            str(label): count for label, count in sorted(class_counts.items())
        },
        "min_assigned_probability": min_assigned_probability,
        "per_class": per_class,
        "max_per_source": max_per_source,
        "max_records": max_records,
        "seed": seed,
        "require_semantic_pass": require_semantic_pass,
        "exclude_too_easy": exclude_too_easy,
        "confident_thresholds": thresholds.tolist(),
    }

    return accepted, summary
