"""Contact sheets for auditing filtered ALIA image quality."""

from __future__ import annotations

import csv
import textwrap
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from allthemix.competitors.alia.ablation import resolve_paired_sources
from allthemix.competitors.alia.manifest import read_stage_records
from allthemix.competitors.generative.artifacts import atomic_write_json
from allthemix.competitors.generative.sources import SourceExample


def _score(record: dict[str, object]) -> tuple[int, float, float, float]:
    """Return class fidelity first, then classifier and CLIP confidence."""
    label = int(record["label"])
    prediction = int(record.get("classifier_predicted_label", -1))
    agreement = int(label == prediction)
    assigned_probability = float(
        record.get("classifier_assigned_label_probability", -1.0)
    )
    clip_probability = float(record.get("clip_positive_probability", -1.0))
    max_probability = float(record.get("classifier_max_probability", -1.0))

    return agreement, assigned_probability, clip_probability, max_probability


def rank_quality_records(
    records: Iterable[dict[str, object]],
    ranking: str = "best",
    max_per_class: int = 1,
    num_samples: int = 24,
) -> list[dict[str, object]]:
    """Select class-diverse best or worst accepted generated records."""
    if ranking not in {"best", "worst"}:
        raise ValueError("ranking must be 'best' or 'worst'.")
    if max_per_class == 0 or max_per_class < -1:
        raise ValueError("max_per_class must be positive or -1 for no cap.")
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    def key(record: dict[str, object]):
        score = _score(record)
        numeric = tuple(-value for value in score) if ranking == "best" else score
        return (*numeric, str(record["record_id"]))

    selected = []
    class_counts: Counter[int] = Counter()
    for record in sorted(records, key=key):
        label = int(record["label"])
        if max_per_class > 0 and class_counts[label] >= max_per_class:
            continue
        selected.append(record)
        class_counts[label] += 1
        if len(selected) == num_samples:
            break

    if not selected:
        raise ValueError("No ALIA records were available for visualization.")

    return selected


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Use Pillow's bundled DejaVu font with a portable fallback."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    """Fit a complete image into a square without hiding edit failures."""
    canvas = Image.new("RGB", (size, size), "white")
    fitted = ImageOps.contain(image.convert("RGB"), (size, size))
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    canvas.paste(fitted, offset)

    return canvas


def _changed_ratio(original: Image.Image, generated: Image.Image) -> float:
    """Measure normalized mean absolute pixel change at display resolution."""
    size = (224, 224)
    first = np.asarray(original.convert("RGB").resize(size), dtype=np.float32)
    second = np.asarray(generated.convert("RGB").resize(size), dtype=np.float32)

    return float(np.mean(np.abs(first - second)) / 255.0)


def _open_generated(record: dict[str, object]) -> Image.Image:
    """Open one generated image detached from its underlying file handle."""
    with Image.open(str(record["resolved_image_path"])) as image:
        return image.convert("RGB").copy()


def _write_ranking_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write exact ranking metadata beside the visual contact sheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def visualize_filtered_quality(
    artifact_dir: str | Path,
    output_path: str | Path,
    dataset: str,
    data_dir: str,
    validation_split: float,
    num_samples: int = 24,
    ranking: str = "best",
    max_per_class: int = 1,
    pairs_per_row: int = 3,
    tile_size: int = 224,
    sources: Iterable[SourceExample] | None = None,
) -> dict[str, object]:
    """Render accepted edits beside the exact original source images."""
    if pairs_per_row < 1 or tile_size < 64:
        raise ValueError("pairs_per_row must be positive and tile_size >= 64.")
    records = read_stage_records(
        artifact_dir,
        stage="final",
        check_images=True,
        require_complete=True,
    )
    selected = rank_quality_records(
        records,
        ranking=ranking,
        max_per_class=max_per_class,
        num_samples=num_samples,
    )
    paired_sources = resolve_paired_sources(
        source_ids=(str(record["source_id"]) for record in selected),
        dataset=dataset,
        data_dir=data_dir,
        validation_split=validation_split,
        sources=sources,
    )

    header_height = 72
    caption_height = 118
    row_height = tile_size + caption_height
    row_count = (len(selected) + pairs_per_row - 1) // pairs_per_row
    pair_width = tile_size * 2
    canvas = Image.new(
        "RGB",
        (pair_width * pairs_per_row, header_height + row_height * row_count),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24, bold=True)
    label_font = _font(15, bold=True)
    detail_font = _font(13)
    draw.text(
        (16, 12),
        f"ALIA filtered images: {ranking} classifier-supported quality",
        fill="black",
        font=title_font,
    )
    draw.text(
        (16, 44),
        "Each pair: original source (left) / generated edit (right)",
        fill=(70, 70, 70),
        font=detail_font,
    )

    ranking_rows: list[dict[str, object]] = []
    for index, record in enumerate(selected):
        grid_row, grid_column = divmod(index, pairs_per_row)
        x = grid_column * pair_width
        y = header_height + grid_row * row_height
        source = paired_sources[str(record["source_id"])]
        original = source.image.convert("RGB")
        generated = _open_generated(record)
        canvas.paste(_letterbox(original, tile_size), (x, y))
        canvas.paste(_letterbox(generated, tile_size), (x + tile_size, y))
        draw.rectangle((x, y, x + 74, y + 22), fill="white")
        draw.rectangle(
            (x + tile_size, y, x + tile_size + 92, y + 22),
            fill="white",
        )
        draw.text((x + 5, y + 3), "original", fill="black", font=label_font)
        draw.text(
            (x + tile_size + 5, y + 3),
            "generated",
            fill="black",
            font=label_font,
        )

        label = int(record["label"])
        prediction = int(record.get("classifier_predicted_label", -1))
        assigned_probability = float(
            record.get("classifier_assigned_label_probability", -1.0)
        )
        max_probability = float(record.get("classifier_max_probability", -1.0))
        clip_probability = float(record.get("clip_positive_probability", -1.0))
        change = _changed_ratio(original, generated)
        class_name = source.class_name
        details = (
            f"#{index + 1} class={label} {class_name} | match="
            f"{'yes' if prediction == label else 'no'} pred={prediction}\n"
            f"p(label)={assigned_probability:.3f}  p(max)={max_probability:.3f}  "
            f"CLIP={clip_probability:.3f}  pixel_change={change:.3f}"
        )
        prompt = str(record.get("prompt", ""))
        prompt_lines = textwrap.wrap(prompt, width=66)[:2]
        draw.multiline_text(
            (x + 6, y + tile_size + 7),
            details + "\n" + "\n".join(prompt_lines),
            fill="black",
            font=detail_font,
            spacing=3,
        )
        ranking_rows.append(
            {
                "rank": index + 1,
                "record_id": str(record["record_id"]),
                "source_id": str(record["source_id"]),
                "label": label,
                "class_name": class_name,
                "predicted_label": prediction,
                "label_agreement": int(prediction == label),
                "assigned_label_probability": assigned_probability,
                "max_probability": max_probability,
                "clip_positive_probability": clip_probability,
                "class_confident_threshold": float(
                    record.get("class_confident_threshold", -1.0)
                ),
                "pixel_change": change,
                "prompt": prompt,
                "generated_image_path": str(record["resolved_image_path"]),
            }
        )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    csv_path = output.with_suffix(".csv")
    _write_ranking_csv(csv_path, ranking_rows)
    summary = {
        "artifact_dir": str(Path(artifact_dir).expanduser().resolve()),
        "class_diversity": len({int(row["label"]) for row in ranking_rows}),
        "label_agreement_count": sum(
            int(row["label_agreement"]) for row in ranking_rows
        ),
        "num_samples": len(ranking_rows),
        "output_csv": str(csv_path),
        "output_png": str(output),
        "ranking": ranking,
    }
    atomic_write_json(output.with_suffix(".json"), summary)

    return summary
