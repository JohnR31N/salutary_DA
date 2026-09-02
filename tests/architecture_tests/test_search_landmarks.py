"""Lock the searchable boundaries around the SalDA implementation."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LANDMARKS_BY_FILE = {
    "salutary_da/scorers/gradient_alignment.py": (
        "GA HARD-LABEL GAIN PROJECTION",
        "GA FULL-PARAMETER VALIDATION DIRECTION",
        "GA FULL-PARAMETER JVP",
        "GA CLASSIFIER-HEAD VALIDATION DIRECTION",
        "GA CLASSIFIER-HEAD DIRECTIONAL DERIVATIVE",
    ),
    "salutary_da/gradient_alignment_strategy.py": (
        "GA VALIDATION DIRECTION SCHEDULE",
        "GA STEP DIRECTION REFRESH",
        "GA STEP SCORE DISPATCH",
        "GA STEP POLICY DECISION",
        "GA STEP ACTION UPDATE",
    ),
    "salutary_da/policies/per_row_continuous.py": (
        "GA POLICY ROW SELECTION",
        "GA POLICY ACTION MATERIALIZATION",
    ),
    "allthemix/cli/train.py": (
        "SALDA VTEST PRELOAD EXCLUSION",
        "SALDA PRE-ENDPOINT WORKLOAD CLOSURE",
        "SALDA BEST-VDEV CHECKPOINT GATE",
        "SALDA SEALED VTEST DATASET GATE",
        "SALDA BEST-VDEV RESTORE",
        "SALDA SEALED VTEST DATASET OPEN",
        "SHARED FINAL-TEST EVALUATION",
    ),
}
MARKER_PATTERN = re.compile(
    r"^[ \t]*# #### (?P<name>[A-Z][A-Z0-9 -]*): "
    r"(?P<edge>START|END) ####[ \t]*$",
    flags=re.MULTILINE,
)


def _marker(name: str, edge: str) -> str:
    return f"# #### {name}: {edge} ####"


def _source(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def _landmark_block(relative_path: str, name: str) -> str:
    source = _source(relative_path)
    start = source.index(_marker(name, "START"))
    end = source.index(_marker(name, "END"), start)
    return source[start:end]


def _all_source_paths() -> list[Path]:
    return sorted(
        path
        for directory in ("allthemix", "salutary_da")
        for path in (REPOSITORY_ROOT / directory).rglob("*.py")
    )


def test_salda_search_landmark_registry_matches_all_source_markers() -> None:
    """Reject unregistered, duplicated, crossed, or misplaced landmarks."""

    expected = Counter(
        (relative_path, name, edge)
        for relative_path, names in LANDMARKS_BY_FILE.items()
        for name in names
        for edge in ("START", "END")
    )
    observed = Counter()
    markers_by_file = defaultdict(list)
    for path in _all_source_paths():
        relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        for match in MARKER_PATTERN.finditer(source):
            name = match.group("name")
            edge = match.group("edge")
            observed[(relative_path, name, edge)] += 1
            markers_by_file[relative_path].append((name, edge))

    assert observed == expected
    for relative_path, markers in markers_by_file.items():
        stack = []
        for name, edge in markers:
            if edge == "START":
                stack.append(name)
            else:
                assert stack and stack[-1] == name, (relative_path, markers)
                stack.pop()
        assert not stack, (relative_path, stack)


def test_full_parameter_jvp_landmark_contains_every_generic_jvp() -> None:
    """Keep both generic JVP calls inside their named search boundary."""

    relative_path = "salutary_da/scorers/gradient_alignment.py"
    source = _source(relative_path)
    block = _landmark_block(relative_path, "GA FULL-PARAMETER JVP")
    assert source.count("jax.jvp(") == 2
    assert block.count("jax.jvp(") == 2


def test_classifier_head_landmark_contains_both_closed_form_tangents() -> None:
    """Keep the affine-head derivative distinct from generic JVP execution."""

    relative_path = "salutary_da/scorers/gradient_alignment.py"
    source = _source(relative_path)
    block = _landmark_block(
        relative_path,
        "GA CLASSIFIER-HEAD DIRECTIONAL DERIVATIVE",
    )
    formula = 'features @ direction["kernel"] + direction["bias"]'
    assert source.count(formula) == 2
    assert block.count(formula) == 2
    assert "jax.jvp(" not in block


def test_endpoint_landmarks_contain_their_defining_operations() -> None:
    """Keep Vtest sealing, opening, restore, and evaluation labels accurate."""

    relative_path = "allthemix/cli/train.py"
    preload = _landmark_block(relative_path, "SALDA VTEST PRELOAD EXCLUSION")
    restore = _landmark_block(relative_path, "SALDA BEST-VDEV RESTORE")
    dataset_open = _landmark_block(
        relative_path,
        "SALDA SEALED VTEST DATASET OPEN",
    )
    evaluation = _landmark_block(relative_path, "SHARED FINAL-TEST EVALUATION")

    assert "include_final_test=(" in preload
    assert "not salda_ga_active" in preload
    assert "_salda_directory_sha256(" in restore
    assert "restore_checkpoint(" in restore
    assert "_validate_salda_best_checkpoint_pre_endpoint(" in restore
    assert "_build_salda_endpoint_after_closure(" in dataset_open
    assert "parallel_evaluate(" in evaluation
    assert re.search(r"\) = evaluate\(", evaluation)


def test_salda_readme_landmark_index_exactly_matches_registry() -> None:
    """Keep the human-facing index equal to the source registry both ways."""

    readme = _source("salutary_da/README.md")
    landmark_section = readme.split("## Search landmarks\n", maxsplit=1)[1]
    landmark_section = landmark_section.split("\n## ", maxsplit=1)[0]
    observed = re.findall(
        r"^- `([A-Z][A-Z0-9 -]+)`$",
        landmark_section,
        flags=re.MULTILINE,
    )
    expected = [
        name
        for names in LANDMARKS_BY_FILE.values()
        for name in names
    ]
    assert observed == expected
