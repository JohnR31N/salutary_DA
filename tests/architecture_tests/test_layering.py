"""Freeze the package-level import graph so the structure cannot rot.

The map below is the complete set of allowed cross-package edges inside
``allthemix``. Adding a new cross-package import fails this test until the
edge is deliberately added here (with review). Two legacy inversions are
recorded explicitly; do not add more.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "allthemix"

# Complete allowed edge set: source package -> destination packages.
ALLOWED_EDGES = {
    "cli": {
        "competitors",
        "config",
        "data",
        "diagnostics",
        "methods",
        "networks",
        "training",
        "utils",
    },
    "competitors": {"config", "data", "methods", "networks", "training", "utils"},
    # LEGACY INVERSION (cleanup scheduled): diffusemix manifest validation
    # inside the training pipeline. Do not extend.
    "data": {"competitors", "utils"},
    "debug": {"methods", "networks", "training"},
    "diagnostics": {"competitors", "data"},
    # "utils" added in S2#5: engine loops now reuse utils.parallel.shard_array
    # (deliberate downward edge, reviewed).
    "training": {"methods", "networks", "utils"},
    "utils": set(),
    "visualize": {"config", "data", "methods", "utils"},
    "methods": set(),
    "networks": set(),
    "config": set(),
}


def _package_of(module_path: Path) -> str:
    relative = module_path.relative_to(PACKAGE_ROOT)
    if len(relative.parts) == 1:
        return "(root)"
    return relative.parts[0]


def _module_edges() -> dict[tuple[str, str], list[str]]:
    edges: dict[tuple[str, str], list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source_package = _package_of(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("allthemix")
            ):
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("allthemix")
                )
            for module in modules:
                parts = module.split(".")
                destination = parts[1] if len(parts) > 1 else "(root)"
                if destination in ("(root)", source_package):
                    continue
                edges.setdefault((source_package, destination), []).append(
                    str(path.relative_to(REPO_ROOT))
                )
    return edges


def test_cross_package_imports_stay_inside_the_allowed_graph() -> None:
    violations = []
    for (source, destination), files in sorted(_module_edges().items()):
        if destination not in ALLOWED_EDGES.get(source, set()):
            violations.append(
                f"{source} -> {destination} (from {sorted(set(files))[:3]})"
            )
    assert not violations, (
        "New cross-package import edges detected; either remove them or "
        "deliberately extend ALLOWED_EDGES with review:\n"
        + "\n".join(violations)
    )


def test_core_never_imports_competitors_beyond_recorded_legacy_edges() -> None:
    core_sources_allowed_to_touch_competitors = {"cli", "data", "diagnostics"}
    offenders = {
        source
        for (source, destination) in _module_edges()
        if destination == "competitors"
        and source not in core_sources_allowed_to_touch_competitors
    }
    assert not offenders, (
        "competitors/ must stay parallel to the core library; new importers: "
        f"{sorted(offenders)}"
    )
