"""Shared YAML configuration loading and inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_BASE_KEYS = frozenset({"base", "bases"})


def load_raw_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping without resolving inherited configs."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return config


def merge_config_dicts(
    base_config: dict[str, Any],
    override_config: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = dict(base_config)
    for key, override_value in override_config.items():
        if key in CONFIG_BASE_KEYS:
            continue
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = merge_config_dicts(base_value, override_value)
        else:
            merged[key] = override_value

    return merged


def _base_paths(config: dict[str, Any], config_path: Path) -> list[Path]:
    """Resolve one config's ordered base references."""
    values: list[object] = []
    if "base" in config:
        values.append(config["base"])
    if "bases" in config:
        bases = config["bases"]
        if not isinstance(bases, list):
            raise ValueError(
                f"'bases' must be a list in config file: {config_path}"
            )
        values.extend(bases)

    paths = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(
                "'base' and 'bases' entries must be strings in config file: "
                f"{config_path}"
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths.append(path.resolve())

    return paths


def load_yaml_config(
    config_path: str | Path,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load a YAML config and recursively merge ordered base configs."""
    path = Path(config_path).expanduser().resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Circular config base reference detected: {chain}")

    config = load_raw_yaml_config(path)
    merged: dict[str, Any] = {}
    stack = (*_stack, path)
    for base_path in _base_paths(config, path):
        merged = merge_config_dicts(
            merged,
            load_yaml_config(base_path, _stack=stack),
        )

    return merge_config_dicts(merged, config)


def load_optional_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load a supplied config, otherwise return an empty mapping."""
    if config_path is None or not str(config_path).strip():
        return {}

    return load_yaml_config(config_path)


def validate_config_keys(
    config: dict[str, Any],
    valid_keys: set[str],
    config_path: str | Path,
) -> None:
    """Reject unknown top-level YAML keys before applying defaults."""
    for key in config:
        if key not in valid_keys:
            raise ValueError(f"Unknown config key in {config_path}: {key}")


__all__ = [
    "CONFIG_BASE_KEYS",
    "load_optional_yaml_config",
    "load_raw_yaml_config",
    "load_yaml_config",
    "merge_config_dicts",
    "validate_config_keys",
]
