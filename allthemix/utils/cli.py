"""Shared command-line parsing helpers."""

from __future__ import annotations

import argparse


def str2bool(value: bool | str) -> bool:
    """Parse common command-line boolean spellings."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


__all__ = ["str2bool"]
