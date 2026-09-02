"""Deterministic and crash-safe helpers for offline image artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically for fingerprints and metadata."""
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_fingerprint(payload: Any) -> str:
    """Return the SHA-256 fingerprint of one JSON-compatible config."""
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8"),
    ).hexdigest()


def stable_seed(global_seed: int, *parts: object) -> int:
    """Derive a stable signed-31-bit seed from a job identity."""
    digest = hashlib.sha256()
    digest.update(str(int(global_seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))

    return int.from_bytes(
        digest.digest()[:8],
        byteorder="big",
        signed=False,
    ) % (2**31 - 1)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _atomic_replace_text(path: Path, text: str) -> None:
    """Write text beside its destination and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically write one readable JSON object."""
    _atomic_replace_text(
        Path(path),
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Atomically write a sequence of deterministic JSONL records."""
    lines = [canonical_json(record) for record in records]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    _atomic_replace_text(Path(path), text)


def append_jsonl_records(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Durably append completed records to a streaming manifest."""
    lines = [canonical_json(record) for record in records]
    if not lines:
        return
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    """Durably append one completed record to a streaming manifest."""
    append_jsonl_records(path, (record,))
