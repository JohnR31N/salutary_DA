"""Prepare shared datasets before launching PyTorch/XLA workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def prepare_generation_dataset(
    dataset: str,
    data_dir: str,
) -> None:
    """Prepare one dataset in an isolated process before XLA fan-out.

    TFDS preparation mutates a shared ``incomplete.*`` directory. Running it
    independently in every XLA worker can make several processes race on the
    final directory rename. The launched workers therefore use read-only
    loading after this subprocess completes.
    """
    if not dataset:
        return

    command = [
        sys.executable,
        "-m",
        "allthemix.competitors.generative.source_preflight",
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
    ]
    print(
        f"[source-preflight] preparing dataset={dataset!r} "
        f"under data_dir={data_dir!r}",
        flush=True,
    )
    subprocess.run(command, check=True)
    print(
        f"[source-preflight] dataset={dataset!r} is ready for "
        "read-only XLA workers",
        flush=True,
    )


def _prepare_in_current_process(
    dataset: str,
    data_dir: str,
) -> dict[str, Any]:
    """Perform the mutable TFDS preparation in this short-lived process."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Keep TensorFlow and its TPU plugin out of the long-lived torch_xla
    # launcher process. This import intentionally happens only here.
    from allthemix.data.datasets.loader import load_train_dataset

    load_train_dataset(
        name=dataset,
        data_dir=data_dir,
        shuffle_files=False,
        download=True,
    )

    return {
        "data_dir": str(Path(data_dir).expanduser()),
        "dataset": dataset,
        "prepared": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an AllTheMix generation source dataset.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="./data")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Prepare the requested dataset and print a compact completion record."""
    args = _parse_args(argv)
    summary = _prepare_in_current_process(
        dataset=args.dataset,
        data_dir=args.data_dir,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
