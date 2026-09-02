from __future__ import annotations

from types import SimpleNamespace

from allthemix.competitors.generative import source_preflight
from allthemix.competitors.generative.sources import (
    iter_allthemix_sources,
)


def test_prepare_generation_dataset_uses_isolated_python(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(source_preflight.sys, "executable", "/venv/python")
    monkeypatch.setattr(source_preflight.subprocess, "run", fake_run)

    source_preflight.prepare_generation_dataset(
        dataset="caltech_birds2011",
        data_dir="./data",
    )

    assert calls == [
        (
            [
                "/venv/python",
                "-m",
                "allthemix.competitors.generative.source_preflight",
                "--dataset",
                "caltech_birds2011",
                "--data-dir",
                "./data",
            ],
            True,
        )
    ]


def test_prepare_generation_dataset_skips_class_folder_source(
    monkeypatch,
) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("class-folder sources need no TFDS preflight")

    monkeypatch.setattr(source_preflight.subprocess, "run", fail_run)

    source_preflight.prepare_generation_dataset(dataset="", data_dir="./data")


def test_preflight_enables_mutating_download_once(monkeypatch) -> None:
    import allthemix.data.datasets.loader as loader

    calls = []

    def fake_load_train_dataset(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(loader, "load_train_dataset", fake_load_train_dataset)

    summary = source_preflight._prepare_in_current_process(
        dataset="caltech_birds2011",
        data_dir="./data",
    )

    assert calls == [
        {
            "name": "caltech_birds2011",
            "data_dir": "./data",
            "shuffle_files": False,
            "download": True,
        }
    ]
    assert summary["prepared"] is True


def test_source_worker_forbids_tfds_download(monkeypatch) -> None:
    import allthemix.competitors.generative.sources as sources
    import allthemix.data.datasets.loader as loader

    calls = []

    class FakeDataset:
        def as_numpy_iterator(self):
            return iter(())

    def fake_load_train_dataset(**kwargs):
        calls.append(kwargs)
        return FakeDataset()

    monkeypatch.setattr(loader, "load_train_dataset", fake_load_train_dataset)
    monkeypatch.setattr(sources, "_tfds_label_names", lambda **kwargs: ())

    records = list(
        iter_allthemix_sources(
            dataset="caltech_birds2011",
            data_dir="./data",
            validation_split=0.1,
            download=False,
        )
    )

    assert records == []
    assert calls[0]["download"] is False


def test_saspa_preflight_happens_before_xla_launch(monkeypatch) -> None:
    import allthemix.competitors.generative.torch_xla as torch_xla
    import allthemix.competitors.saspa.cli as cli

    events = []
    args = SimpleNamespace(
        command="generate",
        data_dir="./data",
        dataset="caltech_birds2011",
        device="auto",
        num_shards=4,
        xla_launch=True,
    )

    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        source_preflight,
        "prepare_generation_dataset",
        lambda **kwargs: events.append(("prepare", kwargs)),
    )
    monkeypatch.setattr(
        torch_xla,
        "import_torch_xla",
        lambda: SimpleNamespace(
            launch=lambda function, args: events.append("launch")
        ),
    )

    cli.main([])

    assert events[0] == (
        "prepare",
        {"dataset": "caltech_birds2011", "data_dir": "./data"},
    )
    assert events[1] == "launch"
    assert args.device == "xla"
    assert args.num_shards == 0


def test_alia_preflight_happens_before_xla_launch(monkeypatch) -> None:
    import allthemix.competitors.alia.cli as cli
    import allthemix.competitors.generative.torch_xla as torch_xla

    events = []
    args = SimpleNamespace(
        command="edit",
        data_dir="./data",
        dataset="caltech_birds2011",
        device="auto",
        num_shards=4,
        train_dir="",
        xla_launch=True,
    )

    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        source_preflight,
        "prepare_generation_dataset",
        lambda **kwargs: events.append(("prepare", kwargs)),
    )
    monkeypatch.setattr(
        torch_xla,
        "import_torch_xla",
        lambda: SimpleNamespace(
            launch=lambda function, args: events.append("launch")
        ),
    )

    cli.main([])

    assert events[0] == (
        "prepare",
        {"dataset": "caltech_birds2011", "data_dir": "./data"},
    )
    assert events[1] == "launch"


def test_diffusemix_preflight_happens_before_xla_launch(monkeypatch) -> None:
    import allthemix.competitors.diffusemix.generate as generate

    events = []
    args = SimpleNamespace(
        data_dir="./data",
        dataset="caltech_birds2011",
        device="auto",
        num_shards=4,
        xla_launch=True,
    )

    monkeypatch.setattr(generate, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        source_preflight,
        "prepare_generation_dataset",
        lambda **kwargs: events.append(("prepare", kwargs)),
    )
    monkeypatch.setattr(
        generate,
        "import_torch_xla",
        lambda: SimpleNamespace(
            launch=lambda function, args: events.append("launch")
        ),
    )

    generate.main([])

    assert events[0] == (
        "prepare",
        {"dataset": "caltech_birds2011", "data_dir": "./data"},
    )
    assert events[1] == "launch"
