"""Tests for shared command-line parsing helpers."""

import argparse

import pytest

from allthemix.utils.cli import str2bool


@pytest.mark.parametrize("value", [True, "true", "yes", "y", "1"])
def test_str2bool_accepts_true_values(value: bool | str) -> None:
    assert str2bool(value) is True


@pytest.mark.parametrize("value", [False, "false", "no", "n", "0"])
def test_str2bool_accepts_false_values(value: bool | str) -> None:
    assert str2bool(value) is False


def test_str2bool_rejects_unknown_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("sometimes")
