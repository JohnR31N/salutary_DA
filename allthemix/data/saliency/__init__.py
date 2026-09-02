"""Saliency preprocessing package."""

from __future__ import annotations

from allthemix.data.saliency import saliency_io as _io
from allthemix.data.saliency import saliency_methods as _methods
from allthemix.data.saliency.saliency_io import *
from allthemix.data.saliency.saliency_maps import main, str2bool
from allthemix.data.saliency.saliency_methods import *
from allthemix.data.saliency.saliency_methods import (
    _to_grayscale_float,
)

__all__ = [
    *_methods.__all__,
    *_io.__all__,
    "main",
    "str2bool",
]
