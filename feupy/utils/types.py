# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for types validation."""

from typing import Annotated, Tuple
from pydantic.functional_validators import BeforeValidator

__all__ = ["validate_irf", "IrfType"]


def validate_irf(v):
    from feupy.irf.manager import CTAOIRFManager

    if isinstance(v, list):
        v = tuple(v)

    if not isinstance(v, tuple):
        raise TypeError("IRF must be a tuple")

    if len(v) != 4:
        raise ValueError("IRF must have 4 elements")

    manager = CTAOIRFManager()

    try:
        manager._build_path(v)
    except Exception as e:
        raise ValueError(f"Invalid IRF: {v}") from e

    return v


IrfType = Annotated[
    Tuple[str, str, str, str],
    BeforeValidator(validate_irf),
]