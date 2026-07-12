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
        raise ValueError("IRF must have exactly 4 elements")

    options = CTAOIRFManager.get_irfs_options()

    if v not in options:
        raise ValueError(
            f"Invalid IRF option: {v!r}. Choose one from: {options!r}"
        )

    return v


IrfType = Annotated[
    Tuple[str, str, str, str],
    BeforeValidator(validate_irf),
]