# Licensed under a 3-clause BSD style license - see LICENSE
"""Type definitions and validation utilities."""

from typing import Annotated

from pydantic.functional_validators import BeforeValidator

__all__ = [
    "validate_irf",
    "IrfType",
]


def validate_irf(value):
    """Validate a CTAO IRF configuration.

    Parameters
    ----------
    value : tuple or list
        IRF configuration containing array, azimuth, zenith, and livetime.

    Returns
    -------
    irf : tuple of str
        Validated IRF configuration.

    Raises
    ------
    TypeError
        If the input is not a tuple or list.
    ValueError
        If the IRF does not contain exactly four elements or is not a
        supported CTAO IRF configuration.
    """
    from feupy.irf.manager import CTAOIRFManager

    if isinstance(value, list):
        value = tuple(value)

    if not isinstance(value, tuple):
        raise TypeError("IRF must be a tuple or list.")

    if len(value) != 4:
        raise ValueError("IRF must have exactly 4 elements.")

    options = CTAOIRFManager.get_irfs_options()

    if value not in options:
        raise ValueError(f"Invalid IRF option: {value!r}. Choose one from: {options!r}")

    return value


IrfType = Annotated[
    tuple[str, str, str, str],
    BeforeValidator(validate_irf),
]
