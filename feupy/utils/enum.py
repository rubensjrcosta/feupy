# Licensed under a 3-clause BSD style license - see LICENSE
"""Enumeration classes used across FeuPy."""

from enum import Enum

__all__ = [
    "TableEnum",
    "NaimaFunctionalModelsEnum",
    "ParticleTypeEnum",
]


class TableEnum(str, Enum):
    """Supported table formats."""

    csv = "csv"
    fits = "fits"


class NaimaFunctionalModelsEnum(str, Enum):
    """Supported Naima functional particle-distribution models."""

    PowerLaw = "PowerLaw"
    ExponentialCutoffPowerLaw = "ExponentialCutoffPowerLaw"
    BrokenPowerLaw = "BrokenPowerLaw"
    ExponentialCutoffBrokenPowerLaw = "ExponentialCutoffBrokenPowerLaw"
    LogParabola = "LogParabola"


class ParticleTypeEnum(str, Enum):
    """Supported particle types."""

    electrons = "electrons"
    protons = "protons"
    both = "both"
