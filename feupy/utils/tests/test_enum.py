# Licensed under a 3-clause BSD style license - see LICENSE

import pytest

from feupy.utils.enum import (
    NaimaFunctionalModelsEnum,
    ParticleTypeEnum,
    TableEnum,
)


def test_table_enum_values():
    assert TableEnum.csv.value == "csv"
    assert TableEnum.fits.value == "fits"


def test_table_enum_from_string():
    assert TableEnum("csv") is TableEnum.csv
    assert TableEnum("fits") is TableEnum.fits


def test_naima_functional_models_enum_values():
    assert NaimaFunctionalModelsEnum.PowerLaw.value == "PowerLaw"
    assert (
        NaimaFunctionalModelsEnum.ExponentialCutoffPowerLaw.value
        == "ExponentialCutoffPowerLaw"
    )
    assert NaimaFunctionalModelsEnum.BrokenPowerLaw.value == "BrokenPowerLaw"
    assert (
        NaimaFunctionalModelsEnum.ExponentialCutoffBrokenPowerLaw.value
        == "ExponentialCutoffBrokenPowerLaw"
    )
    assert NaimaFunctionalModelsEnum.LogParabola.value == "LogParabola"


@pytest.mark.parametrize(
    ("value", "member"),
    [
        ("PowerLaw", NaimaFunctionalModelsEnum.PowerLaw),
        (
            "ExponentialCutoffPowerLaw",
            NaimaFunctionalModelsEnum.ExponentialCutoffPowerLaw,
        ),
        ("BrokenPowerLaw", NaimaFunctionalModelsEnum.BrokenPowerLaw),
        (
            "ExponentialCutoffBrokenPowerLaw",
            NaimaFunctionalModelsEnum.ExponentialCutoffBrokenPowerLaw,
        ),
        ("LogParabola", NaimaFunctionalModelsEnum.LogParabola),
    ],
)
def test_naima_functional_models_enum_from_string(value, member):
    assert NaimaFunctionalModelsEnum(value) is member


def test_particle_type_enum_values():
    assert ParticleTypeEnum.electrons.value == "electrons"
    assert ParticleTypeEnum.protons.value == "protons"
    assert ParticleTypeEnum.both.value == "both"


@pytest.mark.parametrize(
    ("value", "member"),
    [
        ("electrons", ParticleTypeEnum.electrons),
        ("protons", ParticleTypeEnum.protons),
        ("both", ParticleTypeEnum.both),
    ],
)
def test_particle_type_enum_from_string(value, member):
    assert ParticleTypeEnum(value) is member


@pytest.mark.parametrize(
    ("enum_class", "invalid_value"),
    [
        (TableEnum, "ecsv"),
        (NaimaFunctionalModelsEnum, "InvalidModel"),
        (ParticleTypeEnum, "photons"),
    ],
)
def test_invalid_enum_value(enum_class, invalid_value):
    with pytest.raises(ValueError):
        enum_class(invalid_value)
