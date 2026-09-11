# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import pytest

from feupy.utils import units
from feupy.utils.units import is_energy_unit


def test_all():
    assert units.__all__ == ["is_energy_unit"]
    assert hasattr(units, "is_energy_unit")


@pytest.mark.parametrize(
    "unit",
    [
        u.eV,
        u.keV,
        u.MeV,
        u.GeV,
        u.TeV,
        u.erg,
        "TeV",
    ],
)
def test_is_energy_unit_true(unit):
    assert is_energy_unit(unit)


@pytest.mark.parametrize(
    "unit",
    [
        u.cm,
        u.s,
        u.deg,
        u.Hz,
        "cm",
        "invalid-unit",
        None,
    ],
)
def test_is_energy_unit_false(unit):
    assert not is_energy_unit(unit)
