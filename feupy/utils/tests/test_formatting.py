# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import pytest

from feupy.utils import formatting
from feupy.utils.formatting import (
    energy_range_to_string,
    energy_to_string,
    string_to_filename,
)


def test_all():
    expected = {
        "string_to_filename",
        "energy_to_string",
        "energy_range_to_string",
    }

    assert set(formatting.__all__) == expected

    for name in formatting.__all__:
        assert hasattr(formatting, name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Crab Nebula", "Crab_Nebula"),
        ("Crab   Nebula", "Crab_Nebula"),
        ("HESS J1825:137", "HESS_J1825:137"),
    ],
)
def test_string_to_filename(name, expected):
    assert string_to_filename(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("HESS J1825:137", "HESS_J1825137"),
        ("Crab (Nebula)", "Crab_Nebula"),
        ("source@test!", "sourcetest"),
    ],
)
def test_string_to_filename_strict(name, expected):
    assert string_to_filename(name, strict=True) == expected


@pytest.mark.parametrize(
    ("energy", "expected"),
    [
        (2 * u.TeV, "2TeV"),
        (2000 * u.GeV, "2TeV"),
        (10 * u.GeV, "10GeV"),
        (1000 * u.MeV, "1GeV"),
        (1 * u.PeV, "1PeV"),
        (0.5 * u.MeV, "0.5MeV"),
    ],
)
def test_energy_to_string(energy, expected):
    assert energy_to_string(energy) == expected


def test_energy_to_string_explicit_unit():
    result = energy_to_string(1000 * u.GeV, unit=u.TeV)

    assert result == "1TeV"


@pytest.mark.parametrize(
    ("energy_min", "energy_max", "expected"),
    [
        (100 * u.GeV, 10 * u.TeV, "100GeV_10TeV"),
        (1 * u.TeV, 10 * u.TeV, "1TeV_10TeV"),
    ],
)
def test_energy_range_to_string(energy_min, energy_max, expected):
    result = energy_range_to_string(energy_min, energy_max)

    assert result == expected
