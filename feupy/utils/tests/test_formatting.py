# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u

from feupy.utils import formatting
from feupy.utils.formatting import (
    string_to_filename,
    energy_to_string,
    energy_range_to_string,
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
        
def test_string_to_filename_spaces():
    assert string_to_filename("Crab Nebula") == "Crab_Nebula"


def test_string_to_filename_colon():
    assert string_to_filename("HESS J1825:137") == "HESS_J1825:137"


def test_string_to_filename_colon_strict():
    assert string_to_filename("HESS J1825:137", strict=True) == "HESS_J1825137"


def test_energy_to_string_tev():
    assert energy_to_string(2 * u.TeV) == "2TeV"


def test_energy_to_string_auto_unit():
    assert energy_to_string(2000 * u.GeV) == "2TeV"


def test_energy_to_string_integer_format():
    assert energy_to_string(10 * u.GeV) == "10GeV"


def test_energy_range_to_string():
    result = energy_range_to_string(100 * u.GeV, 10 * u.TeV)
    assert result == "100GeV_10TeV"


def test_energy_range_same_unit():
    result = energy_range_to_string(1 * u.TeV, 10 * u.TeV)
    assert result == "1TeV_10TeV"