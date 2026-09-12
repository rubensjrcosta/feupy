# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u

from feupy.utils import sed
from feupy.utils.sed import (
    DEFAULT_ENERGY_UNIT,
    DEFAULT_SED_UNIT,
    ENERGY_COLUMNS,
    SED_COLUMNS,
)


def test_all():
    expected = {
        "ENERGY_COLUMNS",
        "SED_COLUMNS",
        "DEFAULT_ENERGY_UNIT",
        "DEFAULT_SED_UNIT",
    }

    assert set(sed.__all__) == expected

    for name in sed.__all__:
        assert hasattr(sed, name)


def test_energy_columns():
    assert ENERGY_COLUMNS == {
        "dnde": ["e_ref"],
        "e2dnde": ["e_ref"],
        "flux": ["e_min", "e_max", "flux_err"],
        "eflux": ["e_min", "e_max", "eflux_err"],
    }


def test_sed_columns():
    assert SED_COLUMNS == {
        "dnde": ["dnde", "dnde_err", "dnde_ul"],
        "e2dnde": ["e2dnde", "e2dnde_err", "e2dnde_ul"],
    }


def test_default_energy_units():
    assert DEFAULT_ENERGY_UNIT == {
        "dnde": u.TeV,
        "e2dnde": u.TeV,
    }


def test_default_sed_units():
    assert DEFAULT_SED_UNIT["dnde"] == u.Unit("cm-2 s-1 TeV-1")
    assert DEFAULT_SED_UNIT["e2dnde"] == u.Unit("erg cm-2 s-1")
