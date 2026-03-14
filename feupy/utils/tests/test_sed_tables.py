import astropy.units as u

from feupy.utils.sed_tables import (
    ENERGY_COLUMNS,
    SED_COLUMNS,
    DEFAULT_ENERGY_UNIT,
    DEFAULT_SED_UNIT,
)


def test_energy_columns_keys():
    assert "dnde" in ENERGY_COLUMNS
    assert "e2dnde" in ENERGY_COLUMNS


def test_sed_columns_keys():
    assert "dnde" in SED_COLUMNS
    assert "e2dnde" in SED_COLUMNS


def test_default_energy_unit():
    assert DEFAULT_ENERGY_UNIT["dnde"] == u.TeV


def test_default_sed_unit():
    assert DEFAULT_SED_UNIT["dnde"] == u.Unit("cm-2 s-1 TeV-1")