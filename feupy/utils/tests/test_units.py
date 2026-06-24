# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u

from feupy.utils.units import is_energy_unit


def test_is_energy_unit_true():
    assert is_energy_unit(u.eV)
    assert is_energy_unit(u.TeV)
    assert is_energy_unit(u.erg)


def test_is_energy_unit_false():
    assert not is_energy_unit(u.cm)
    assert not is_energy_unit(u.s)
    assert not is_energy_unit(u.deg)