# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u

from feupy.utils.constants import (
    UNIT_DEG,
    FRAME_ICRS,
    FRAME_FK5,
    CU,
)


def test_unit_deg():
    assert UNIT_DEG == "deg"


def test_frames():
    assert FRAME_ICRS == "icrs"
    assert FRAME_FK5 == "fk5"


def test_cu_unit():
    assert CU.unit == u.Unit("TeV-1 cm-2 s-1")


def test_cu_value():
    assert CU.value == 6.1e-17