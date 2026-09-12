# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import pytest

from feupy.utils.constants import CU, FRAME_FK5, FRAME_ICRS, UNIT_DEG


def test_unit_deg():
    assert UNIT_DEG == "deg"


def test_coordinate_frames():
    assert FRAME_ICRS == "icrs"
    assert FRAME_FK5 == "fk5"


def test_cu():
    assert CU.unit == u.Unit("TeV-1 cm-2 s-1")
    assert CU.value == pytest.approx(6.1e-17)
