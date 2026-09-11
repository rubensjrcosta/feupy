# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from feupy.irf.utils import get_irf_groups


class FakeIRFManager:
    """Fake CTAOIRFManager used for testing."""

    def get_irf(self, options):
        return {"irf": tuple(options)}


@pytest.fixture
def fake_manager(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.utils.CTAOIRFManager",
        lambda: FakeIRFManager(),
    )


def test_get_irf_groups_single_values(fake_manager):
    groups = ["South", "AverageAz", "20deg", "0.5h"]

    required_irfs, irfs = get_irf_groups(groups)

    expected = [["South", "AverageAz", "20deg", "0.5h"]]

    assert required_irfs == expected
    assert irfs == [tuple(expected[0])]


def test_get_irf_groups_multiple_arrays(fake_manager):
    groups = [
        ["South", "North"],
        "AverageAz",
        "20deg",
        "0.5h",
    ]

    required_irfs, irfs = get_irf_groups(groups)

    expected = [
        ["South", "AverageAz", "20deg", "0.5h"],
        ["North", "AverageAz", "20deg", "0.5h"],
    ]

    assert required_irfs == expected
    assert irfs == [tuple(x) for x in expected]


def test_get_irf_groups_multiple_parameters(fake_manager):
    groups = [
        ["South", "North"],
        ["AverageAz", "NorthAz"],
        ["20deg", "40deg"],
        "0.5h",
    ]

    required_irfs, irfs = get_irf_groups(groups)

    assert len(required_irfs) == 8
    assert len(irfs) == 8

    assert ["South", "AverageAz", "20deg", "0.5h"] in required_irfs
    assert ["North", "NorthAz", "40deg", "0.5h"] in required_irfs

    assert ("South", "AverageAz", "20deg", "0.5h") in irfs
    assert ("North", "NorthAz", "40deg", "0.5h") in irfs


def test_get_irf_groups_all_lists(fake_manager):
    groups = [
        ["South"],
        ["AverageAz"],
        ["20deg"],
        ["0.5h"],
    ]

    required_irfs, irfs = get_irf_groups(groups)

    assert required_irfs == [["South", "AverageAz", "20deg", "0.5h"]]
    assert irfs == [("South", "AverageAz", "20deg", "0.5h")]
