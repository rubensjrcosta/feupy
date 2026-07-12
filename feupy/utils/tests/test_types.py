# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest

from feupy.utils.types import validate_irf


VALID_IRFS = [
    ("a", "b", "c", "d"),
    ("e", "f", "g", "h"),
]


class FakeManager:
    @classmethod
    def get_irfs_options(cls):
        return VALID_IRFS


def test_validate_irf_valid_tuple(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManager,
    )

    v = ("a", "b", "c", "d")
    assert validate_irf(v) == v


def test_validate_irf_valid_list(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManager,
    )

    v = ["a", "b", "c", "d"]
    assert validate_irf(v) == ("a", "b", "c", "d")


def test_validate_irf_invalid_type(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManager,
    )

    with pytest.raises(TypeError):
        validate_irf("not-a-tuple")


def test_validate_irf_invalid_length(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManager,
    )

    with pytest.raises(ValueError):
        validate_irf(("a", "b", "c"))


def test_validate_irf_invalid_option(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManager,
    )

    with pytest.raises(ValueError):
        validate_irf(("x", "y", "z", "w"))