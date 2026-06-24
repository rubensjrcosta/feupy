# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest

from feupy.utils.types import validate_irf, IrfType


class FakeManager:
    def _build_path(self, v):
        return True


class FakeManagerFail:
    def _build_path(self, v):
        raise RuntimeError("broken IRF")


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


def test_validate_irf_manager_failure(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManagerFail,
    )

    with pytest.raises(ValueError):
        validate_irf(("a", "b", "c", "d"))