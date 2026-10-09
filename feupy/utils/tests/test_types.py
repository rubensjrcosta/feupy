# Licensed under a 3-clause BSD style license - see LICENSE

import pytest

from feupy.utils import types
from feupy.utils.types import validate_irf


PROD5_IRFS = [
    ("a", "b", "c", "d"),
    ("e", "f", "g", "h"),
]

PROD6_IRFS = [
    ("a", "b", "c", "d"),
    ("i", "j", "k", "l"),
]


class FakeManager:
    """Minimal CTAO IRF manager used for validation tests."""

    def __init__(
        self,
        production="prod5",
        condition="dark",
    ):
        self.production = production
        self.condition = condition

    def get_irfs_options(self):
        if self.production == "prod5":
            return PROD5_IRFS

        if self.production == "prod6":
            return PROD6_IRFS

        return []


@pytest.fixture
def mock_irf_manager(monkeypatch):
    monkeypatch.setattr(
        "feupy.irf.manager.CTAOIRFManager",
        FakeManager,
    )


def test_all():
    expected = {
        "validate_irf",
        "IrfType",
    }

    assert set(types.__all__) == expected

    for name in types.__all__:
        assert hasattr(types, name)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            ("a", "b", "c", "d"),
            ("a", "b", "c", "d"),
        ),
        (
            ["a", "b", "c", "d"],
            ("a", "b", "c", "d"),
        ),
        (
            ("e", "f", "g", "h"),
            ("e", "f", "g", "h"),
        ),
        (
            ("i", "j", "k", "l"),
            ("i", "j", "k", "l"),
        ),
    ],
)
def test_validate_irf_valid(
    mock_irf_manager,
    value,
    expected,
):
    assert validate_irf(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "not-a-tuple",
        42,
        None,
    ],
)
def test_validate_irf_invalid_type(
    mock_irf_manager,
    value,
):
    with pytest.raises(
        TypeError,
        match="tuple or list",
    ):
        validate_irf(value)


@pytest.mark.parametrize(
    "value",
    [
        ("a", "b", "c"),
        ("a", "b", "c", "d", "e"),
        [],
    ],
)
def test_validate_irf_invalid_length(
    mock_irf_manager,
    value,
):
    with pytest.raises(
        ValueError,
        match="exactly 4 elements",
    ):
        validate_irf(value)


def test_validate_irf_invalid_option(
    mock_irf_manager,
):
    value = ("x", "y", "z", "w")

    with pytest.raises(
        ValueError,
        match="Invalid IRF option",
    ):
        validate_irf(value)