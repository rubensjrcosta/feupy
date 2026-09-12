# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import patch

import pytest

from feupy.visualization.styles.markers.registry import CatalogStyle
from feupy.visualization.styles.markers.resolver import (
    extract_catalog_tag,
    resolve_marker,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Crab (hgps)", "hgps"),
        ("Source (4FGL)", "4fgl"),
        ("PSR J0000+0000 (psrcat)", "psrcat"),
        ("source", None),
        (None, None),
    ],
)
def test_extract_catalog_tag(label, expected):
    assert extract_catalog_tag(label) == expected


def test_extract_catalog_tag_uses_first_parentheses():
    assert extract_catalog_tag("Source (4fgl) extra (hgps)") == "4fgl"


def test_resolve_marker_registered_catalog():
    with patch(
        "feupy.visualization.styles.markers.resolver.CATALOG_STYLE_REGISTRY.get",
        return_value=CatalogStyle("hgps", marker="p"),
    ) as mock_get:
        marker = resolve_marker("Source (HGPS)")

    assert marker == "p"
    mock_get.assert_called_once_with("hgps")


def test_resolve_marker_without_catalog_tag():
    with patch(
        "feupy.visualization.styles.markers.resolver.CATALOG_STYLE_REGISTRY.get"
    ) as mock_get:
        marker = resolve_marker("Source", default="s")

    assert marker == "s"
    mock_get.assert_not_called()


def test_resolve_marker_unknown_catalog():
    with patch(
        "feupy.visualization.styles.markers.resolver.CATALOG_STYLE_REGISTRY.get",
        return_value=None,
    ):
        marker = resolve_marker("Source (unknown)", default="D")

    assert marker == "D"
