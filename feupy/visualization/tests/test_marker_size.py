# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from feupy.visualization.styles.markers.marker_size import (
    MARKERS_DEFAULT_DICT,
    resolve_marker_size,
)


def test_marker_size_defaults():
    assert MARKERS_DEFAULT_DICT["o"] == (1.0,)
    assert MARKERS_DEFAULT_DICT["*"] == (1.6,)
    assert MARKERS_DEFAULT_DICT[">"] == (1.2,)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("o", 10.0),
        ("^", 12.0),
        ("*", 16.0),
        ("p", 14.0),
        ("D", 11.0),
    ],
)
def test_resolve_marker_size(marker, expected):
    assert resolve_marker_size(marker, base_size=10.0) == expected


def test_resolve_marker_size_without_normalization():
    assert resolve_marker_size("*", base_size=10.0, uniform_size=False) == 10.0


def test_resolve_marker_size_unknown_marker():
    assert resolve_marker_size("x", base_size=10.0) == 10.0


def test_resolve_marker_size_rounding():
    assert resolve_marker_size("^", base_size=3.333) == 4.0
