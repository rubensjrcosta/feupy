# Licensed under a 3-clause BSD style license - see LICENSE.rst

from matplotlib.markers import MarkerStyle

from feupy.visualization.styles.markers.defaults import DEFAULT_MARKERS
from feupy.visualization.styles.markers.registry import CATALOG_STYLE_REGISTRY


def test_default_markers_keys():
    assert set(DEFAULT_MARKERS) == {
        "psrcat",
        "2pc",
        "3pc",
        "gamma-cat",
        "hgps",
        "hess-2019a&a",
        "3fgl",
        "4fgl",
        "2fhl",
        "3fhl",
        "2hwc",
        "3hwc",
        "ehwc",
        "hwc-2021apj",
        "veritas-2018apj",
        "vtscat",
        "1lhaaso",
        "lhaaso",
    }


def test_default_markers_are_valid_matplotlib_markers():
    for marker in DEFAULT_MARKERS.values():
        MarkerStyle(marker)


def test_default_markers_are_registered():
    for tag, marker in DEFAULT_MARKERS.items():
        style = CATALOG_STYLE_REGISTRY.get(tag)

        assert style.tag == tag
        assert style.marker == marker
