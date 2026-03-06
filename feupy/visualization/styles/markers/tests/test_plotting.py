# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.visualization.styles.markers import make_marker_dict

def test_make_marker_dict():
    labels = ["a", "b"]
    markers = make_marker_dict(labels, marker="o")

    assert len(markers) == 2
    assert markers["a"]["marker"] == "o"


