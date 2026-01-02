# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.visualization.styles.markers import write_marker_dict, read_marker_dict

def test_marker_io(tmp_path):
    data = {"a": {"marker": "o"}}
    file = tmp_path / "markers.yaml"

    write_marker_dict(data, file)
    loaded = read_marker_dict(file)

    assert loaded == data
