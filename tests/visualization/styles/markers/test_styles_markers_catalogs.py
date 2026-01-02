import pytest
from feupy.visualization.styles.markers.catalogs import map_catalog_to_marker

def test_map_catalog_to_marker():
    assert map_catalog_to_marker("hgps") == "p"
    assert map_catalog_to_marker("fgl") == "v"

    with pytest.raises(ValueError):
        map_catalog_to_marker("unknown")
