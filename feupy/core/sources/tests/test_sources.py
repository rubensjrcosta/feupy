# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import pytest
import yaml
from astropy.coordinates import SkyCoord

from feupy.core.sources import Sources


class FakeSource:
    """Minimal source object for testing."""

    def __init__(self, name, ra=0, dec=0):
        self.name = name
        self.position = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")


class FakeCatalog:
    """Minimal catalog registry entry for testing."""

    source_object_class = FakeSource


@pytest.fixture
def source_a():
    return FakeSource("Crab", 83.63, 22.01)


@pytest.fixture
def source_b():
    return FakeSource("Vela", 128.75, -45.2)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    import feupy.core.sources.sources as sources_module

    monkeypatch.setattr(
        sources_module,
        "FEUPY_CATALOG_REGISTRY",
        [FakeCatalog],
    )
    monkeypatch.setattr(
        sources_module,
        "get_catalog_tag",
        lambda source: "fakecat",
    )


def test_sources_init_empty():
    sources = Sources()

    assert len(sources) == 0


def test_sources_init_single(source_a):
    sources = Sources(source_a)

    assert len(sources) == 1
    assert sources.names == ["Crab"]


def test_sources_init_list(source_a, source_b):
    sources = Sources([source_a, source_b])

    assert len(sources) == 2
    assert sources.names == ["Crab", "Vela"]


def test_sources_init_from_sources(source_a):
    original = Sources([source_a])
    sources = Sources(original)

    assert sources.names == ["Crab"]


def test_sources_duplicate(source_a):
    with pytest.raises(ValueError, match="already exists"):
        Sources([source_a, source_a])


def test_getitem_by_index(source_a):
    sources = Sources([source_a])

    assert sources[0] is source_a


def test_getitem_by_name(source_a):
    sources = Sources([source_a])

    assert sources["Crab"] is source_a


def test_setitem(source_a, source_b):
    sources = Sources([source_a])

    sources[0] = source_b

    assert sources.names == ["Vela"]


def test_setitem_invalid_type(source_a):
    sources = Sources([source_a])

    with pytest.raises(TypeError):
        sources[0] = "invalid"


def test_delitem(source_a, source_b):
    sources = Sources([source_a, source_b])

    del sources["Crab"]

    assert sources.names == ["Vela"]


def test_insert(source_a, source_b):
    sources = Sources([source_a])

    sources.insert(1, source_b)

    assert sources.names == ["Crab", "Vela"]


def test_insert_duplicate(source_a):
    sources = Sources([source_a])

    with pytest.raises(ValueError, match="already exists"):
        sources.insert(1, source_a)


def test_index(source_a):
    sources = Sources([source_a])

    assert sources.index(0) == 0
    assert sources.index("Crab") == 0
    assert sources.index(source_a) == 0


def test_len(source_a, source_b):
    assert len(Sources([source_a, source_b])) == 2


def test_copy(source_a):
    sources = Sources([source_a])

    copied = sources.copy()

    assert copied is not sources
    assert copied.names == sources.names
    assert copied[0] is not sources[0]


def test_labels(source_a):
    sources = Sources([source_a])

    assert sources.labels == ["Crab (fakecat)"]


def test_positions(source_a, source_b):
    positions = Sources([source_a, source_b]).positions

    assert len(positions) == 2
    assert positions[0].ra.deg == pytest.approx(83.63)
    assert positions[1].dec.deg == pytest.approx(-45.2)


def test_select(source_a, source_b):
    selected = Sources([source_a, source_b]).select(["Crab"])

    assert selected.names == ["Crab"]


def test_invalid_init_type():
    with pytest.raises(TypeError):
        Sources("invalid")


def test_invalid_insert_type(source_a):
    sources = Sources([source_a])

    with pytest.raises(TypeError):
        sources.insert(0, "invalid")


def test_write(tmp_path, source_a, source_b):
    filename = tmp_path / "sources.yaml"
    sources = Sources([source_a, source_b])

    sources.write(filename)

    with filename.open() as stream:
        data = yaml.safe_load(stream)

    assert data["Sources"][0] == {
        "name": "Crab",
        "catalog": "fakecat",
    }
    assert data["Sources"][1] == {
        "name": "Vela",
        "catalog": "fakecat",
    }


def test_write_existing_file(tmp_path, source_a):
    filename = tmp_path / "sources.yaml"
    filename.write_text("existing")

    with pytest.raises(OSError, match="File exists"):
        Sources([source_a]).write(filename)


def test_write_overwrite(tmp_path, source_a):
    filename = tmp_path / "sources.yaml"
    filename.write_text("existing")

    Sources([source_a]).write(filename, overwrite=True)

    assert "Crab" in filename.read_text()
