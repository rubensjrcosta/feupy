# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from feupy.core.sources import Sources


# -------------------------------------------------------------------------
# Fake source object
# -------------------------------------------------------------------------
class FakeSource:
    def __init__(self, name, ra=0, dec=0):
        self.name = name
        self.position = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")


# -------------------------------------------------------------------------
# Fake catalog registry entry
# -------------------------------------------------------------------------
class FakeCatalog:
    source_object_class = FakeSource


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------
@pytest.fixture
def source_a():
    return FakeSource("Crab", 83.63, 22.01)


@pytest.fixture
def source_b():
    return FakeSource("Vela", 128.75, -45.2)


# -------------------------------------------------------------------------
# Monkeypatch registry and tag
# -------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):

    import feupy.core.sources.sources as sources_module

    monkeypatch.setattr(sources_module, "FEUPY_CATALOG_REGISTRY", [FakeCatalog])

    monkeypatch.setattr(sources_module, "get_catalog_tag", lambda source: "fakecat")


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------
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


def test_sources_duplicate(source_a):
    with pytest.raises(ValueError):
        Sources([source_a, source_a])


def test_insert(source_a, source_b):
    sources = Sources([source_a])

    sources.insert(1, source_b)

    assert len(sources) == 2
    assert sources.names == ["Crab", "Vela"]


def test_insert_duplicate(source_a):
    sources = Sources([source_a])

    with pytest.raises(ValueError):
        sources.insert(1, source_a)


def test_getitem_by_index(source_a):
    sources = Sources([source_a])

    assert sources[0].name == "Crab"


def test_getitem_by_name(source_a):
    sources = Sources([source_a])

    assert sources["Crab"].name == "Crab"


def test_index_by_name(source_a):
    sources = Sources([source_a])

    assert sources.index("Crab") == 0


def test_len(source_a, source_b):
    sources = Sources([source_a, source_b])

    assert len(sources) == 2


def test_copy(source_a):
    sources = Sources([source_a])

    copied = sources.copy()

    assert copied is not sources
    assert copied.names == sources.names


def test_labels(source_a):
    sources = Sources([source_a])

    assert sources.labels == ["Crab (fakecat)"]


def test_positions(source_a, source_b):
    sources = Sources([source_a, source_b])

    positions = sources.positions

    assert len(positions) == 2
    assert positions[0].ra.deg == pytest.approx(83.63)


def test_select(source_a, source_b):
    sources = Sources([source_a, source_b])

    selected = sources.select(["Crab"])

    assert len(selected) == 1
    assert selected.names == ["Crab"]


def test_invalid_init_type():
    with pytest.raises(TypeError):
        Sources("invalid")


def test_invalid_insert_type(source_a):
    sources = Sources([source_a])

    with pytest.raises(TypeError):
        sources.insert(0, "invalid")
