# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import patch

import pytest
from gammapy.utils.registry import Registry

from feupy.catalogs import utils


class FakeSource:
    pass


class FakeCatalog:
    tag = "fake"
    source_object_class = FakeSource

    def __init__(self):
        self.name = "fake"


class BrokenCatalog:
    tag = "broken"
    source_object_class = FakeSource

    def __init__(self):
        raise RuntimeError("boom")


def test_load_catalogs():
    registry = Registry([FakeCatalog])

    catalogs = utils.load_catalogs(registry)

    assert len(catalogs) == 1
    assert isinstance(catalogs[0], FakeCatalog)


def test_load_catalogs_default_registry():
    registry = Registry([FakeCatalog])

    with patch.object(utils, "FEUPY_CATALOG_REGISTRY", registry):
        catalogs = utils.load_catalogs()

    assert len(catalogs) == 1
    assert isinstance(catalogs[0], FakeCatalog)


def test_load_catalogs_failure():
    registry = Registry([BrokenCatalog])

    with pytest.raises(
        ValueError,
        match="Error loading catalog 'broken' at index 0",
    ):
        utils.load_catalogs(registry)


def test_get_catalog_tag():
    registry = Registry([FakeCatalog])

    with patch.object(utils, "FEUPY_CATALOG_REGISTRY", registry):
        tag = utils.get_catalog_tag(FakeSource())

    assert tag == "fake"


def test_get_catalog_tag_missing():
    registry = Registry([FakeCatalog])

    with (
        patch.object(utils, "FEUPY_CATALOG_REGISTRY", registry),
        pytest.raises(ValueError, match="No matching catalog found"),
    ):
        utils.get_catalog_tag(object())
