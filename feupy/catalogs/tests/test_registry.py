# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
import sys

from gammapy.catalog import CATALOG_REGISTRY
from gammapy.utils.registry import Registry

MODULE = "feupy.catalogs.registry"


def _reload_registry(monkeypatch, feupy_data=None):
    if feupy_data is None:
        monkeypatch.delenv("FEUPY_DATA", raising=False)
    else:
        monkeypatch.setenv("FEUPY_DATA", str(feupy_data))

    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def test_registry_without_feupy_data(monkeypatch):
    registry = _reload_registry(monkeypatch)

    assert registry.HAS_FEUPY_DATASETS is False
    assert isinstance(registry.FEUPY_CATALOG_REGISTRY, Registry)
    assert len(registry.FEUPY_CATALOG_REGISTRY) == len(CATALOG_REGISTRY)


def test_registry_with_missing_feupy_data(monkeypatch, tmp_path):
    missing = tmp_path / "missing"

    registry = _reload_registry(monkeypatch, missing)

    assert registry.HAS_FEUPY_DATASETS is False
    assert len(registry.FEUPY_CATALOG_REGISTRY) == len(CATALOG_REGISTRY)


def test_registry_with_feupy_data(monkeypatch, tmp_path):
    data_path = tmp_path / "feupy-data"
    data_path.mkdir()

    registry = _reload_registry(monkeypatch, data_path)

    assert registry.HAS_FEUPY_DATASETS is True
    assert isinstance(registry.FEUPY_CATALOG_REGISTRY, Registry)

    expected = {
        "SourceCatalogEHWC",
        "SourceCatalogExtraHAWC",
        "SourceCatalogExtraHESS",
        "SourceCatalogVTSCat",
        "SourceCatalogVERITASCygnus",
        "SourceCatalogPSRCAT",
        "SourceCatalogLHAASO",
        "SourceCatalogExtraLHAASO",
    }

    names = {catalog.__name__ for catalog in registry.FEUPY_CATALOG_REGISTRY}
    assert expected.issubset(names)


def test_registry_exports(monkeypatch):
    registry = _reload_registry(monkeypatch)

    assert registry.__all__ == [
        "FEUPY_CATALOG_REGISTRY",
        "HAS_FEUPY_DATASETS",
    ]
