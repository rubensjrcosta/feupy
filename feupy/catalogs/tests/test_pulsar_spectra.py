# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.table import Table

from feupy.catalogs.pulsar_spectra import (
    create_pulsar_flux_points_table,
    get_pulsar_flux_points_datasets,
    get_pulsar_flux_points_tables,
    read_pulsar_spectra_catalog,
)

# =====================================================
# Fixtures
# =====================================================


@pytest.fixture
def sample_catalog(tmp_path, monkeypatch):
    import yaml

    data_dir = tmp_path / "catalogs" / "pulsar_spectra"
    data_dir.mkdir(parents=True, exist_ok=True)

    catalog_file = data_dir / "pulsar_spectra.yaml"

    sample_data = {
        "J0000+0000": [
            [100, 200],  # freqs
            None,  # bands
            [1.0, 2.0],  # fluxs
            [0.1, 0.2],  # flux_errs
            ["ref1", "ref2"],  # refs
        ]
    }

    with open(catalog_file, "w") as f:
        yaml.safe_dump(sample_data, f)

    monkeypatch.setenv("FEUPY_DATA", str(tmp_path))

    return catalog_file


# =====================================================
# Tests
# =====================================================


def test_read_pulsar_catalog(sample_catalog):
    catalog = read_pulsar_spectra_catalog()

    assert isinstance(catalog, dict)
    assert "J0000+0000" in catalog


def test_create_flux_points_table(sample_catalog):
    table = create_pulsar_flux_points_table("J0000+0000")

    assert isinstance(table, Table)
    assert "e_ref" in table.colnames
    assert "e2dnde" in table.colnames


def test_create_flux_points_table_missing_source(sample_catalog):
    table = create_pulsar_flux_points_table("UNKNOWN")

    assert table is None


def test_get_pulsar_flux_points_tables(sample_catalog):
    tables = get_pulsar_flux_points_tables("J0000+0000")

    assert isinstance(tables, list)

    if len(tables) > 0:
        assert isinstance(tables[0], Table)


def test_get_pulsar_flux_points_datasets(sample_catalog):
    datasets = get_pulsar_flux_points_datasets("J0000+0000")

    assert datasets is not None


# =====================================================
# Edge cases
# =====================================================


def test_catalog_file_not_found(monkeypatch):
    monkeypatch.setenv("FEUPY_DATA", "/non/existent/path")

    catalog = read_pulsar_spectra_catalog()

    assert isinstance(catalog, dict)
