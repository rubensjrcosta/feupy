# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import yaml
from astropy.table import Table
from gammapy.datasets import Datasets, FluxPointsDataset

from feupy.catalogs.pulsar_spectra import (
    create_pulsar_flux_points_table,
    get_pulsar_flux_points_datasets,
    get_pulsar_flux_points_table,
    get_pulsar_flux_points_tables,
    read_pulsar_spectra_catalog,
)


@pytest.fixture
def sample_catalog(tmp_path, monkeypatch):
    data_dir = tmp_path / "catalogs" / "pulsar_spectra"
    data_dir.mkdir(parents=True)

    filename = data_dir / "pulsar_spectra.yaml"
    data = {
        "J0000+0000": [
            [100, 200, 300],
            None,
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            ["ref1", "ref1", "ref2"],
        ]
    }

    with filename.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream)

    monkeypatch.setenv("FEUPY_DATA", str(tmp_path))
    return filename


def test_read_pulsar_spectra_catalog(sample_catalog):
    catalog = read_pulsar_spectra_catalog()

    assert isinstance(catalog, dict)
    assert "J0000+0000" in catalog
    assert len(catalog["J0000+0000"][0]) == 3


def test_read_pulsar_spectra_catalog_explicit_filename(sample_catalog):
    catalog = read_pulsar_spectra_catalog(sample_catalog)

    assert "J0000+0000" in catalog


def test_create_pulsar_flux_points_table(sample_catalog):
    table = create_pulsar_flux_points_table("J0000+0000")

    assert isinstance(table, Table)
    assert len(table) == 3
    assert table.colnames == ["ref", "e_ref", "e2dnde", "e2dnde_err"]
    assert table.meta["source_name"] == "PSR J0000+0000"
    assert table.meta["pulsar_jname"] == "J0000+0000"
    assert table.meta["sed_type"] == "e2dnde"
    assert table["e_ref"].unit.to_string() == "eV"
    assert table["e2dnde"].unit.to_string() == "erg / (s cm2)"
    assert table["e2dnde_err"].unit.to_string() == "erg / (s cm2)"


def test_create_pulsar_flux_points_table_missing_source(sample_catalog):
    assert create_pulsar_flux_points_table("UNKNOWN") is None


def test_get_pulsar_flux_points_table(sample_catalog):
    table = get_pulsar_flux_points_table("J0000+0000")

    assert isinstance(table, Table)
    assert len(table) == 3


def test_get_pulsar_flux_points_tables(sample_catalog):
    tables = get_pulsar_flux_points_tables("J0000+0000")

    assert len(tables) == 2
    assert all(isinstance(table, Table) for table in tables)
    assert {table.meta["ref"] for table in tables} == {"ref1", "ref2"}
    assert [len(table) for table in tables] == [2, 1]
    assert all("ref" not in table.colnames for table in tables)
    assert all(table.meta["sed_type"] == "e2dnde" for table in tables)


def test_get_pulsar_flux_points_tables_missing_source(sample_catalog):
    assert get_pulsar_flux_points_tables("UNKNOWN") == []


def test_get_pulsar_flux_points_datasets(sample_catalog):
    datasets = get_pulsar_flux_points_datasets("J0000+0000")

    assert isinstance(datasets, Datasets)
    assert len(datasets) == 2
    assert all(isinstance(dataset, FluxPointsDataset) for dataset in datasets)
    assert set(datasets.names) == {"ref1", "ref2"}


def test_get_pulsar_flux_points_datasets_missing_source(sample_catalog):
    datasets = get_pulsar_flux_points_datasets("UNKNOWN")

    assert isinstance(datasets, Datasets)
    assert len(datasets) == 0


def test_catalog_file_not_found(monkeypatch):
    monkeypatch.setenv("FEUPY_DATA", "/non/existent/path")

    assert read_pulsar_spectra_catalog() == {}


def test_invalid_yaml(tmp_path):
    filename = tmp_path / "invalid.yaml"
    filename.write_text("key: [invalid", encoding="utf-8")

    assert read_pulsar_spectra_catalog(filename) == {}
