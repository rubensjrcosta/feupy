# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import pytest
from astropy.table import Table
from gammapy.datasets import FluxPointsDataset
from gammapy.estimators import FluxPoints

from feupy.naima.io import (
    REQUIRED_NAIMA_COLUMNS,
    REQUIRED_NAIMA_COLUMNS_NAMES,
    make_naima_tables,
)


def make_test_dataset(name="test-dataset"):
    """Create a minimal flux-points dataset for testing."""
    table = Table()
    table["e_ref"] = [1.0, 2.0] * u.TeV
    table["dnde"] = [1e-12, 2e-12] * u.Unit("TeV-1 cm-2 s-1")
    table["dnde_err"] = [1e-13, 2e-13] * u.Unit("TeV-1 cm-2 s-1")

    flux_points = FluxPoints.from_table(table, sed_type="dnde")
    return FluxPointsDataset(data=flux_points, name=name)


def test_required_columns_structure():
    assert set(REQUIRED_NAIMA_COLUMNS) == {"dnde", "e2dnde"}

    for columns in REQUIRED_NAIMA_COLUMNS.values():
        assert isinstance(columns, list)
        assert all(column in REQUIRED_NAIMA_COLUMNS_NAMES for column in columns)


def test_make_naima_tables_single_dataset():
    dataset = make_test_dataset()

    tables = make_naima_tables([dataset])

    assert len(tables) == 1

    table = tables[0]
    assert isinstance(table, Table)
    assert table.meta["name"] == "test-dataset"
    assert table.colnames == ["energy", "flux", "flux_error"]
    assert len(table) == 2
    assert table["energy"].unit == u.TeV
    assert table["flux"].unit.is_equivalent(u.Unit("TeV-1 cm-2 s-1"))


def test_make_naima_tables_multiple_datasets():
    datasets = [
        make_test_dataset("ds1"),
        make_test_dataset("ds2"),
    ]

    tables = make_naima_tables(datasets)

    assert len(tables) == 2
    assert all(isinstance(table, Table) for table in tables)
    assert [table.meta["name"] for table in tables] == ["ds1", "ds2"]


def test_make_naima_tables_e2dnde():
    dataset = make_test_dataset()

    tables = make_naima_tables([dataset], sed_type="e2dnde")

    table = tables[0]
    assert "energy" in table.colnames
    assert "flux" in table.colnames
    assert table.meta["name"] == "test-dataset"


def test_make_naima_tables_empty():
    assert make_naima_tables([]) == []


def test_make_naima_tables_invalid_sed_type():
    with pytest.raises(ValueError, match="Invalid sed_type"):
        make_naima_tables([], sed_type="invalid")
