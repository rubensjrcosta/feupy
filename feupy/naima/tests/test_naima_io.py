# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.datasets import FluxPointsDataset
import astropy.units as u

from feupy.naima.io import (
    REQUIRED_NAIMA_COLUMNS_NAMES,
    REQUIRED_NAIMA_COLUMNS,
    make_naima_tables,
)


def test_required_columns_structure():
    """Test that required column definitions are consistent."""

    assert isinstance(REQUIRED_NAIMA_COLUMNS_NAMES, dict)
    assert isinstance(REQUIRED_NAIMA_COLUMNS, dict)

    for sed_type, columns in REQUIRED_NAIMA_COLUMNS.items():

        assert isinstance(columns, list)

        for column in columns:
            assert column in REQUIRED_NAIMA_COLUMNS_NAMES


def make_test_dataset(name="test-dataset"):
    """Create a minimal FluxPointsDataset for testing."""

    table = Table()

    table["e_ref"] = [1.0] * u.TeV
    table["dnde"] = [1e-12] * u.Unit("1 / (cm2 s TeV)")
    table["dnde_err"] = [1e-13] * u.Unit("1 / (cm2 s TeV)")

    flux_points = FluxPoints.from_table(
        table,
        sed_type="dnde"
    )

    return FluxPointsDataset(
        data=flux_points,
        name=name,
    )


def test_make_naima_tables_single_dataset():
    """Test conversion of a single FluxPointsDataset."""

    dataset = make_test_dataset()

    tables = make_naima_tables([dataset])

    assert isinstance(tables, list)
    assert len(tables) == 1
    assert isinstance(tables[0], Table)

    table = tables[0]

    assert "energy" in table.colnames
    assert "flux" in table.colnames
    assert "flux_error" in table.colnames


def test_make_naima_tables_multiple_datasets():
    """Test conversion of multiple FluxPointsDataset."""

    ds1 = make_test_dataset("ds1")
    ds2 = make_test_dataset("ds2")

    tables = make_naima_tables([ds1, ds2])

    assert isinstance(tables, list)
    assert len(tables) == 2
    assert all(isinstance(t, Table) for t in tables)