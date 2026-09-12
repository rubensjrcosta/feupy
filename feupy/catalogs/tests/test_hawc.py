# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from feupy.catalogs.hawc import (
    SourceCatalogEHWC,
    SourceCatalogExtraHAWC,
    create_flux_points_table_2hwc,
    create_flux_points_table_3hwc,
    get_flux_points_2hwc,
    get_flux_points_3hwc,
)


@pytest.fixture
def fake_3hwc_source():
    source = MagicMock()
    source.name = "3HWC J0000+000"
    source.data = {
        "search_radius": 1.0 * u.deg,
        "spec0_radius": 0.5 * u.deg,
        "spec0_dnde": 1.0e-14 / (u.TeV * u.cm**2 * u.s),
        "spec0_dnde_errn": -2.0e-15 / (u.TeV * u.cm**2 * u.s),
        "spec0_dnde_errp": 3.0e-15 / (u.TeV * u.cm**2 * u.s),
    }
    return source


@pytest.fixture
def fake_2hwc_source():
    source = MagicMock()
    source.name = "2HWC J0000+000"
    source.n_models = 2

    spectral_model = MagicMock()
    spectral_model.return_value = u.Quantity([1.0e-14], "TeV-1 cm-2 s-1")
    spectral_model.evaluate_error.return_value = (
        u.Quantity([1.0e-14], "TeV-1 cm-2 s-1"),
        u.Quantity([1.0e-15], "TeV-1 cm-2 s-1"),
    )
    source.spectral_model.return_value = spectral_model
    return source


def _fake_3hwc_catalog():
    table = Table()
    table["spec0_dnde"] = [1.0] * u.Unit("TeV-1 cm-2 s-1")
    table["spec0_dnde_errn"] = [0.1] * u.Unit("TeV-1 cm-2 s-1")
    table["spec0_dnde_errp"] = [0.1] * u.Unit("TeV-1 cm-2 s-1")

    for column in table.colnames:
        table[column].description = column
        table[column].format = ".3e"

    table.meta["catalog_name"] = "3HWC"
    table.meta["reference"] = "test"
    return MagicMock(table=table)


def _fake_2hwc_catalog():
    table = Table()
    table.meta["catalog_name"] = "2HWC"
    table.meta["reference"] = "test"
    return MagicMock(table=table)


def test_create_flux_points_table_3hwc(fake_3hwc_source):
    with patch(
        "feupy.catalogs.hawc.SourceCatalog3HWC",
        return_value=_fake_3hwc_catalog(),
    ):
        table = create_flux_points_table_3hwc(fake_3hwc_source)

    assert table.meta["SED_TYPE"] == "dnde"
    assert table.meta["source_name"] == fake_3hwc_source.name
    assert len(table) == 1
    assert np.isclose(table["e_ref"].quantity[0].to_value("TeV"), 7.0)
    assert not table["is_ul"][0]
    assert np.isnan(table["dnde_ul"][0])


def test_get_flux_points_3hwc(fake_3hwc_source):
    table = Table()
    table.meta["SED_TYPE"] = "dnde"
    spectral_model = fake_3hwc_source.spectral_model.return_value
    expected = MagicMock()

    with (
        patch(
            "feupy.catalogs.hawc.create_flux_points_table_3hwc",
            return_value=table,
        ),
        patch(
            "feupy.catalogs.hawc.FluxPoints.from_table",
            return_value=expected,
        ) as from_table,
    ):
        result = get_flux_points_3hwc(fake_3hwc_source)

    from_table.assert_called_once_with(
        table,
        sed_type="dnde",
        reference_model=spectral_model,
    )
    assert result is expected


def test_create_flux_points_table_2hwc(fake_2hwc_source):
    with patch(
        "feupy.catalogs.hawc.SourceCatalog2HWC",
        return_value=_fake_2hwc_catalog(),
    ):
        table = create_flux_points_table_2hwc(fake_2hwc_source)

    assert table.meta["SED_TYPE"] == "dnde"
    assert table.meta["source_name"] == fake_2hwc_source.name
    assert len(table) == 1
    assert np.isclose(table["e_ref"].quantity[0].to_value("TeV"), 7.0)
    assert not table["is_ul"][0]


def test_create_flux_points_table_2hwc_extended_missing(fake_2hwc_source):
    fake_2hwc_source.n_models = 1

    with (
        patch(
            "feupy.catalogs.hawc.SourceCatalog2HWC",
            return_value=_fake_2hwc_catalog(),
        ),
        pytest.raises(ValueError, match="No extended model"),
    ):
        create_flux_points_table_2hwc(
            fake_2hwc_source,
            which="extended",
        )


def test_get_flux_points_2hwc(fake_2hwc_source):
    table = Table()
    table.meta["SED_TYPE"] = "dnde"
    spectral_model = fake_2hwc_source.spectral_model.return_value
    expected = MagicMock()

    with (
        patch(
            "feupy.catalogs.hawc.create_flux_points_table_2hwc",
            return_value=table,
        ),
        patch(
            "feupy.catalogs.hawc.FluxPoints.from_table",
            return_value=expected,
        ) as from_table,
    ):
        result = get_flux_points_2hwc(fake_2hwc_source, which="extended")

    fake_2hwc_source.spectral_model.assert_called_once_with(which="extended")
    from_table.assert_called_once_with(
        table,
        sed_type="dnde",
        reference_model=spectral_model,
    )
    assert result is expected


def test_source_catalog_ehwc(tmp_path):
    table = Table()
    table["source_name"] = ["eHWC J0000+000"]
    table.meta["catalog_name"] = "eHWC"
    table.meta["comments"] = ["test"]

    filename = tmp_path / "ehwc.ecsv"
    table.write(filename, format="ascii.ecsv")

    catalog = SourceCatalogEHWC(filename=filename)

    assert len(catalog.table) == 1
    assert catalog.tag == "ehwc"


def test_source_catalog_extra_hawc(tmp_path):
    table = Table()
    table["source_name"] = ["HAWC J0000+000"]

    filename = tmp_path / "extra_hawc.ecsv"
    table.write(filename, format="ascii.ecsv")

    catalog = SourceCatalogExtraHAWC(filename=filename)

    assert len(catalog.table) == 1
    assert catalog.tag == "hwc-2021ApJ"
