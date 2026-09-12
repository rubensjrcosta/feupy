# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from gammapy.modeling.models import PowerLawSpectralModel

from feupy.catalogs.lhaaso import (
    SourceCatalogExtraLHAASO,
    SourceCatalogLHAASO,
    SourceCatalogObjectExtraLHAASO,
    SourceCatalogObjectLHAASO,
    create_flux_points_table_1lhaaso,
    get_flux_points_1lhaaso,
)


@pytest.fixture
def fake_1lhaaso_source():
    source = MagicMock()
    source.name = "1LHAASO J0000+0000"
    source.data = {
        "Model_a": "point",
        "Model_b": "extended",
        "E0": 10.0 * u.TeV,
        "E0_b": 20.0 * u.TeV,
    }

    spectral_model = MagicMock()
    spectral_model.return_value = u.Quantity(
        [1.0e-14],
        "TeV-1 cm-2 s-1",
    )
    spectral_model.evaluate_error.return_value = (
        u.Quantity([1.0e-14], "TeV-1 cm-2 s-1"),
        u.Quantity([1.0e-15], "TeV-1 cm-2 s-1"),
    )
    source.spectral_model.return_value = spectral_model

    return source


def _make_catalog_file(tmp_path, name="LHAASO J0000+0000"):
    table = Table()
    table["source_name"] = [name]
    table["ra"] = [10.0] * u.deg
    table["dec"] = [20.0] * u.deg

    filename = tmp_path / "lhaaso.ecsv"
    table.write(filename, format="ascii.ecsv")
    return filename


def test_create_flux_points_table_1lhaaso(fake_1lhaaso_source):
    table = create_flux_points_table_1lhaaso(
        fake_1lhaaso_source,
        which="point",
    )

    assert table.meta["SED_TYPE"] == "dnde"
    assert table.meta["model"] == "point"
    assert table.meta["source_name"] == fake_1lhaaso_source.name
    assert len(table) == 1
    assert np.isclose(
        table["e_ref"].quantity[0].to_value("TeV"),
        10.0,
    )
    assert not table["is_ul"][0]


def test_create_flux_points_table_1lhaaso_extended(fake_1lhaaso_source):
    table = create_flux_points_table_1lhaaso(
        fake_1lhaaso_source,
        which="extended",
    )

    assert np.isclose(
        table["e_ref"].quantity[0].to_value("TeV"),
        20.0,
    )


def test_create_flux_points_table_1lhaaso_invalid(fake_1lhaaso_source):
    with pytest.raises(ValueError, match="Invalid model component name"):
        create_flux_points_table_1lhaaso(
            fake_1lhaaso_source,
            which="invalid",
        )


def test_get_flux_points_1lhaaso(fake_1lhaaso_source):
    table = Table()
    table.meta["SED_TYPE"] = "dnde"
    expected = MagicMock()
    spectral_model = fake_1lhaaso_source.spectral_model.return_value

    with (
        patch(
            "feupy.catalogs.lhaaso.create_flux_points_table_1lhaaso",
            return_value=table,
        ),
        patch(
            "feupy.catalogs.lhaaso.FluxPoints.from_table",
            return_value=expected,
        ) as from_table,
    ):
        result = get_flux_points_1lhaaso(
            fake_1lhaaso_source,
            "point",
        )

    fake_1lhaaso_source.spectral_model.assert_called_once_with("point")
    from_table.assert_called_once_with(
        table=table,
        reference_model=spectral_model,
        sed_type="dnde",
    )
    assert result is expected


def test_source_catalog_lhaaso(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogLHAASO(filename=filename)

    assert len(catalog.table) == 1
    assert catalog.tag == "LHAASO"
    assert catalog.source_object_class is SourceCatalogObjectLHAASO


def test_source_catalog_extra_lhaaso(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraLHAASO(filename=filename)

    assert len(catalog.table) == 1
    assert catalog.tag == "LHAASO-2024icrc"
    assert catalog.source_object_class is SourceCatalogObjectExtraLHAASO


def test_extra_lhaaso_spectral_model_found(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraLHAASO(filename=filename)
    source = catalog["LHAASO J0000+0000"]

    spectral_model = PowerLawSpectralModel()
    sky_model = MagicMock()
    sky_model.spectral_model = spectral_model

    models = MagicMock()
    models.names = ["LHAASO J0000+0000"]
    models.__getitem__.return_value = sky_model

    with patch.object(
        SourceCatalogObjectExtraLHAASO,
        "_MODELS",
        models,
    ):
        result = source.spectral_model()

    assert result is spectral_model


def test_extra_lhaaso_sky_model(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraLHAASO(filename=filename)
    source = catalog["LHAASO J0000+0000"]

    spectral_model = PowerLawSpectralModel()

    with patch.object(
        SourceCatalogObjectExtraLHAASO,
        "spectral_model",
        return_value=spectral_model,
    ):
        model = source.sky_model()

    assert model.name == "LHAASO J0000+0000"
    assert model.spectral_model is spectral_model


def test_lhaaso_info_spectrum_without_model(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogLHAASO(filename=filename)
    source = catalog["LHAASO J0000+0000"]

    with patch.object(
        SourceCatalogObjectLHAASO,
        "spectral_model",
        return_value=None,
    ):
        info = source._info_spectrum()

    assert "No spectrum available" in info


def test_lhaaso_spectral_model_unknown_type(tmp_path):
    table = Table()
    table["source_name"] = ["LHAASO J0000+0000"]
    table["ra"] = [10.0] * u.deg
    table["dec"] = [20.0] * u.deg
    table["spec_reference"] = [10.0] * u.TeV
    table["spec_type"] = ["unknown"]

    filename = tmp_path / "lhaaso_unknown.ecsv"
    table.write(filename, format="ascii.ecsv")

    catalog = SourceCatalogLHAASO(filename=filename)
    source = catalog["LHAASO J0000+0000"]

    with patch.object(
        SourceCatalogObjectLHAASO,
        "flux_points_table",
        new_callable=MagicMock,
    ):
        result = source.spectral_model()

    assert result is None
