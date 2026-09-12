# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

import astropy.units as u
from astropy.table import Table
from gammapy.modeling.models import PowerLawSpectralModel

from feupy.catalogs.hess import (
    SourceCatalogExtraHESS,
    SourceCatalogObjectExtraHESS,
)


def _make_catalog_file(tmp_path):
    table = Table()
    table["source_name"] = ["HESS J1825-137"]
    table["ra"] = [276.4] * u.deg
    table["dec"] = [-13.8] * u.deg

    filename = tmp_path / "hess.ecsv"
    table.write(filename, format="ascii.ecsv")
    return filename


def test_catalog_initialization(tmp_path):
    filename = _make_catalog_file(tmp_path)

    catalog = SourceCatalogExtraHESS(filename=filename)

    assert len(catalog.table) == 1
    assert catalog.tag == "hess-2019A&A"
    assert catalog.source_object_class is SourceCatalogObjectExtraHESS


def test_source_info(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    with patch.object(
        SourceCatalogObjectExtraHESS,
        "spectral_model",
        return_value=None,
    ):
        info = source.info()

    assert "*** Basic info ***" in info
    assert "*** Position info ***" in info
    assert "No spectral information available." in info


def test_source_info_selected_sections(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    info = source.info("basic,position")

    assert "*** Basic info ***" in info
    assert "*** Position info ***" in info
    assert "*** Spectral info ***" not in info


def test_spectral_model_found(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    spectral_model = PowerLawSpectralModel()
    sky_model = MagicMock()
    sky_model.spectral_model = spectral_model

    models = MagicMock()
    models.names = ["HESS J1825-137"]
    models.__getitem__.return_value = sky_model

    with patch.object(
        SourceCatalogObjectExtraHESS,
        "_MODELS",
        models,
    ):
        result = source.spectral_model()

    assert result is spectral_model
    models.__getitem__.assert_called_once_with("HESS J1825-137")


def test_spectral_model_missing(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    models = MagicMock()
    models.names = []

    with patch.object(
        SourceCatalogObjectExtraHESS,
        "_MODELS",
        models,
    ):
        result = source.spectral_model()

    assert result is None


def test_sky_model(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    spectral_model = PowerLawSpectralModel()

    with patch.object(
        SourceCatalogObjectExtraHESS,
        "spectral_model",
        return_value=spectral_model,
    ):
        model = source.sky_model()

    assert model.name == "HESS J1825-137"
    assert model.spectral_model is spectral_model


def test_flux_points_missing_file(tmp_path, monkeypatch):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    monkeypatch.setenv("FEUPY_DATA", str(tmp_path))

    assert source.flux_points is None


def test_flux_points_read(tmp_path):
    filename = _make_catalog_file(tmp_path)
    catalog = SourceCatalogExtraHESS(filename=filename)
    source = catalog["HESS J1825-137"]

    expected = MagicMock()
    sky_model = MagicMock()
    fake_path = MagicMock()
    fake_path.exists.return_value = True

    with (
        patch(
            "feupy.catalogs.hess.make_path",
            return_value=fake_path,
        ),
        patch.object(
            SourceCatalogObjectExtraHESS,
            "sky_model",
            return_value=sky_model,
        ),
        patch(
            "feupy.catalogs.hess.FluxPoints.read",
            return_value=expected,
        ) as read,
    ):
        result = source.flux_points

    read.assert_called_once_with(
        fake_path,
        reference_model=sky_model,
        sed_type="e2dnde",
    )
    assert result is expected
