# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
from astropy.table import Table
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

from feupy.catalogs.veritas import (
    SourceCatalogObjectVERITASCygnus,
    SourceCatalogObjectVTSCat,
    SourceCatalogVERITASCygnus,
    SourceCatalogVTSCat,
    generate_unique_name,
)


def _make_vtscat_file(tmp_path):
    table = Table()
    table["source_name"] = ["VER J0000+000"]
    table["veritas_name"] = ["VER J0000+000"]
    table["common_name"] = ["Test source"]
    table["other_names"] = ["None"]
    table["veritas_id"] = [1]
    table["where"] = ["Galactic"]
    table["type"] = ["PWN"]
    table["veritas_components"] = ["A"]
    table["simbad_id"] = ["TEST"]
    table["reference_id"] = ["ref001, ref002"]
    table["ra"] = [10.0] * u.deg
    table["dec"] = [20.0] * u.deg

    filename = tmp_path / "vtscat.ecsv"
    table.write(filename, format="ascii.ecsv")
    return filename


def _make_veritas_file(tmp_path):
    table = Table()
    table["source_name"] = ["VER J0000+000"]
    table["ra"] = [10.0] * u.deg
    table["dec"] = [20.0] * u.deg

    filename = tmp_path / "veritas.fits"
    table.write(filename, format="fits")
    return filename


def test_generate_unique_name():
    assert generate_unique_name("Source", "reference123", []) == ("Source (referen)")


def test_generate_unique_name_duplicate():
    names = ["Source (referen)"]

    result = generate_unique_name(
        "Source",
        "reference123",
        names,
    )

    assert result == "Source (referen-a)"


def test_source_catalog_vtscat(tmp_path):
    catalog = SourceCatalogVTSCat(
        filename=_make_vtscat_file(tmp_path),
    )

    assert len(catalog.table) == 1
    assert catalog.tag == "vtscat"
    assert catalog.source_object_class is SourceCatalogObjectVTSCat


def test_vtscat_source_info(tmp_path):
    catalog = SourceCatalogVTSCat(
        filename=_make_vtscat_file(tmp_path),
    )
    source = catalog["VER J0000+000"]

    info = source.info("basic,position")

    assert "*** Basic info ***" in info
    assert "*** Position info ***" in info
    assert "VER-000001" in info


def test_vtscat_reference_id(tmp_path):
    catalog = SourceCatalogVTSCat(
        filename=_make_vtscat_file(tmp_path),
    )
    source = catalog["VER J0000+000"]

    assert source._reference_id() == ["ref001", "ref002"]


def test_vtscat_get_flux_points_tables(tmp_path):
    catalog = SourceCatalogVTSCat(
        filename=_make_vtscat_file(tmp_path),
    )
    source = catalog["VER J0000+000"]

    sed = Table()
    sed["e_ref"] = [1.0, 2.0] * u.TeV
    sed["dnde"] = [1.0e-12, 2.0e-12] * u.Unit("TeV-1 cm-2 s-1")
    sed["dnde_ul"] = [
        np.nan,
        3.0e-12,
    ] * u.Unit("TeV-1 cm-2 s-1")

    filename = tmp_path / "sed.ecsv"
    sed.write(filename, format="ascii.ecsv")

    with patch.object(
        SourceCatalogObjectVTSCat,
        "_get_file_paths",
        return_value=[filename],
    ):
        tables = source.get_flux_points_tables("ref001")

    assert len(tables) == 1
    assert tables[0].meta["SED_TYPE"] == "dnde"
    assert list(tables[0]["is_ul"]) == [False, True]


def test_source_catalog_veritas_cygnus(tmp_path):
    catalog = SourceCatalogVERITASCygnus(
        filename=_make_veritas_file(tmp_path),
    )

    assert len(catalog.table) == 1
    assert catalog.tag == "veritas-2018ApJ"
    assert catalog.source_object_class is SourceCatalogObjectVERITASCygnus


def test_veritas_cygnus_spectral_model(tmp_path):
    catalog = SourceCatalogVERITASCygnus(
        filename=_make_veritas_file(tmp_path),
    )
    source = catalog["VER J0000+000"]

    spectral_model = PowerLawSpectralModel()
    sky_model = SkyModel(
        spectral_model=spectral_model,
        name="VER J0000+000",
    )
    models = MagicMock()
    models.__getitem__.return_value = sky_model

    with patch.object(
        SourceCatalogObjectVERITASCygnus,
        "_MODELS",
        models,
    ):
        result = source.spectral_model()

    assert result is spectral_model


def test_veritas_cygnus_sky_model(tmp_path):
    catalog = SourceCatalogVERITASCygnus(
        filename=_make_veritas_file(tmp_path),
    )
    source = catalog["VER J0000+000"]

    expected = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        name="VER J0000+000",
    )
    models = MagicMock()
    models.__getitem__.return_value = expected

    with patch.object(
        SourceCatalogObjectVERITASCygnus,
        "_MODELS",
        models,
    ):
        result = source.sky_model()

    assert result is expected


def test_veritas_cygnus_info_without_model(tmp_path):
    catalog = SourceCatalogVERITASCygnus(
        filename=_make_veritas_file(tmp_path),
    )
    source = catalog["VER J0000+000"]

    with patch.object(
        SourceCatalogObjectVERITASCygnus,
        "spectral_model",
        return_value=None,
    ):
        info = source._info_spectrum()

    assert "No spectrum available" in info


def test_veritas_cygnus_flux_points(tmp_path):
    catalog = SourceCatalogVERITASCygnus(
        filename=_make_veritas_file(tmp_path),
    )
    source = catalog["VER J0000+000"]
    expected = MagicMock()

    with (
        patch(
            "feupy.catalogs.veritas.make_path",
            return_value=tmp_path / "flux.fits",
        ),
        patch(
            "feupy.catalogs.veritas.FluxPoints.read",
            return_value=expected,
        ) as read,
    ):
        result = source.flux_points

    read.assert_called_once_with(tmp_path / "flux.fits")
    assert result is expected
