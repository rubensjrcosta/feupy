# Licensed under a 3-clause BSD style license - see LICENSE.rst

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from gammapy.maps import MapAxis
from gammapy.modeling.models import Models

from feupy.analysis.core import CTAOAnalysis, ROIAnalysis


def make_roi_analysis():
    config = MagicMock()
    config.set_logging = MagicMock()

    with patch(
        "feupy.analysis.core.ROIAnalysisConfig",
        return_value=config,
    ):
        analysis = ROIAnalysis({})

    return analysis, config


def make_ctao_analysis():
    config = MagicMock()
    config.set_logging = MagicMock()

    with (
        patch(
            "feupy.analysis.core.CTAOAnalysisConfig",
            return_value=config,
        ),
        patch("feupy.analysis.core.Observations"),
        patch("feupy.analysis.core.Fit"),
        patch("feupy.analysis.core.CTAOIRFManager"),
    ):
        analysis = CTAOAnalysis({})

    return analysis, config


def test_roi_analysis_config_from_dict():
    config = MagicMock()
    config.set_logging = MagicMock()

    with patch(
        "feupy.analysis.core.ROIAnalysisConfig",
        return_value=config,
    ) as config_class:
        analysis = ROIAnalysis({"general": {}})

    config_class.assert_called_once_with(general={})
    config.set_logging.assert_called_once_with()
    assert analysis.config is config
    assert analysis.datasets is None
    assert analysis.sources is None
    assert analysis.catalog is None


def test_roi_analysis_invalid_config():
    analysis = ROIAnalysis.__new__(ROIAnalysis)

    with pytest.raises(
        TypeError,
        match="config must be dict or ROIAnalysisConfig",
    ):
        analysis.config = object()


def test_roi_models_without_datasets():
    analysis, _ = make_roi_analysis()
    analysis.datasets = None

    with pytest.raises(
        RuntimeError,
        match="No datasets defined",
    ):
        _ = analysis.models


def test_roi_models_setter_calls_set_models():
    analysis, _ = make_roi_analysis()
    analysis.set_models = MagicMock()

    models = MagicMock()
    analysis.models = models

    analysis.set_models.assert_called_once_with(
        models,
        extend=False,
    )


@pytest.mark.parametrize(
    ("which", "expected_call"),
    [
        (None, {}),
        ("point", {"which": "point"}),
    ],
)
def test_create_spectral_model(which, expected_call):
    source = MagicMock()
    expected = MagicMock()
    source.spectral_model.return_value = expected

    result = ROIAnalysis._create_spectral_model(
        source,
        which=which,
    )

    assert result is expected
    source.spectral_model.assert_called_once_with(**expected_call)


def test_roi_run_calls_pipeline():
    analysis, _ = make_roi_analysis()

    analysis._get_sources = MagicMock()
    analysis._get_flux_points_datasets = MagicMock()
    analysis._get_catalog_roi = MagicMock()

    analysis.run()

    analysis._get_sources.assert_called_once_with()
    analysis._get_flux_points_datasets.assert_called_once_with()
    analysis._get_catalog_roi.assert_called_once_with()


def test_roi_set_models_missing_datasets():
    analysis, _ = make_roi_analysis()
    analysis.datasets = None

    with pytest.raises(
        RuntimeError,
        match="Missing datasets",
    ):
        analysis.set_models(Models())


def test_roi_set_models_invalid_type():
    analysis, _ = make_roi_analysis()

    datasets = MagicMock()
    datasets.__len__.return_value = 1
    analysis.datasets = datasets

    with pytest.raises(TypeError, match="Invalid type"):
        analysis.set_models(object())


def test_ctao_analysis_config_from_dict():
    config = MagicMock()
    config.set_logging = MagicMock()

    with (
        patch(
            "feupy.analysis.core.CTAOAnalysisConfig",
            return_value=config,
        ) as config_class,
        patch("feupy.analysis.core.Observations") as observations_class,
        patch("feupy.analysis.core.Fit") as fit_class,
        patch("feupy.analysis.core.CTAOIRFManager") as irf_manager_class,
    ):
        analysis = CTAOAnalysis({"general": {}})

    config_class.assert_called_once_with(general={})
    config.set_logging.assert_called_once_with()
    observations_class.assert_called_once_with()
    fit_class.assert_called_once_with()
    irf_manager_class.assert_called_once_with()

    assert analysis.datasets is None
    assert analysis.spectrum_dataset is None
    assert analysis.fit_result is None
    assert analysis.flux_points is None
    assert analysis.table_sens is None


def test_ctao_invalid_config():
    analysis = CTAOAnalysis.__new__(CTAOAnalysis)

    with pytest.raises(
        TypeError,
        match="config must be dict or CTAOAnalysisConfig",
    ):
        analysis.config = object()


def test_create_pointing_position():
    position = SkyCoord(
        ra=10 * u.deg,
        dec=20 * u.deg,
        frame="icrs",
    )

    result = CTAOAnalysis._create_pointing_position(
        position,
        90 * u.deg,
        1 * u.deg,
    )

    assert isinstance(result, SkyCoord)
    assert position.separation(result).to_value(u.deg) == pytest.approx(1)


def test_create_pointing():
    pointing_position = SkyCoord(
        ra=10 * u.deg,
        dec=20 * u.deg,
        frame="icrs",
    )

    with patch("feupy.analysis.core.FixedPointingInfo") as fixed_pointing_info:
        result = CTAOAnalysis._create_pointing(pointing_position)

    assert result is fixed_pointing_info.return_value
    fixed_pointing_info.assert_called_once()

    kwargs = fixed_pointing_info.call_args.kwargs
    assert kwargs["fixed_icrs"].separation(pointing_position.icrs).to_value(
        u.deg
    ) == pytest.approx(0)


def test_get_spectrum_dataset_requires_observations():
    analysis, config = make_ctao_analysis()
    analysis.observations = []

    with pytest.raises(
        RuntimeError,
        match="No observations defined",
    ):
        analysis.get_spectrum_dataset()


def test_get_spectrum_dataset_requires_1d():
    analysis, config = make_ctao_analysis()
    analysis.observations = [MagicMock()]
    config.datasets.type = "3d"

    with pytest.raises(
        ValueError,
        match="Only 1D ON/OFF supported",
    ):
        analysis.get_spectrum_dataset()


def test_get_spectrum_dataset_runs_extraction():
    analysis, config = make_ctao_analysis()
    analysis.observations = [MagicMock()]
    config.datasets.type = "1d"
    analysis._spectrum_extraction = MagicMock()

    model = MagicMock()

    analysis.get_spectrum_dataset(
        model=model,
        obs_id=2,
        random_state=7,
    )

    analysis._spectrum_extraction.assert_called_once_with(
        model,
        2,
        7,
    )


def test_get_datasets_requires_spectrum_dataset():
    analysis, _ = make_ctao_analysis()
    analysis.spectrum_dataset = None

    with pytest.raises(
        RuntimeError,
        match="No spectrum dataset",
    ):
        analysis.get_datasets()


def test_get_datasets_runs_on_off():
    analysis, _ = make_ctao_analysis()
    analysis.spectrum_dataset = MagicMock()
    analysis._run_on_off = MagicMock()

    analysis.get_datasets()

    analysis._run_on_off.assert_called_once_with()


@pytest.mark.parametrize(
    ("method", "class_name"),
    [
        ("reflected", "ReflectedRegionsBackgroundMaker"),
        ("ring", "RingBackgroundMaker"),
        ("fov_background", "FoVBackgroundMaker"),
    ],
)
def test_create_background_maker(method, class_name):
    analysis, config = make_ctao_analysis()

    config.datasets.background.method = method
    config.datasets.background.parameters = {"test": 1}

    target = f"feupy.analysis.core.{class_name}"

    with patch(target) as maker_class:
        result = analysis._create_background_maker()

    assert result is maker_class.return_value
    maker_class.assert_called_once_with(test=1)


def test_create_background_maker_unknown_method():
    analysis, config = make_ctao_analysis()

    config.datasets.background.method = "unknown"
    config.datasets.background.parameters = {}

    assert analysis._create_background_maker() is None


def test_make_energy_axis():
    axis_config = SimpleNamespace(
        min=0.1 * u.TeV,
        max=10 * u.TeV,
        nbins=4,
    )

    axis = CTAOAnalysis._make_energy_axis(
        axis_config,
        name="energy",
    )

    assert isinstance(axis, MapAxis)
    assert axis.name == "energy"
    assert axis.nbin == 4
    assert axis.unit == u.TeV
    assert axis.edges[0].to_value(u.TeV) == pytest.approx(0.1)
    assert axis.edges[-1].to_value(u.TeV) == pytest.approx(10)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (None, 10 * u.TeV),
        (0.1 * u.TeV, None),
    ],
)
def test_make_energy_axis_missing_bounds(minimum, maximum):
    axis_config = SimpleNamespace(
        min=minimum,
        max=maximum,
        nbins=4,
    )

    assert CTAOAnalysis._make_energy_axis(axis_config) is None


def test_run_fit_requires_datasets():
    analysis, _ = make_ctao_analysis()
    analysis.datasets = None

    with pytest.raises(
        RuntimeError,
        match="No datasets",
    ):
        analysis.run_fit()


def test_run_fit():
    analysis, _ = make_ctao_analysis()

    datasets = MagicMock()
    datasets.__bool__.return_value = True
    analysis.datasets = datasets

    fit = MagicMock()
    fit.run.return_value = "fit-result"
    analysis.fit = fit

    analysis.run_fit()

    fit.run.assert_called_once_with(
        datasets=datasets,
    )
    assert analysis.fit_result == "fit-result"


def test_write_table_sensitivity_requires_table():
    analysis, _ = make_ctao_analysis()
    analysis.table_sens = None

    with pytest.raises(
        RuntimeError,
        match="Missing table_sens",
    ):
        analysis.write_table_sensitivity()


def test_get_table_meta():
    analysis, config = make_ctao_analysis()

    config.datasets.on_region.radius = 0.2 * u.deg
    config.observation.offset = 0.5 * u.deg
    config.observation.livetime = 50 * u.h
    config.observation.required_irfs = (
        "South",
        "AverageAz",
        "20deg",
        "50h",
    )

    analysis.irf_manager = MagicMock()
    analysis.irf_manager.get_irf.return_value = {
        "name": "test-irf",
        "label": "Test IRF",
    }

    meta = analysis._get_table_meta()

    assert meta["ONRADIUS"] == "0.2 deg"
    assert meta["IRF_NAME"] == "test-irf"
    assert meta["IRF_LABEL"] == "Test IRF"
    assert meta["IRF_ARR"] == "South"
    assert meta["IRF_AZ"] == "AverageAz"
    assert meta["IRF_ZEN"] == "20deg"
    assert meta["IRF_LT"] == "50h"


def test_get_file_name():
    analysis, config = make_ctao_analysis()

    config.observation.required_irfs = (
        "South",
        "AverageAz",
        "20deg",
        "50h",
    )
    config.observation.livetime = 50 * u.h

    analysis.irf_manager = MagicMock()
    analysis.irf_manager.get_irf.return_value = {"name": "prod5-test"}

    result = analysis.get_file_name()

    assert result == "sens_prod5-test_livetime50.0h"
