# Licensed under a 3-clause BSD style license - see LICENSE

from gammapy.analysis.config import ReductionTypeEnum
from gammapy.makers import MapDatasetMaker

from feupy.utils.config import (
    DatasetsConfig,
    ObservationConfig,
    OnOffConfig,
    SensitivityConfig,
    StatisticsConfig,
)


def test_onoff_defaults():
    config = OnOffConfig()

    assert config.acceptance == 1
    assert config.acceptance_off == 5


def test_datasets_defaults():
    config = DatasetsConfig()

    assert config.type == ReductionTypeEnum.spectrum
    assert config.stack is True
    assert config.map_selection == MapDatasetMaker.available_selection
    assert config.containment_correction is True
    assert config.containment == 0.68
    assert config.use_region_center is False


def test_datasets_nested_configs():
    config = DatasetsConfig()

    assert config.geom is not None
    assert config.background is not None
    assert config.safe_mask is not None
    assert config.on_region is not None
    assert config.on_off is not None


def test_statistics_defaults():
    config = StatisticsConfig()

    assert config.n_obs == 1


def test_sensitivity_defaults():
    config = SensitivityConfig()

    assert config.gamma_min == 10
    assert config.n_sigma == 5
    assert config.bkg_syst_fraction == 0.05
    assert config.data_path is None
    assert config.table_format == "fits"


def test_observation_defaults():
    config = ObservationConfig()

    assert config.livetime is None
    assert config.offset is None
    assert config.position_angle is None
    assert tuple(config.required_irfs) == (
        "South",
        "AverageAz",
        "20deg",
        "50h",
    )


def test_observation_has_cone():
    config = ObservationConfig()

    assert config.obs_cone is not None
