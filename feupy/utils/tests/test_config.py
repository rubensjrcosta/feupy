# Licensed under a 3-clause BSD style license - see LICENSE.rst

from feupy.utils.config import (
    ObservationConfig, DatasetsConfig, OnOffConfig, SensitivityConfig, StatisticsConfig
)


def test_configs_instantiation():

    observation = ObservationConfig()
    datasets = DatasetsConfig()
    onoff = OnOffConfig()
    sensitivity = SensitivityConfig()
    statistics = StatisticsConfig()

    assert observation is not None
    assert datasets is not None
    assert onoff is not None
    assert sensitivity is not None
    assert statistics is not None

def test_onoff_defaults():

    config = OnOffConfig()

    assert config.acceptance == 1
    assert config.acceptance_off == 5

def test_sensitivity_defaults():

    config = SensitivityConfig()

    assert config.gamma_min == 10
    assert config.n_sigma == 5
    assert config.bkg_syst_fraction == 0.05
    assert config.table_format == "fits"

def test_datasets_structure():

    config = DatasetsConfig()

    assert config.geom is not None
    assert config.background is not None
    assert config.safe_mask is not None
    assert config.on_region is not None
    assert config.on_off is not None

def test_observation_defaults():

    config = ObservationConfig()

    assert config.livetime is None
    assert config.offset is None
    assert config.position_angle is None
    assert config.required_irfs == ["South", "AverageAz", "20deg", "50h"]

