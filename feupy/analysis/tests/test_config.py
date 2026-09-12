# Licensed under a 3-clause BSD style license - see LICENSE.rst

from gammapy.analysis.config import (
    FitConfig,
    FluxPointsConfig,
    GeneralConfig,
)

from feupy.analysis.config import CTAOAnalysisConfig
from feupy.utils.config import (
    DatasetsConfig,
    ObservationConfig,
    SensitivityConfig,
    StatisticsConfig,
)


def test_ctao_config_init():
    config = CTAOAnalysisConfig()

    assert isinstance(config.general, GeneralConfig)
    assert isinstance(config.observation, ObservationConfig)
    assert isinstance(config.datasets, DatasetsConfig)
    assert isinstance(config.statistics, StatisticsConfig)
    assert isinstance(config.fit, FitConfig)
    assert isinstance(config.flux_points, FluxPointsConfig)
    assert isinstance(config.sensitivity, SensitivityConfig)


def test_ctao_config_to_yaml():
    config = CTAOAnalysisConfig()

    yaml_str = config.to_yaml()

    assert isinstance(yaml_str, str)
    assert "general:" in yaml_str
    assert "observation:" in yaml_str
    assert "datasets:" in yaml_str
    assert "statistics:" in yaml_str
    assert "fit:" in yaml_str
    assert "flux_points:" in yaml_str
    assert "sensitivity:" in yaml_str


def test_ctao_config_from_yaml():
    config = CTAOAnalysisConfig()
    yaml_str = config.to_yaml()

    result = CTAOAnalysisConfig.from_yaml(yaml_str)

    assert isinstance(result, CTAOAnalysisConfig)
    assert result.model_dump() == config.model_dump()


def test_ctao_config_str():
    config = CTAOAnalysisConfig()

    result = str(config)

    assert result.startswith("CTAOAnalysisConfig")
    assert "general:" in result
    assert "observation:" in result
    assert "datasets:" in result
