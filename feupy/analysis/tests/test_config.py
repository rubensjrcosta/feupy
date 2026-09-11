# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.analysis.config import CTAOAnalysisConfig


def test_ctao_config_init():
    config = CTAOAnalysisConfig()
    assert config is not None
    assert hasattr(config, "observation")
    assert hasattr(config, "datasets")
