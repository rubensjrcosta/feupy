# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.analysis.config import CTAOAnalysisConfig
from feupy.analysis.core import CTAOAnalysis

def test_analysis_basic_init():
    config = CTAOAnalysisConfig()
    analysis = CTAOAnalysis(config)
    assert analysis is not None

