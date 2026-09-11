# Licensed under a 3-clause BSD style license - see LICENSE.rst
from feupy.analysis.config import CTAOAnalysisConfig
from feupy.analysis.core import CTAOAnalysis


def create_analysis():
    """Helper function to create a basic CTAOAnalysis instance."""
    config = CTAOAnalysisConfig()
    return CTAOAnalysis(config)


def test_analysis_basic_init():
    """Test that CTAOAnalysis initializes correctly."""
    analysis = create_analysis()
    assert analysis is not None
