# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Analysis."""

from .config import CounterpartsAnalysisConfig, CTAObservationAnalysisConfig, SimulationConfig
from .core import CounterpartsAnalysis, CTAObservationAnalysis, Simulation

from .simulation.observations import ObservationParameters

from .simulation.datasets import create_spectrum_dataset_empty, create_spectrum_dataset_onoff
from .simulation.stats import StatisticalUtilityFunctions
from .simulation.geometry import GeometryParameters

# from .config import CTAObservationAnalysisConfig, CounterpartsAnalysisConfig 
# from .core import CTAObservationAnalysis, CounterpartsAnalysis

__all__ = [
    'Simulation',
    'SimulationConfig',
    "CounterpartsAnalysis",
    "CounterpartsAnalysisConfig",
    "CTAObservationAnalysis",
    "CTAObservationAnalysisConfig",
    "ObservationParameters",
    "GeometryParameters",
    "create_spectrum_dataset_empty",
    "create_spectrum_dataset_onoff",
    "StatisticalUtilityFunctions"
]
