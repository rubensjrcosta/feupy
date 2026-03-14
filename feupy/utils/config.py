# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for analysis configuration."""

from typing import List, Optional

from gammapy.analysis.config import (
    GammapyBaseConfig,
    SpatialCircleConfig,
    GeomConfig,
    BackgroundConfig,
    SafeMaskConfig,
    MapSelectionEnum,
    ReductionTypeEnum,
)

from gammapy.makers import MapDatasetMaker
from gammapy.utils.types import QuantityType, AngleType, PathType

from feupy.utils.types import IrfType
from feupy.utils.enum import TableEnum


__all__ = [
    "ObservationConfig",
    "DatasetsConfig",
    "OnOffConfig",
    "SensitivityConfig",
    "StatisticsConfig",
]


# =========================================================
# ON / OFF configuration
# =========================================================

class OnOffConfig(GammapyBaseConfig):
    """Configuration for On-Off analysis."""
    
    acceptance: int = 1
    acceptance_off: int = 5


# =========================================================
# Dataset configuration
# =========================================================

class DatasetsConfig(GammapyBaseConfig):
    """Dataset reduction configuration."""

    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    stack: bool = True

    geom: GeomConfig = GeomConfig()

    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection

    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()

    on_region: SpatialCircleConfig = SpatialCircleConfig()

    containment_correction: bool = True
    containment: float = 0.68
    use_region_center: bool = False

    on_off: OnOffConfig = OnOffConfig()


# =========================================================
# Statistics configuration
# =========================================================

class StatisticsConfig(GammapyBaseConfig):
    """Statistical configuration for sensitivity calculations."""
    
    n_obs: int = 1


# =========================================================
# Sensitivity configuration
# =========================================================

class SensitivityConfig(GammapyBaseConfig):
    """Sensitivity calculation configuration."""

    gamma_min: int = 10
    n_sigma: int = 5

    bkg_syst_fraction: float = 0.05

    data_path: Optional[PathType] = None
    table_format: TableEnum = "fits"


# =========================================================
# Observation configuration
# =========================================================

class ObservationConfig(GammapyBaseConfig):
    """Observation setup configuration."""

    obs_cone: SpatialCircleConfig = SpatialCircleConfig()

    livetime: Optional[QuantityType] = None
    offset: Optional[QuantityType] = None
    position_angle: Optional[AngleType] = None

    required_irfs: IrfType = ["South", "AverageAz", "20deg", "50h"]