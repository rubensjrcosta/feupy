# Licensed under a 3-clause BSD style license - see LICENSE
"""Configuration classes used by FeuPy analysis utilities."""

from gammapy.analysis.config import (
    BackgroundConfig,
    GammapyBaseConfig,
    GeomConfig,
    MapSelectionEnum,
    ReductionTypeEnum,
    SafeMaskConfig,
    SpatialCircleConfig,
)
from gammapy.makers import MapDatasetMaker
from gammapy.utils.types import AngleType, PathType, QuantityType

from feupy.utils.enum import TableEnum
from feupy.utils.types import IrfType

__all__ = [
    "ObservationConfig",
    "DatasetsConfig",
    "OnOffConfig",
    "SensitivityConfig",
    "StatisticsConfig",
]


class OnOffConfig(GammapyBaseConfig):
    """Configuration for ON/OFF spectral analysis."""

    acceptance: int = 1
    acceptance_off: int = 5


class DatasetsConfig(GammapyBaseConfig):
    """Configuration for dataset reduction."""

    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    stack: bool = True
    geom: GeomConfig = GeomConfig()
    map_selection: list[MapSelectionEnum] = MapDatasetMaker.available_selection
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True
    containment: float = 0.68
    use_region_center: bool = False
    on_off: OnOffConfig = OnOffConfig()


class StatisticsConfig(GammapyBaseConfig):
    """Configuration for statistical calculations."""

    n_obs: int = 1


class SensitivityConfig(GammapyBaseConfig):
    """Configuration for sensitivity calculations."""

    gamma_min: int = 10
    n_sigma: int = 5
    bkg_syst_fraction: float = 0.05
    data_path: PathType | None = None
    table_format: TableEnum = "fits"


class ObservationConfig(GammapyBaseConfig):
    """Configuration for CTAO observations."""

    obs_cone: SpatialCircleConfig = SpatialCircleConfig()
    livetime: QuantityType | None = None
    offset: QuantityType | None = None
    position_angle: AngleType | None = None
    required_irfs: IrfType = ("South", "AverageAz", "20deg", "50h")
