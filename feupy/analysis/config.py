#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pyflakes config.py


# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Configuration."""


# In[2]:


from pydantic import BaseModel
# from pydantic.v1 import BaseModel
# from pydantic.v1.utils import lenient_isinstance
from pydantic.utils import deep_update

from feupy.roi import ROI
from feupy.target import Target

# from feupy.analysis.simulation.observations import ObservationParameters
from feupy.analysis.simulation import ObservationParameters

from feupy.cta.irfs import Irfs
from feupy.analysis.simulation import GeometryParameters

from feupy.utils.observation import create_observation
from feupy.utils.geometry import (
    create_energy_axis, 
    define_on_region, 
    create_region_geometry
)
from feupy.plotters import *

from feupy.utils.types import (
    AngleType,
    EnergyType,
    QuantityType,
    TimeType,
    IrfType,
)

from feupy.utils.enum import(
    CatalogsTypeEnum,
    ReductionTypeEnum,
    FrameEnum,
    RequiredHDUEnum,
    BackgroundMethodEnum,
    SafeMaskMethodsEnum,
    MapSelectionEnum,
)


from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity

from gammapy.utils.units import energy_unit_format
from gammapy.utils.scripts import make_path, read_yaml
from gammapy.makers import MapDatasetMaker

import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
import yaml


# In[ ]:





# In[ ]:





# In[ ]:


# __file__ = 'file'
# __name__ = 'name'


from feupy.utils.scripts import pickling, unpickling
from feupy.cta.irfs import Irfs
from feupy.catalog.pulsar.atnf import SourceCatalogATNF

from feupy.target import Target

from feupy.utils.coordinates import skcoord_to_dict, dict_to_skcoord

from feupy.analysis.simulation import ObservationParameters

from feupy.plotters import *

from pathlib import Path

from astropy import units as u
from astropy.table import Table

from gammapy.modeling.models import SkyModel

from gammapy.modeling import Fit

import matplotlib.pyplot as plt
import numpy as np


# In[ ]:





# In[ ]:





# In[ ]:


__all__ = ["CounterpartsConfig", 'SimulationConfig']


# In[ ]:


CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


# In[ ]:





# In[ ]:


class GammapyBaseConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        arbitrary_types_allowed=True
        from_attributes=True
        extra = "forbid"
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
        }

        
class SkyCoordConfig(GammapyBaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None

        
class EnergyAxisConfig(GammapyBaseConfig):
    min: EnergyType = None
    max: EnergyType = None
    nbins: int = None
    name: str = "energy"


class SpatialCircleConfig(GammapyBaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None
    radius: AngleType = None


class EnergyRangeConfig(GammapyBaseConfig):
    min: EnergyType = None
    max: EnergyType = None


class TimeRangeConfig(GammapyBaseConfig):
    start: TimeType = None
    stop: TimeType = None


class FluxPointsConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(GammapyBaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class FitConfig(GammapyBaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class ExcessMapConfig(GammapyBaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


class BackgroundConfig(GammapyBaseConfig):
    method: BackgroundMethodEnum = None
    exclusion: Path = None
    parameters: dict = {}


class SafeMaskConfig(GammapyBaseConfig):
    methods: List[SafeMaskMethodsEnum] = [SafeMaskMethodsEnum.aeff_default]
    parameters: dict = {}


class EnergyAxesConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig(min="1 TeV", max="10 TeV", nbins=5, name="energy")
    energy_true: EnergyAxisConfig = EnergyAxisConfig(
        min="0.5 TeV", max="20 TeV", nbins=16, name="energy_true"
    )


class SelectionConfig(GammapyBaseConfig):
    offset_max: AngleType = "2.5 deg"


class WidthConfig(GammapyBaseConfig):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class WcsConfig(GammapyBaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.02 deg"
    width: WidthConfig = WidthConfig()
    binsize_irf: AngleType = "0.2 deg"


class GeomConfig(GammapyBaseConfig):
#     wcs: WcsConfig = WcsConfig()
#     selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()


class DatasetsConfig(GammapyBaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
#     stack: bool = True
    geom: GeomConfig = GeomConfig()
    selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    use_region_center: bool = False
#     background: BackgroundConfig = BackgroundConfig()
#     on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = False
    safe_mask: SafeMaskConfig = SafeMaskConfig()
        

class DatasetsOnOffConfig(GammapyBaseConfig):
    acceptance: int = None
    acceptance_off: int = None
    
        
        
class SensitivityConfig(GammapyBaseConfig):
    gamma_min: int = None
    n_sigma: int = None
    bkg_syst_fraction: float = None
    table: dict = {}

# class WStatisticsConfig(GammapyBaseConfig):
#     excess: float = None
#     ts: float = None
#     sqrt_ts: float = None

class StatisticsConfig(GammapyBaseConfig):
    alpha: float = None
    wstat: dict = {}
    fitted_parameters: dict = {}   
                
class ObservationsParametersConfig(GammapyBaseConfig):
    livetime: QuantityType = None
    offset: QuantityType = None
    on_region_radius: AngleType = None
    n_obs: int = None


class TargetConfig(GammapyBaseConfig):
    name: str = None
    position: SkyCoordConfig = SkyCoordConfig()
    model: dict = {}  
    model_fitted: dict = {}
        
class IrfsConfig(GammapyBaseConfig):
    opt: IrfType = ['South', 'AverageAz', '20deg', '50h']

        
class PointingConfig(GammapyBaseConfig):
    angle: AngleType = None
        
class ObservationsConfig(GammapyBaseConfig):
    target: TargetConfig = TargetConfig()
    parameters: ObservationsParametersConfig = ObservationsParametersConfig()
    irfs: IrfsConfig = IrfsConfig()
    pointing: PointingConfig = PointingConfig()
        
class ROIConfig(GammapyBaseConfig):
    target: TargetConfig = TargetConfig()
    radius: AngleType = None
    catalogs: CatalogsTypeEnum = "all"
    dict_sep: dict = {} 
    leg_style: dict = {}
        
#     datastore: Path = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
#     obs_ids: List[int] = []
#     obs_file: Path = None
#     obs_cone: SpatialCircleConfig = SpatialCircleConfig()
#     obs_time: TimeRangeConfig = TimeRangeConfig()
#     required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]


class LogConfig(GammapyBaseConfig):
    level: str = "info"
    filename: Path = None
    filemode: str = None
    format: str = None
    datefmt: str = None


class GeneralConfig(GammapyBaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    n_jobs: int = 1
    datasets_file: Path = None
    models_file: Path = None

        

class GeneralCounterpartsConfig(GammapyBaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    path_file: Path = None


# In[ ]:





# In[ ]:


class CounterpartsConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralCounterpartsConfig = GeneralCounterpartsConfig()
    roi: ROIConfig = ROIConfig()
    energy_range: EnergyRangeConfig = EnergyRangeConfig()
#     irfs: IrfsConfig = IrfsConfig()
#     datasets: DatasetsConfig = DatasetsConfig()
#     sensitivity: SensitivityConfig = SensitivityConfig()
#     fit: FitConfig = FitConfig()
#     flux_points: FluxPointsConfig = FluxPointsConfig()
#     excess_map: ExcessMapConfig = ExcessMapConfig()
#     light_curve: LightCurveConfig = LightCurveConfig()

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Reads from YAML file."""
        config = read_yaml(path)
        return CounterpartsConfig(**config)


    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return CounterpartsConfig(**settings)


    def write(self, path = None, overwrite=False):
        """Write to YAML file."""
        if path is not None:
            path = make_path(path)
        else: path = make_path(f"{self.general.path_file}/counterparts_config.yaml")
            
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())


    def to_yaml(self):
        """Convert to YAML string."""
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(
            config, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))


    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : string dict or `CounterpartsConfig` object
            Configuration settings provided in dict() syntax.
        """
        if isinstance(config, str):
            other = CounterpartsConfig.from_yaml(config)
        elif isinstance(config, CounterpartsConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return CounterpartsConfig(**config_new)


    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc    


# In[ ]:





# In[ ]:


class SimulationConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    observations: ObservationsConfig = ObservationsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    datasets_onoff: DatasetsOnOffConfig = DatasetsOnOffConfig()
    statistics:StatisticsConfig = StatisticsConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()
#     fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Reads from YAML file."""
        config = read_yaml(path)
        return SimulationConfig(**config)


    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return SimulationConfig(**settings)


    def write(self, path, overwrite=False):
        """Write to YAML file."""
        path = make_path(path)
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())


    def to_yaml(self):
        """Convert to YAML string."""
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(
            config, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))


    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : string dict or `SimulationConfig` object
            Configuration settings provided in dict() syntax.
        """
        if isinstance(config, str):
            other = SimulationConfig.from_yaml(config)
        elif isinstance(config, SimulationConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return SimulationConfig(**config_new)


    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc


# In[ ]:




