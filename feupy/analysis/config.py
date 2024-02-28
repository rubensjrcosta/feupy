#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pyflakes config.py


# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Configuration."""


# In[2]:


# from pydantic import BaseModel
from pydantic.v1 import BaseModel
from pydantic.v1.utils import lenient_isinstance
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





# In[3]:


__file__ = 'file'
__name__ = 'name'


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





# In[4]:


__all__ = ["CounterpartsAnalysisConfig", "CTAObservationAnalysisConfig", 'SimulationConfig']


# In[5]:


CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


# In[ ]:





# In[28]:


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
#     type: ReductionTypeEnum = ReductionTypeEnum.spectrum
#     stack: bool = True
    geom: GeomConfig = GeomConfig()
    selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    use_region_center: bool = False
#     background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
#     on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = False
    acceptance: int = None
    acceptance_off: int = None
    alpha: float = None
        
        
class SensitivityConfig(GammapyBaseConfig):
    gamma_min: int = None
    n_sigma: int = None
    bkg_syst_fraction: float = None

                
class ObservationsParametersConfig(GammapyBaseConfig):
    livetime: QuantityType = None
    offset: QuantityType = None
    on_region_radius: AngleType = None
    n_obs: int = None


class TargetConfig(GammapyBaseConfig):
    name: str = None
    position: SkyCoordConfig = SkyCoordConfig()
    model:  dict = {}  
        
class IrfsConfig(GammapyBaseConfig):
    opt: IrfType = ['South', 'AverageAz', '20deg', '50h']

class PointingConfig(GammapyBaseConfig):
    angle: AngleType = 0 * u.deg
        
class ObservationsConfig(GammapyBaseConfig):
    target: TargetConfig = TargetConfig()
    parameters: ObservationsParametersConfig = ObservationsParametersConfig()
    irfs: IrfsConfig = IrfsConfig()
    pointing: PointingConfig = PointingConfig()
        
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

        

        


# In[ ]:





# In[29]:


class SimulationConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    observations: ObservationsConfig = ObservationsConfig()
#     irfs: IrfsConfig = IrfsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()
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


# In[30]:


config = SimulationConfig()
print(config)


# In[ ]:





# In[31]:


from feupy.catalog.pulsar.atnf import SourceCatalogATNF

catalog = SourceCatalogATNF()
source = catalog['PSR J1826-1256']
name = source.name
pos_ra = source.position.ra
pos_dec = source.position.dec
from gammapy.modeling.models import SkyModel
from gammapy.modeling.models import ExpCutoffPowerLawSpectralModel
model = SkyModel(spectral_model=ExpCutoffPowerLawSpectralModel(), name=name)
print(model)
target = Target(name, pos_ra, pos_dec, model)


# In[32]:


config.observations.target = target.dict


# In[33]:


livetime=10*u.h 
offset=0.11*u.deg 
on_region_radius=0.5*u.deg 
n_obs=10

obs_params = ObservationParameters(
    livetime=livetime, 
    on_region_radius = on_region_radius,
    offset=offset,
    n_obs=n_obs,
)
print(obs_params)


# In[34]:


config.observations.parameters = obs_params.dict


# In[35]:


print(config)


# In[36]:


irfs_opt = ['South', 'AverageAz', '20deg', '50h']
config.observations.irfs.opt = irfs_opt
print(config)


# In[ ]:





# In[ ]:





# In[38]:


e_edges_min=0.1*u.TeV 
e_edges_max=100.*u.TeV


# In[39]:


# We now fix the energy axis for the counts map
config.datasets.geom.axes.energy.min = ".1 TeV"
config.datasets.geom.axes.energy.max = "100 TeV"
config.datasets.geom.axes.energy.nbins = 5
config.datasets.geom.axes.energy.name = 'energy'

# We now fix the energy axis for the IRF maps (exposure, etc)
# config.datasets.geom.axes.energy_true.min = "0.5 TeV"
# config.datasets.geom.axes.energy_true.max = "20 TeV"
# config.datasets.geom.axes.energy_true.nbins = 20
config.datasets.geom.axes.energy_true.min = .3*e_edges_min
config.datasets.geom.axes.energy_true.max = 3*e_edges_max
config.datasets.geom.axes.energy_true.nbins = 8
config.datasets.geom.axes.energy_true.name = 'energy_true'
print(config)


# In[41]:


config.write("config.yaml", overwrite=True)

config = SimulationConfig.read("config.yaml")
print(config)


# In[ ]:


# How to create a class:
class CounterpartsAnalysisConfig:
   # ADD others parameters
   
    # color="red" # The color of the flux ponts
    all=[]
#     @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg, radius=u.deg, e_ref_min=u.eV, e_ref_max=u.eV)
    def __init__(self, 
                 roi,
                 e_ref_min=None, 
                 e_ref_max=None
                ):
  
        # Assign to self object
        self.roi = roi 
        self.target = self.roi.target
        if e_ref_min is not None:
            self.e_ref_min = Quantity(e_ref_min, "TeV")
        else: self.e_ref_min = e_ref_min
        if e_ref_max is not None:
            self.e_ref_max = Quantity(e_ref_max, "TeV")
        else: self.e_ref_max = e_ref_max
        self.energy_range = [self.e_ref_min, self.e_ref_max]
          
        # Actions to execute
        CounterpartsAnalysisConfig.all.append(self)
    
    @property
    def roi(self):
        """ROI as an `~feupy.roi.ROI` object."""
        return self._roi

    @roi.setter
    def roi(self, value):
        if isinstance(value, ROI):
            self._roi = value
        else:
            raise TypeError("roi must be ROI")        
            
#     @property
#     # Property Decorator=Read-Only Attribute
#     def info(self):
#         info={}
#         info["target_name"] = self.target_name
#         info["position"] = self.target.position
#         info["radius"] = self.radius
#         info["energy_range"] = self.energy_range
#         return info    
    
#     @property
#     def target_name(self):
#         return self.__target_name

    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"target_name={self.target.name}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.target.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'), ".format(self.target.position.dec).replace(' ', '')
        ss += "radius=Quantity('{:.2f}'), ".format(self.roi.radius).replace(' ', '')
        if self.e_ref_min is None: ss += "e_ref_min=None, "
        else: ss += "e_ref_min=Quantity('{}'), ".format(energy_unit_format(self.e_ref_min).replace(' ', ''))
        if self.e_ref_max is None: ss += "e_ref_max=None)"
        else: ss += "e_ref_max=Quantity('{}'))".format(energy_unit_format(self.e_ref_max).replace(' ', ''))
        return ss 


# In[ ]:


# How to create a class:
class CTAObservationAnalysisConfig:
   # ADD others parameters
   
    # color="red" # The color of the flux ponts
    all=[]
#     @u.quantity_input(livetime=u.h, on_region_radius=u.deg, offset=u.deg, e_edges_min=u.eV, e_edges_max=u.eV)
    def __init__(self,
                 target,
                 obs_params, 
                 irfs_opt, 
                 geom_params
                ):
        self.target = target
        self.obs_params = obs_params
        self.ctao_perf = Irfs
        self.ctao_perf.get_irfs(irfs_opt)
        
        self.geom_params = geom_params
        self.energy_axis_true = None
        self.energy_axis_reco = None
        self._set_energy_axis()
        self.pointing = self.target.position
        self.observation = create_observation(
            self.pointing, 
            self.obs_params.livetime, 
            self.ctao_perf.irfs, 
            self.ctao_perf.obs_loc
        )
        self.on_region = define_on_region(
            self._obs_center(), 
            self.obs_params.on_region_radius
        )
        self.geom = create_region_geometry(self.on_region, [self.energy_axis_reco])
            
        # Actions to execute
        CTAObservationAnalysisConfig.all.append(self)
#         if self.irfs:
# #             self.irfs = irfs
#             self.irfs_label = Irfs.get_irfs_label(self.irfs)
#             self.obs_location = Irfs.get_obs_loc(self.irfs_label)
   

    @property
    def obs_params(self):
        """Analysis configuration as an `~feupy.cta.ObservationParameters` object."""
        return self._obs_params

    @obs_params.setter
    def obs_params(self, value):
        if isinstance(value, ObservationParameters):
            self._obs_params = value
        else:
            raise TypeError("params must be ObservationParameters")
    
    @property
    def target(self):
        """Target as an `~feupy.target.Target` object."""
        return self._target

    @target.setter
    def target(self, value):
        if isinstance(value, Target):
            self._target = value
        else:
            raise TypeError("target must be Target") 
            
    @property
    def geom_params(self):
        """Analysis configuration as an `~feupy.cta.ObservationParameters` object."""
        return self._geom_params

    @geom_params.setter
    def geom_params(self, value):
        if isinstance(value, GeometryParameters):
            self._geom_params = value
        else:
            raise TypeError("params must be ObservationParameters")
            
    def _obs_center(self):
            return self.pointing.directional_offset_by(
                position_angle=self.pointing.dec, 
                separation=self.obs_params.offset
            )

        
    def _set_energy_axis(self):
        self.energy_axis_true = create_energy_axis(
            self.geom_params.e_true_min,
            self.geom_params.e_true_max,
            self.geom_params.nbin_true,
            per_decade=True,
            name='energy_true',
        )
        self.energy_axis_reco = create_energy_axis(
            energy_min=self.geom_params.e_reco_min,
            energy_max=self.geom_params.e_reco_max,
            nbin=self.geom_params.nbin_reco,
            per_decade=True,
            name='energy',
        )
        
            
#     def info(self):
#         info={}
#         info["target_name"] = self.target_name
#         info["position"] = self.position
#         info["radius"] = self.radius
#         info["energy_range"] = self.energy_range
#     return info    

            
#     @property
#     # Property Decorator=Read-Only Attribute
#     def info(self):
#         info={}
#         info["target_name"] = self.target_name
#         info["position"] = self.position
#         info["radius"] = self.radius
#         info["energy_range"] = self.energy_range
#         return info    
    
#     @property
#     def target_name(self):
#         return self.__target_name

#     def __repr__(self):
#         ss = f"{self.__class__.__name__}("
#         ss += f"target_name={self.__target_name}, "
#         ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
#         ss += "pos_dec=Quantity('{:.2f}'), ".format(self.position.dec).replace(' ', '')
#         ss += "radius=Quantity('{:.2f}'), ".format(self.radius).replace(' ', '')
#         if self.e_ref_min is None: ss += "e_ref_min=None, "
#         else: ss += "e_ref_min=Quantity('{}'), ".format(energy_unit_format(self.e_ref_min).replace(' ', ''))
#         if self.e_ref_max is None: ss += "e_ref_max=None)"
#         else: ss += "e_ref_max=Quantity('{}'))".format(energy_unit_format(self.e_ref_max).replace(' ', ''))
#         return ss 


# In[ ]:


# How to create a class:
class CTAObservationAnalysisConfig__:
   # ADD others parameters
   
    # color="red" # The color of the flux ponts
    all=[]
#     @u.quantity_input(livetime=u.h, on_region_radius=u.deg, offset=u.deg, e_edges_min=u.eV, e_edges_max=u.eV)
    def __init__(self,
                 target,
                 obs_params, 
                 irfs_opt, 
                 geom_params
                ):
        self.obs_params = obs_params
        self.target = target
        self.irfs_opt = irfs_opt
        self.irfs = Irfs.get_irfs(irfs_opt)
        self.irfs_label = Irfs.irfs_label
        self.obs_location = Irfs.obs_loc
        self.geom_params = geom_params
        self.energy_axis_true = None
        self.energy_axis_reco = None
        self._set_energy_axis()
        self.pointing = self.target.position
        self.observation = create_observation(self.pointing, self.obs_params.livetime, self.irfs, self.obs_location)
        self.on_region = define_on_region(self._obs_center(), self.obs_params.on_region_radius)
        self.geom = create_region_geometry(self.on_region, [self.energy_axis_reco])
            
        # Actions to execute
        CTAObservationAnalysisConfig.all.append(self)
#         if self.irfs:
# #             self.irfs = irfs
#             self.irfs_label = Irfs.get_irfs_label(self.irfs)
#             self.obs_location = Irfs.get_obs_loc(self.irfs_label)
   

    @property
    def obs_params(self):
        """Analysis configuration as an `~feupy.cta.ObservationParameters` object."""
        return self._obs_params

    @obs_params.setter
    def obs_params(self, value):
        if isinstance(value, ObservationParameters):
            self._obs_params = value
        else:
            raise TypeError("params must be ObservationParameters")
    
    @property
    def target(self):
        """Target as an `~feupy.target.Target` object."""
        return self._target

    @target.setter
    def target(self, value):
        if isinstance(value, Target):
            self._target = value
        else:
            raise TypeError("target must be Target") 
            
    @property
    def geom_params(self):
        """Analysis configuration as an `~feupy.cta.ObservationParameters` object."""
        return self._geom_params

    @geom_params.setter
    def geom_params(self, value):
        if isinstance(value, GeometryParameters):
            self._geom_params = value
        else:
            raise TypeError("params must be ObservationParameters")
            
    def _obs_center(self):
            return self.pointing.directional_offset_by(position_angle=self.pointing.dec, separation=self.obs_params.offset)

        
    def _set_energy_axis(self):
        self.energy_axis_true = create_energy_axis(
            self.geom_params.e_true_min,
            self.geom_params.e_true_max,
            self.geom_params.nbin_true,
            per_decade=True,
            name='energy_true',
        )
        self.energy_axis_reco = create_energy_axis(
            energy_min=self.geom_params.e_reco_min,
            energy_max=self.geom_params.e_reco_max,
            nbin=self.geom_params.nbin_reco,
            per_decade=True,
            name='energy',
        )
        
            
#     def info(self):
#         info={}
#         info["target_name"] = self.target_name
#         info["position"] = self.position
#         info["radius"] = self.radius
#         info["energy_range"] = self.energy_range
#     return info    

            
#     @property
#     # Property Decorator=Read-Only Attribute
#     def info(self):
#         info={}
#         info["target_name"] = self.target_name
#         info["position"] = self.position
#         info["radius"] = self.radius
#         info["energy_range"] = self.energy_range
#         return info    
    
#     @property
#     def target_name(self):
#         return self.__target_name

#     def __repr__(self):
#         ss = f"{self.__class__.__name__}("
#         ss += f"target_name={self.__target_name}, "
#         ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
#         ss += "pos_dec=Quantity('{:.2f}'), ".format(self.position.dec).replace(' ', '')
#         ss += "radius=Quantity('{:.2f}'), ".format(self.radius).replace(' ', '')
#         if self.e_ref_min is None: ss += "e_ref_min=None, "
#         else: ss += "e_ref_min=Quantity('{}'), ".format(energy_unit_format(self.e_ref_min).replace(' ', ''))
#         if self.e_ref_max is None: ss += "e_ref_max=None)"
#         else: ss += "e_ref_max=Quantity('{}'))".format(energy_unit_format(self.e_ref_max).replace(' ', ''))
#         return ss 


# In[ ]:





# In[ ]:


# from feupy.tests import test_roi, test_target, test_cta_obs_parm


# target = test_target()
# params = test_cta_obs_parm()

# config = CTAObservationAnalysisConfig(target, params)

# config.params.on_region_radius

# config.target



# In[ ]:


# label = Irfs.get_irfs_label(irfs)
# location = Irfs.get_obs_loc(label)


# In[ ]:


# config.irfs, config.irfs_label, config.obs_location


# In[ ]:


# from feupy.target import Target
# from feupy.roi import ROI
# from feupy.analysis.simulation.geometry import *
# from feupy.analysis.simulation.datasets import *

# from astropy import units as u
# from astropy.units import Quantity
# from gammapy.modeling.models import (
#     PowerLawSpectralModel,
#     SkyModel,
# )
# from astropy.coordinates import Angle

# name = "LHAASO J1825-1326"
# pos_ra = u.Quantity("276.45deg") 
# pos_dec = -13.45* u.Unit('deg')

# on_region_radius = on_region_radius=Angle("1.0 deg")
# spec_model = PowerLawSpectralModel()
# target = Target(name, pos_ra, pos_dec, spectral_model=spec_model)
# roi = ROI(target, radius=on_region_radius)

# print(target.info)

# target_position = target.position

# print(roi.info)

# model = target.model
# model_name = model.name
# print(model)

# #### Define Observational Parameters

# # from feupy.cta.irfs import Irfs
# from feupy.analysis.simulation import ObservationParameters

# params=ObservationParameters(
#     livetime=50*u.h, 
#     offset=0.11*u.deg, 
#     e_edges_min=0.1*u.TeV, 
#     e_edges_max=100.*u.TeV,
#     on_region_radius=Angle("1.0 deg"),
#     n_obs=10
# )
# print(params)

# livetime = params.livetime 
# offset = params.offset
# e_edges_min = params.e_edges_min 
# e_edges_max = params.e_edges_max
# on_region_radius = params.on_region_radius 
# n_obs = params.n_obs


# # Defines reconstructed energy axis bounds
# e_reco_min=e_edges_min 
# e_reco_max=e_edges_max 
# nbin_reco=5

# # Defines the true energy axis:
# e_true_min=e_edges_min*.3
# e_true_max=e_edges_max*3
# nbin_true=8

# geom_params = GeometryParameters(
#     e_reco_min=e_reco_min, 
#     e_reco_max=e_reco_max, 
#     nbin_reco=nbin_reco,
#     e_true_min=e_true_min,
#     e_true_max=e_true_max,
#     nbin_true=nbin_true,
# )
# print(geom_params)

# Irfs.load_all_irfs()

# irfs = Irfs.irfs_list[3]

# # from feupy.analysis.config import CTAObservationAnalysisConfig as AnalysisConfig


# In[ ]:


# config = CTAObservationAnalysisConfig(target, params, irfs, geom_params)


# In[ ]:




