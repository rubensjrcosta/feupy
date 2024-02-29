#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high level interface API"""
import logging
import yaml
import pandas as pd 
import json

from regions import CircleSkyRegion
from collections import defaultdict
from pathlib import Path
from typing import List


from astropy.units import Quantity
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# from pydantic import BaseModel
from pydantic.utils import deep_update

from gammapy.utils.units import energy_unit_format
from gammapy.utils.pbar import progress_bar
from gammapy.utils.scripts import make_path, read_yaml

from gammapy.datasets import (
    Datasets,  
    MapDataset, 
    FluxPointsDataset, 
    SpectrumDatasetOnOff, 
    SpectrumDataset,
)

from gammapy.estimators import FluxPoints, SensitivityEstimator
from gammapy.maps import Map, MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    SkyModel, 
    Models,
    Model,
    DatasetModels, 
    FoVBackgroundModel, 
    Models, 
    SkyModel, 
    ExpCutoffPowerLawSpectralModel
)

from gammapy.data import DataStore
from gammapy.estimators import (
    ExcessMapEstimator,
    FluxPointsEstimator,
    LightCurveEstimator,
)
from gammapy.makers import (
    FoVBackgroundMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    RingBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)

from gammapy.data import Observation

from gammapy.stats import WStatCountsStatistic

from feupy.utils.string_handling import name_to_txt
from feupy.utils.datasets import cut_energy_table_fp, write_datasets, read_datasets

# from feupy.analysis import CounterpartsAnalysisConfig, SimulationConfig, CTAObservationAnalysisConfig
from feupy.cta.irfs import Irfs
from feupy.utils.coordinates import skcoord_to_dict, dict_to_skcoord

from feupy.analysis.config import CounterpartsConfig, SimulationConfig

from feupy.plotters import *

from feupy.catalog.config import *

from gammapy.data import FixedPointingInfo, PointingMode


# In[ ]:


__all__ = ["Counterparts", "Simulation"]


# In[ ]:


log = logging.getLogger(__name__)

class Counterparts:
    """Config-driven high level simulation interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `CounterpartsConfig`
        Configuration options following `CounterpartsConfig` schema
    """

    def __init__(self, config):
        self.config = config
        self.config.set_logging()
#         self.roi = self._create_roi()
        
#         self._obs = self.config.observations
#         self._obs_params = self.config.observations.parameters
#         self._obs_target = self.config.observations.target
#         self._obs_irfs = self.config.observations.irfs
#         self._datasets = self.config.datasets

#         self.pointing_position = self._create_pointing_position(
#             dict_to_skcoord(self._obs_target.position), 
#             self._obs_params.offset)
#         self.pointing = self._create_pointing(self.pointing_position)

#         self._ctao_perf = Irfs
#         self._ctao_perf.get_irfs(self._obs_irfs.opt)
#         self.observation = self._create_observation(
#             dict_to_skcoord(self._obs_target.position), 
#             self._obs_params.livetime, 
#             self._ctao_perf.irfs, 
#             self._ctao_perf.obs_loc
#         )
        
    
#         self.datastore = None
#         self.observations = None
#         self.datasets = None
#         self.fit = Fit()
#         self.fit_result = None
#         self.flux_points = None
#         self.dataset_map = None
#         self.pointing = dict_to_skcoord(self.config.target.position)
#         self._datasets_settings = self.config.datasets
#         self._observations_settings = self.config.observations
#         self._ctao_perf = Irfs
#         self._ctao_perf.get_irfs(self.config.irfs.opt)
#         self.observation = None
#         self.geom = None
#         self.energy_axis_true = self._make_energy_axis(self._datasets_settings.geom.axes.energy_true)
#         self.energy_axis_reco = self._make_energy_axis(self._datasets_settings.geom.axes.energy)
#         self.spectrum_dataset_empty = None
#         self.maker = None
#         self.safe_maker = None
#         self.spectrum_dataset = None
#         self.spectrum_dataset_onoff = None
#         self.wstat = None
        
#     @staticmethod
#     def _create_region_geometry(center, axes):
#         """Create the region geometry."""
#         on_lon = target_position.lon
#         on_lat = target_position.lat
#         frame = target_position.frame
#         pointing = SkyCoord(on_lon, on_lat, frame=frame)
#         self._create_pointing_position(
#             dict_to_skcoord(self._obs_target.position), 
#             self._obs_params.offset)
        
#         on_center = pointing.directional_offset_by(
#             position_angle=pointing.dec, 
#             separation=offset)
#         on_region = CircleSkyRegion(on_center, on_region_settings.radius)
#         return 


#     def _create_roi(self):
#         """Create the geometry."""
#         log.debug("Creating geometry.")
#         target_settings = self.config.target
#         config.roi.target.position.lat
# config.roi.target.position.lon
# config.roi.region_radius
# config.roi.region_radius

#         obs_settings = self.config.observations
#         axes = [self._make_energy_axis(geom_settings.axes.energy)]
#         center = dict_to_skcoord(obs_settings.target.position)
#         radius = obs_settings.parameters.on_region_radius
#         region = self._create_on_region(center, radius)
#         return RegionGeom.create(region=region, axes=axes)
    
#     @staticmethod
#     def _create_on_region(center, radius):
#         """Create the region geometry.
#         on_region_radius :Angle()
#         """
#         return CircleSkyRegion(
#             center=center, 
#             radius=radius
#         )
    

#     @staticmethod
#     def _create_pointing_position(position, separation, position_angle = 0 * u.deg):
#         return position.directional_offset_by(position_angle, separation)
    
#     @staticmethod
#     def _create_pointing(pointing_position):
#         """Create pointing."""
#         return FixedPointingInfo(
#             mode=PointingMode.POINTING,
#             fixed_icrs=pointing_position.icrs,
#         )
    
#     @staticmethod
#     def _create_observation(pointing, livetime, irfs, location):
#         """Create an observation."""
#         return Observation.create(
#             pointing=pointing,
#             livetime=livetime,
#             irfs=irfs,
#             location=location,
#         )
    
    @property
    def config(self):
        """Simulation configuration (`CounterpartsConfig`)"""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = CounterpartsConfig(**value)
        elif isinstance(value, CounterpartsConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or CounterpartsConfig.")

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)
        
        
    @staticmethod
    def _make_energy_axis(config_axis_energy, per_decade=True):
        return MapAxis.from_energy_bounds(        
            energy_min=config_axis_energy.min, 
            energy_max=config_axis_energy.max, 
            nbin=config_axis_energy.nbins, 
            per_decade=per_decade, 
            name=config_axis_energy.name,
            )


# In[ ]:





# In[ ]:


log = logging.getLogger(__name__)

class Simulation:
    """Config-driven high level simulation interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `SimulationConfig`
        Configuration options following `SimulationConfig` schema
    """

    def __init__(self, config):
        self.config = config
        self.config.set_logging()
        self.geom = self._create_geometry()
        
        self._obs = self.config.observations
        self._obs_params = self.config.observations.parameters
        self._obs_target = self.config.observations.target
        self._obs_irfs = self.config.observations.irfs
        self._datasets = self.config.datasets

        self.pointing_position = self._create_pointing_position(
            dict_to_skcoord(self._obs_target.position), 
            self._obs_params.offset)
        self.pointing = self._create_pointing(self.pointing_position)

        self._ctao_perf = Irfs
        self._ctao_perf.get_irfs(self._obs_irfs.opt)
        self.observation = self._create_observation(
            dict_to_skcoord(self._obs_target.position), 
            self._obs_params.livetime, 
            self._ctao_perf.irfs, 
            self._ctao_perf.obs_loc
        )
        
#         self.datastore = None
#         self.observations = None
#         self.datasets = None
#         self.fit = Fit()
#         self.fit_result = None
#         self.flux_points = None
#         self.dataset_map = None
#         self.pointing = dict_to_skcoord(self.config.target.position)
#         self._datasets_settings = self.config.datasets
#         self._observations_settings = self.config.observations
#         self._ctao_perf = Irfs
#         self._ctao_perf.get_irfs(self.config.irfs.opt)
#         self.observation = None
#         self.geom = None
#         self.energy_axis_true = self._make_energy_axis(self._datasets_settings.geom.axes.energy_true)
#         self.energy_axis_reco = self._make_energy_axis(self._datasets_settings.geom.axes.energy)
#         self.spectrum_dataset_empty = None
#         self.maker = None
#         self.safe_maker = None
#         self.spectrum_dataset = None
#         self.spectrum_dataset_onoff = None
#         self.wstat = None
        
#     @staticmethod
#     def _create_region_geometry(center, axes):
#         """Create the region geometry."""
#         on_lon = target_position.lon
#         on_lat = target_position.lat
#         frame = target_position.frame
#         pointing = SkyCoord(on_lon, on_lat, frame=frame)
#         self._create_pointing_position(
#             dict_to_skcoord(self._obs_target.position), 
#             self._obs_params.offset)
        
#         on_center = pointing.directional_offset_by(
#             position_angle=pointing.dec, 
#             separation=offset)
#         on_region = CircleSkyRegion(on_center, on_region_settings.radius)
#         return 


    def _create_geometry(self):
        """Create the geometry."""
        log.debug("Creating geometry.")
        geom_settings = self.config.datasets.geom
        obs_settings = self.config.observations
        axes = [self._make_energy_axis(geom_settings.axes.energy)]
        center = dict_to_skcoord(obs_settings.target.position)
        radius = obs_settings.parameters.on_region_radius
        region = self._create_on_region(center, radius)
        return RegionGeom.create(region=region, axes=axes)
    
    @staticmethod
    def _create_on_region(center, radius):
        """Create the region geometry.
        on_region_radius :Angle()
        """
        return CircleSkyRegion(
            center=center, 
            radius=radius
        )
    

    @staticmethod
    def _create_pointing_position(position, separation, position_angle = 0 * u.deg):
        return position.directional_offset_by(position_angle, separation)
    
    @staticmethod
    def _create_pointing(pointing_position):
        """Create pointing."""
        return FixedPointingInfo(
            mode=PointingMode.POINTING,
            fixed_icrs=pointing_position.icrs,
        )
    
    @staticmethod
    def _create_observation(pointing, livetime, irfs, location):
        """Create an observation."""
        return Observation.create(
            pointing=pointing,
            livetime=livetime,
            irfs=irfs,
            location=location,
        )
    
    @property
    def config(self):
        """Simulation configuration (`SimulationConfig`)"""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = SimulationConfig(**value)
        elif isinstance(value, SimulationConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or SimulationConfig.")

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)
        
        
    @staticmethod
    def _make_energy_axis(config_axis_energy, per_decade=True):
        return MapAxis.from_energy_bounds(        
            energy_min=config_axis_energy.min, 
            energy_max=config_axis_energy.max, 
            nbin=config_axis_energy.nbins, 
            per_decade=per_decade, 
            name=config_axis_energy.name,
            )


# In[ ]:




