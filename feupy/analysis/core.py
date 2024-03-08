#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high level interface API"""
import logging
import yaml
import pandas as pd 
import json
from gammapy.modeling.models import Model
from regions import CircleSkyRegion
from collections import defaultdict
from pathlib import Path
from typing import List

import time

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
from feupy.utils.datasets import flux_points_dataset_from_table


from gammapy.utils.units import energy_unit_format


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
from feupy.analysis.simulation.observations import sensitivity_estimator
from feupy.analysis.simulation.stats import StatisticalUtilityFunctions as stats

from feupy.utils.string_handling import name_to_txt
from feupy.utils.datasets import cut_energy_table_fp, write_datasets, read_datasets

# from feupy.analysis import CounterpartsAnalysisConfig, SimulationConfig, CTAObservationAnalysisConfig
from feupy.cta.irfs import Irfs
from feupy.utils.coordinates import skcoord_to_dict, skcoord_config_to_skcoord

from feupy.analysis.config import CounterpartsConfig, SimulationConfig

from feupy.plotters import *

# from feupy.catalog.config import *
from feupy.plotters import *
from feupy.roi import ROI
from feupy.target import Target

from feupy.catalog.pulsar.atnf import SourceCatalogATNF

from gammapy.data import FixedPointingInfo, PointingMode

from astropy.io.fits.verify import VerifyWarning
import warnings


# In[2]:


__all__ = ["Counterparts", "Simulation"]


# In[3]:


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
        self.roi = self._create_roi()
        self.roi.get_catalogs()
        self.target = None
        self.catalogs = None
        self.datasets = None
        self.sources = None
        self.pulsars = None
        self.df_sep = None
        self.leg_style = None
        
    def _set_leg_style(self):
        datasets_names = self.datasets.names
        models_names = self.datasets.models.names
        for pulsar in self.pulsars:
            name = pulsar.name
            datasets_names.append(name)
        leg_style = set_leg_style(
            leg_style ={}, 
            datasets_names=datasets_names, 
            models_names=models_names
        )
        self.leg_style = leg_style
        
        
    def _create_target(self):
        """Create the target."""
        log.debug("Creating target.")
        target_settings = self.config.roi.target
        name = target_settings.name
        pos_ra = target_settings.position.lon
        pos_dec = target_settings.position.lat
        model = Model.from_dict(target_settings.model)
        return Target(name, pos_ra, pos_dec, spectral_model=model.spectral_model)
    
    def _create_roi(self):
        """Create the target."""
        log.debug("Creating target.")
        target = self._create_target()
        self.target = target
        on_region_radius = self.config.roi.radius
        return ROI(target, on_region_radius )
        
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
    
    def run(self):
        self._get_datasets()
        self._set_leg_style()
        
    def _get_datasets(self):
        """
        Select a catalog subset (only sources within a region of interest)
        """
        _catalogs = self.roi.catalogs
        self.pulsars = self.roi.pulsars
        pulsars = self.pulsars
        e_ref_min = self.config.energy_range.min
        e_ref_max = self.config.energy_range.max

        
        datasets = Datasets() # global datasets object
        models = Models()  # global models object
        sources = [] # global sources object
        catalogs = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            for catalog in _catalogs:
                indexes = []
                cat_tag = catalog.tag
                for source in catalog: 
                    source_name = source.name 
                    index = source.row_index
                    if cat_tag == SourceCatalogATNF.tag:
                        pass
                    else:
                        try:
                            flux_points = source.flux_points

                            spectral_model = source.spectral_model()
                            spectral_model_tag = spectral_model.tag[1]

                            if cat_tag == 'gamma-cat' or cat_tag == 'hgps':
                                dataset_name = f'{source_name}: {cat_tag}'
                            else: dataset_name = source_name

                            file_name = name_to_txt(dataset_name)

                            model = SkyModel(
                                name=f"{file_name}_{spectral_model_tag}",
                                spectral_model=spectral_model,
                                datasets_names=dataset_name
                            )

                            dataset = FluxPointsDataset(
                                models=model,
                                data=flux_points, 
                                name=dataset_name   
                            )
                            
                            if any([e_ref_min !=  None, e_ref_max !=  None]):
                                dataset = cut_energy_table_fp(dataset, e_ref_min, e_ref_max) 

                            models.append(model)  # Add the model to models()
                            datasets.append(dataset)
                            sources.append(source)
                        except Exception as error:
                            indexes.append(index)
                            # By this way we can know about the type of error occurring
                            print(f'The error is: ({source_name}) {error}') 
                if len(indexes)>0:
                    if len(indexes)==1:
                        catalog.table.remove_row(indexes[0])
                    else: catalog.table.remove_rows(indexes)

                if len(catalog.table)>0:
                    catalogs.append(catalog)
            datasets.models = models
            self.datasets = datasets
            self.sources = sources
            self.catalogs = catalogs
            
            target_pos = self.roi.target.position
            _sources = sources.copy()
            _sources.extend(pulsars)
            
            dict_sep = self.roi.get_dict_sep(target_pos, _sources, opt="pos_dict")
            self.config.roi.dict_sep = dict_sep
            self.df_sep = self.roi.get_df_sep(dict_sep) 
        
            print(f"Total number of gamma sources: {len(self.sources)}")
            print(f"Total number of flux points tables: {len(self.datasets)}")
            print(f"Total number of pulsars: {len(self.pulsars)}")
             
                
    def create_analysis_name(self): 
        """ ... """
        ss = f"{self.config.target.name}"
        ss += "_roi_{:.2f}".format(self.roi.radius).replace(' ', '')
        if e_ref_min is None: ss += ""
        else: ss += "_e_ref_min_{}".format(energy_unit_format(e_ref_min).replace(' ', ''))
        if e_ref_max is None: ss += ""
        else: ss += "_e_ref_max_{}".format(energy_unit_format(e_ref_max).replace(' ', ''))
        return ss
    
#     def create_analysis_path(self): 
#         """ ... """
#         return Path(f"analysis_counterparts/{self.create_analysis_name()}")

    def set_models(self, models, extend=True):
        """Set models on datasets.
        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        extend : bool
            Extend the exiting models on the datasets or replace them.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info("Reading model.")
        if isinstance(models, str):
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, DatasetModels) or isinstance(models, list):
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            models.extend(self.datasets.models)

        self.datasets.models = models


        log.info(models)

    def write_datasets(self, overwrite=True, path_file=None):
        """Write Datasets and Models to YAML file.

            Parameters
            ----------
            overwrite : bool, optional
                Overwrite existing file. Default is True.  
            """
        
        if path_file is None:
            path_file = Path(f"{self.config.create_analysis_path()}/datasets")
        write_datasets(self.datasets, path_file, overwrite)
    
    def read_datasets(self, path_file=None):
        """Read Datasets and Models from YAML file."""

        if path_file is None:
            path_file = Path(f"{self.config.create_analysis_path()}/datasets")
        return read_datasets(path_file)


# In[ ]:





# In[4]:


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
    
        self._ctao_perf = Irfs
        self._ctao_perf.get_irfs(self.config.observations.irfs.opt)
        self.fit = Fit()
        self.fit_result = None
        self.flux_points = None
     
        
    def _create_observation(self):
        """Create an observation."""
        observations_settings = self.config.observations
        position = skcoord_config_to_skcoord(observations_settings.target.position)
        print(f"\nposition:\n{position}\n")
        position_angle = observations_settings.pointing.angle
        separation = observations_settings.parameters.offset
        pointing_position = self._create_pointing_position(position, position_angle, separation)
        print(f"\npointing_position:\n{pointing_position}\n")
        pointing = self._create_pointing(pointing_position)
        print(f"\npointing:\n{pointing}\n")
        livetime = observations_settings.parameters.livetime
        irfs = self._ctao_perf.irfs
        location = self._ctao_perf.obs_loc
        observation = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs, location=location)
        print(f"\n{observation}\n")
        return observation
        
    @staticmethod
    def _create_pointing_position(position, position_angle, separation):
        """Create the pointing position"""
        return position.directional_offset_by(position_angle, separation)
    
    @staticmethod
    def _create_pointing(pointing_position):
        """Create the pointing."""
        return FixedPointingInfo(
            mode=PointingMode.POINTING,
            fixed_icrs=pointing_position.icrs,
        )
    
    
    def _create_geometry(self):
        """Create the geometry."""
        geom_settings = self.config.datasets.geom
        observations_settings = self.config.observations
        axes = [self._make_energy_axis(geom_settings.axes.energy)]
        print(f"\nenergy axis:\n{axes[0]}\n")
        center = skcoord_config_to_skcoord(observations_settings.target.position)
        radius = observations_settings.parameters.on_region_radius
        region = self._create_on_region(center, radius)
        
        return RegionGeom.create(region=region, axes=axes)
    
    @staticmethod
    def _make_energy_axis(config_axis_energy, per_decade=True):
        """Create the energy axis."""
        energy_axis = MapAxis.from_energy_bounds(        
            energy_min=config_axis_energy.min, 
            energy_max=config_axis_energy.max, 
            nbin=config_axis_energy.nbins, 
            per_decade=per_decade, 
            name=config_axis_energy.name,
            )
        return energy_axis
    
    @staticmethod
    def _create_on_region(center, radius):
        """Create the region geometry."""
        return CircleSkyRegion(
            center=center, 
            radius=radius
        )
    
    
    def _create_spectrum_dataset_empty(self, name="obs-0"):
        """# Creates a Spectrum Dataset object with zero filled maps."""
        geom_settings = self.config.datasets.geom
        geom = self._create_geometry()
        print(f"\ngeometry:\n{geom}\n")
        energy_axis_true = self._make_energy_axis(geom_settings.axes.energy_true)
        print(f"\nenergy axis true:\n{energy_axis_true}\n")
        return SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true, name=name)
    
    def _create_dataset_maker(self):
        """Create the Spectrum Dataset Maker."""
        datasets_settings = self.config.datasets
            
        if datasets_settings.type == "3d":
            maker = MapDatasetMaker(selection=datasets_settings.selection)
            
        elif datasets_settings.type == "1d":
            maker_config = {}
            if datasets_settings.containment_correction:
                maker_config[
                    "containment_correction"
                ] = datasets_settings.containment_correction

            maker_config["selection"] = datasets_settings.selection
            maker_config["use_region_center"] = datasets_settings.use_region_center
            maker = SpectrumDatasetMaker(**maker_config)

        return maker
    
    def _create_safe_mask_maker(self):
        """Create the Safe Mask Maker."""
        safe_mask_selection = self.config.datasets.safe_mask.methods
        safe_mask_settings = self.config.datasets.safe_mask.parameters

        return SafeMaskMaker(methods=safe_mask_selection, **safe_mask_settings)
        
        
    def set_model_on_dataset(self, random_state=42):
        """Make maps and datasets for 3d analysis"""
        log.debug("Creating observation.")
        observation = self._create_observation()
        target_settings = self.config.observations.target
        
        log.info("Creating reference the dataset, maker and safe maker.")
        dataset_empty = self._create_spectrum_dataset_empty()
        print(f"\nDataset Empty:\n{dataset_empty}\n")
        
        maker = self._create_dataset_maker()
        safe_maker = self._create_safe_mask_maker()
        log.info("Running makers and safing.")
        dataset = maker.run(dataset_empty, observation)
        dataset = safe_maker.run(dataset, observation)
        
        log.info("Set the model on the dataset, and fake.")
        dataset.models = Model.from_dict(target_settings.model)
        dataset.fake(random_state=random_state)
        self.spectrum_dataset = dataset
        print(f"\n:\n{dataset}\n")

    @staticmethod    
    def _create_safe_spectrum_dataset_onoff(dataset, acceptance, acceptance_off):
    # Spectrum dataset for on-off likelihood fitting.
        dataset_onoff = SpectrumDatasetOnOff.from_spectrum_dataset(
            dataset=dataset, 
            acceptance=acceptance, 
            acceptance_off=acceptance_off,
        )
        dataset_onoff.fake(
            random_state='random-seed', 
            npred_background=dataset.npred_background()
        )
        return(dataset_onoff)
    
    def run_onoff(self): 
        n_obs = self.config.observations.parameters.n_obs
        dataset = self.spectrum_dataset
        datasets_onoff_settings = self.config.datasets_onoff
        stat_settings = self.config.statistics
        sens_settings = self.config.sensitivity

        alpha = stat_settings.alpha
        acceptance = self.config.datasets_onoff.acceptance 
        acceptance_off = self.config.datasets_onoff.acceptance_off
        dataset_onoff = self._create_safe_spectrum_dataset_onoff(dataset, acceptance, acceptance_off)
        
        self.spectrum_dataset_onoff = dataset_onoff
        _wstat, wstat_dict = self._compute_wstat(dataset_onoff, alpha)
        self.config.statistics.wstat = wstat_dict
        self.wstat_dict = wstat_dict
        self.wstat = _wstat
        self.update_config(self.config)
        
        spectrum = Model.from_dict(self.config.observations.target.model).spectral_model
        gamma_min = sens_settings.gamma_min  
        n_sigma = sens_settings.n_sigma 
        bkg_syst_fraction = sens_settings.bkg_syst_fraction
        fp_settings = self.config.flux_points
        sens, sensitivity_ds, sensitivity_table = self._compute_sensitivity(spectrum, gamma_min, n_sigma, bkg_syst_fraction, dataset_onoff, sed_type="e2dnde", name=fp_settings.source)
        self.sens = sens 
        self.sensitivity_ds = sensitivity_ds
        self.sensitivity_table = sensitivity_table
        
        
        datasets = Datasets()

        for idx in range(n_obs):
            dataset_onoff.fake(
                random_state=idx, 
                npred_background=dataset.npred_background()
            )
            dataset_fake = dataset_onoff.copy(name=f"obs-{idx}")
            dataset_fake.meta_table["OBS_ID"] = [idx]
            datasets.append(dataset_fake)
        self.datasets = datasets
        self.table_counts = datasets.info_table()
        self.dataset_stacked = datasets.stack_reduce(name=f"stacked {fp_settings.source}".replace("model", ""))
    
    @staticmethod    
    def _compute_wstat(dataset_onoff, alpha):
        log.info("computing wstatistics.")
        wstat = stats.compute_wstat(dataset_onoff=dataset_onoff, alpha=alpha)
        wstat_dict = wstat.info_dict()
        wstat_dict["n_on"] = float(wstat_dict["n_on"])
        wstat_dict["n_off"] = float(wstat_dict["n_off"])
        wstat_dict["background"] = float(wstat_dict["background"])
        wstat_dict["excess"] = float(wstat_dict["excess"])
        wstat_dict["significance"] = float(wstat_dict["significance"])
        wstat_dict["p_value"] = float(wstat_dict["p_value"])
        wstat_dict["alpha"] = float(wstat_dict["alpha"])
        wstat_dict["mu_sig"] =float(wstat_dict["mu_sig"])

        wstat_dict['error'] = float(wstat.error)
        wstat_dict['stat_null'] = float(wstat.stat_null)
        wstat_dict['stat_max'] = float(wstat.stat_max)
        wstat_dict['ts'] = float(wstat.ts)
        print(f"Number of excess counts: {wstat.n_sig}")
        print(f"TS: {wstat.ts}")
        print(f"Significance: {wstat.sqrt_ts}")
        return wstat, wstat_dict
    
    @staticmethod    
    def _compute_sensitivity(spectrum, gamma_min, n_sigma, bkg_syst_fraction, dataset_onoff, sed_type="e2dnde", name="sens"):
        log.info("computing sensitivity.")
        
        sens, sensitivity_table = sensitivity_estimator(
            spectrum=spectrum,
            gamma_min=gamma_min, 
            n_sigma=n_sigma, 
            bkg_syst_fraction=bkg_syst_fraction, 
            dataset_onoff=dataset_onoff)
        name = name.replace("model", "")
        model_name = f"model sens {name}"
        name = f"sens {name}"
       
        sensitivity_ds = flux_points_dataset_from_table(
            sensitivity_table, 
            reference_model=spectrum.copy(),
            sed_type=sed_type,
            name=name,
            model_name=model_name)

        return sens, sensitivity_ds, sensitivity_table

    def fit_model_parameters(self): 
        datasets = self.datasets
        model = Model.from_dict(self.config.observations.target.model)
        fitted_parameters, fitted_parameters_dict = self._fit_params(datasets, model)
        self.config.statistics.fitted_parameters = fitted_parameters_dict
        self.update_config(self.config)
        self.fitted_parameters = fitted_parameters
        self.fitted_parameters_dict = fitted_parameters_dict
        
    @staticmethod    
    def _fit_params(datasets, model):
#         %%time

        results = []

        fit = Fit()

        for dataset in datasets.copy():
            dataset.models = model.copy()
            result = fit.optimize(dataset)

            if result.success:
                par_dict = {}
                for par in result.parameters.free_parameters:
                    par_dict[par.name] = par.quantity
                results.append(par_dict)

        fitted_params = Table(results).to_pandas()
        mean = fitted_params.mean()
        uncertainty = fitted_params.std()
        fitted_params_dict = {}
        for name in list(results[0].keys()):
            fitted_params_dict[name] = {
                "mean": mean[name],
                "uncertainty": uncertainty[name]
            }
            print(f"{name} :\t {mean[name]:.2e} -+ {uncertainty[name]:.2e}")
    
        return fitted_params, fitted_params_dict
    
    def estimate_flux_points(self):
        """Estimate flux points for a specific model component."""
        if not self.datasets:
            raise RuntimeError(
                "No datasets defined. Impossible to compute flux points."
            )

        fp_settings = self.config.flux_points
        log.info("Estimating flux points.")
        energy_edges = self._make_energy_axis(fp_settings.energy).edges
        flux_point_estimator = FluxPointsEstimator(
            energy_edges=energy_edges,
            source=fp_settings.source,
            fit=self.fit,
            n_jobs=self.config.general.n_jobs,
            **fp_settings.parameters,
        )

        fp = flux_point_estimator.run(datasets=self.datasets)

        self.flux_points = FluxPointsDataset(
            data=fp, models=self.models[fp_settings.source], name=f"{fp_settings.source}".replace("model", "")
        )
        
        cols = ["e_ref", "dnde", "dnde_ul", "dnde_err", "sqrt_ts"]
        table = self.flux_points.data.to_table(sed_type="dnde")
        log.info("\n{}".format(table[cols]))


    def fit_joint(self):
        datasets = self.datasets
        model = Model.from_dict(self.config.observations.target.model)

        #Compute flux points
        datasets.models = [model]

        # fit_joint = Fit(backend='sherpa')
        fit_joint = Fit()
        fit_result_joint = fit_joint.run(datasets=datasets)
        print(fit_result_joint)
        self.datasets.models = model
        self.config.observations.target.model_fitted = model.to_dict()
        self.update_config(self.config)

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
        
    def update_config(self, config):
        """Update the configuration."""
        self.config = self.config.update(config=config)
        
    def set_models(self, models, extend=True):
        """Set models on datasets.
        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        extend : bool
            Extend the exiting models on the datasets or replace them.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info("Reading model.")
        if isinstance(models, str):
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, DatasetModels) or isinstance(models, list):
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            models.extend(self.datasets.models)

        self.datasets.models = models


        log.info(models)

    def write_datasets(self, overwrite=True, path_file=None):
        """Write Datasets and Models to YAML file.

            Parameters
            ----------
            overwrite : bool, optional
                Overwrite existing file. Default is True.  
            """

        write_datasets(self.datasets, path_file, overwrite)
    
    def read_datasets(self, path_file=None):
        """Read Datasets and Models from YAML file."""
        return read_datasets(path_file)


# In[ ]:




