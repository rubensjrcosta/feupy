#!/usr/bin/env python
# coding: utf-8

# In[4]:


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

from feupy.analysis import CounterpartsAnalysisConfig, SimulationConfig, CTAObservationAnalysisConfig
from feupy.cta.irfs import Irfs
from feupy.utils.scripts import skcoord_to_dict, dict_to_skcoord


from feupy.plotters import *

from feupy.catalog.config import *


# In[3]:


get_ipython().system('pyflakes core.py')


# In[2]:





# In[ ]:





# In[4]:


__all__ = ["CounterpartsAnalysis", "CTAObservationAnalysis", 'Simulation']


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
        self.datastore = None
        self.observations = None
        self.datasets = None
        self.fit = Fit()
        self.fit_result = None
        self.flux_points = None
        self.dataset_map = None
        self.pointing = dict_to_skcoord(self.config.target.position)
        self._datasets_settings = self.config.datasets
        self._observations_settings = self.config.observations
        self._ctao_perf = Irfs
        self._ctao_perf.get_irfs(self.config.irfs.opt)
        self.observation = None
        self.geom = None
        self.energy_axis_true = self._make_energy_axis(self._datasets_settings.geom.axes.energy_true)
        self.energy_axis_reco = self._make_energy_axis(self._datasets_settings.geom.axes.energy)
        self.spectrum_dataset_empty = None
        self.maker = None
        self.safe_maker = None
        self.spectrum_dataset = None
        self.spectrum_dataset_onoff = None
        self.wstat = None
        
    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)

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

    def _set_data_store(self):
        """Set the datastore on the Simulation object."""
        path = make_path(self.config.observations.datastore)
        if path.is_file():
            log.debug(f"Setting datastore from file: {path}")
            self.datastore = DataStore.from_file(path)
        elif path.is_dir():
            log.debug(f"Setting datastore from directory: {path}")
            self.datastore = DataStore.from_dir(path)
        else:
            raise FileNotFoundError(f"Datastore not found: {path}")

    def _make_obs_table_selection(self):
        """Return list of obs_ids after filtering on datastore observation table."""
        obs_settings = self.config.observations

        # Reject configs with list of obs_ids and obs_file set at the same time
        if len(obs_settings.obs_ids) and obs_settings.obs_file is not None:
            raise ValueError(
                "Values for both parameters obs_ids and obs_file are not accepted."
            )

        # First select input list of observations from obs_table
        if len(obs_settings.obs_ids):
            selected_obs_table = self.datastore.obs_table.select_obs_id(
                obs_settings.obs_ids
            )
        elif obs_settings.obs_file is not None:
            path = make_path(obs_settings.obs_file)
            ids = list(Table.read(path, format="ascii", data_start=0).columns[0])
            selected_obs_table = self.datastore.obs_table.select_obs_id(ids)
        else:
            selected_obs_table = self.datastore.obs_table

        # Apply cone selection
        if obs_settings.obs_cone.lon is not None:
            cone = dict(
                type="sky_circle",
                frame=obs_settings.obs_cone.frame,
                lon=obs_settings.obs_cone.lon,
                lat=obs_settings.obs_cone.lat,
                radius=obs_settings.obs_cone.radius,
                border="0 deg",
            )
            selected_obs_table = selected_obs_table.select_observations(cone)

        return selected_obs_table["OBS_ID"].tolist()

    def get_observations(self):
        """Fetch observations from the data store according to criteria defined
        in the configuration."""
        observations_settings = self.config.observations
        self._set_data_store()

        log.info("Fetching observations.")
        ids = self._make_obs_table_selection()
        required_irf = [_.value for _ in observations_settings.required_irf]
        self.observations = self.datastore.get_observations(
            ids, skip_missing=True, required_irf=required_irf
        )

        if observations_settings.obs_time.start is not None:
            start = observations_settings.obs_time.start
            stop = observations_settings.obs_time.stop
            if len(start.shape) == 0:
                time_intervals = [(start, stop)]
            else:
                time_intervals = [(tstart, tstop) for tstart, tstop in zip(start, stop)]
            self.observations = self.observations.select_time(time_intervals)

        log.info(f"Number of selected observations: {len(self.observations)}")

        for obs in self.observations:
            log.debug(obs)

    def get_datasets(self):
        dataset_onoff = self._making_spectrum_dataset_onoff()
        dataset = self.spectrum_dataset
        
        datasets = Datasets()

        for idx in range(self.config.observations.parameters.n_obs):
            dataset_onoff.fake(
                random_state=idx, 
                npred_background=dataset.npred_background()
            )
            dataset_fake = dataset_onoff.copy(name=f"obs-{idx}")
            dataset_fake.meta_table["OBS_ID"] = [idx]
            datasets.append(dataset_fake)
    
            self.datasets = datasets
    
    def _making_spectrum_dataset_onoff(self):
        """Produce reduced datasets."""
        datasets_settings = self.config.datasets
        observations_settings = self.config.observations
        target_settings = self.config.target
        ctao_perf = self._ctao_perf
        
        observation = self._create_observation(
            dict_to_skcoord(target_settings.position), 
            observations_settings.parameters.livetime, 
            ctao_perf.irfs, 
            ctao_perf.obs_loc
        )
        
        geom = self._create_geometry()
        
        dataset_empty = self._create_spectrum_dataset_empty(
            geom,
            self._make_energy_axis(datasets_settings.geom.axes.energy_true)
        )
        
        maker = self._create_dataset_maker()
        safe_maker = self._create_safe_mask_maker()
        dataset = maker.run(dataset_empty, observation) 
        dataset = safe_maker.run(dataset, observation)
        
        log.info("Set the model on the dataset, and fake.")
        
        model_dict = target_settings.model
        sky_model = Model.from_dict(model_dict)
        dataset.models = sky_model
        dataset.fake(random_state=42)
        self.geom = geom
        self.observation = observation
        self.spectrum_dataset_empty = dataset_empty
        self.maker = maker
        self.safe_maker = safe_maker
        self.spectrum_dataset = dataset
        
        dataset_onoff = self._create_spectrum_dataset_onoff(
            dataset, 
            datasets_settings.acceptance, 
            datasets_settings.acceptance_off)
        
        self.spectrum_dataset_onoff = dataset_onoff
        
        return dataset_onoff
    
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

        bkg_models = []
        for dataset in self.datasets:
            if dataset.tag == "MapDataset" and dataset.background_model is None:
                bkg_models.append(FoVBackgroundModel(dataset_name=dataset.name))
        if bkg_models:
            models.extend(bkg_models)
            self.datasets.models = models

        log.info(models)

    def read_models(self, path, extend=True):
        """Read models from YAML file.

        Parameters
        ----------
        path : str
            path to the model file
        extend : bool
            Extend the exiting models on the datasets or replace them.
        """

        path = make_path(path)
        models = Models.read(path)
        self.set_models(models, extend=extend)
        log.info(f"Models loaded from {path}.")

    def write_models(self, overwrite=True, write_covariance=True):
        """Write models to YAML file.
        File name is taken from the configuration file.
        """

        filename_models = self.config.general.models_file
        if filename_models is not None:
            self.models.write(
                filename_models, overwrite=overwrite, write_covariance=write_covariance
            )
            log.info(f"Models loaded from {filename_models}.")
        else:
            raise RuntimeError("Missing models_file in config.general")

    def read_datasets(self):
        """Read datasets from YAML file.
        File names are taken from the configuration file.

        """

        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets = Datasets.read(filename)
            log.info(f"Datasets loaded from {filename}.")
            if filename_models is not None:
                self.read_models(filename_models, extend=False)
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def write_datasets(self, overwrite=True, write_covariance=True):
        """Write datasets to YAML file.
        File names are taken from the configuration file.

        Parameters
        ----------
        overwrite : bool
            overwrite datasets FITS files
        write_covariance : bool
            save covariance or not
        """

        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets.write(
                filename,
                filename_models,
                overwrite=overwrite,
                write_covariance=write_covariance,
            )
            log.info(f"Datasets stored to {filename}.")
            log.info(f"Datasets stored to {filename_models}.")
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def run_fit(self):
        """Fitting reduced datasets to model."""
        if not self.models:
            raise RuntimeError("Missing models")

        fit_settings = self.config.fit
        for dataset in self.datasets:
            if fit_settings.fit_range:
                energy_min = fit_settings.fit_range.min
                energy_max = fit_settings.fit_range.max
                geom = dataset.counts.geom
                dataset.mask_fit = geom.energy_mask(energy_min, energy_max)

        log.info("Fitting datasets.")
        result = self.fit.run(datasets=self.datasets)
        self.fit_result = result
        log.info(self.fit_result)

    def get_flux_points(self):
        """Calculate flux points for a specific model component."""
        if not self.datasets:
            raise RuntimeError(
                "No datasets defined. Impossible to compute flux points."
            )

        fp_settings = self.config.flux_points
        log.info("Calculating flux points.")
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
            data=fp, models=self.models[fp_settings.source]
        )
        cols = ["e_ref", "dnde", "dnde_ul", "dnde_err", "sqrt_ts"]
        table = self.flux_points.data.to_table(sed_type="dnde")
        log.info("\n{}".format(table[cols]))

    def get_excess_map(self):
        """Calculate excess map with respect to the current model."""
        excess_settings = self.config.excess_map
        log.info("Computing excess maps.")

        # TODO: Here we could possibly stack the datasets if needed
        # or allow to compute the excess map for each dataset
        if len(self.datasets) > 1:
            raise ValueError("Datasets must be stacked to compute the excess map")

        if self.datasets[0].tag not in ["MapDataset", "MapDatasetOnOff"]:
            raise ValueError("Cannot compute excess map for 1D dataset")

        energy_edges = self._make_energy_axis(excess_settings.energy_edges)
        if energy_edges is not None:
            energy_edges = energy_edges.edges

        excess_map_estimator = ExcessMapEstimator(
            correlation_radius=excess_settings.correlation_radius,
            energy_edges=energy_edges,
            **excess_settings.parameters,
        )
        self.excess_map = excess_map_estimator.run(self.datasets[0])

    def get_light_curve(self):
        """Calculate light curve for a specific model component."""
        lc_settings = self.config.light_curve
        log.info("Computing light curve.")
        energy_edges = self._make_energy_axis(lc_settings.energy_edges).edges

        if (
            lc_settings.time_intervals.start is None
            or lc_settings.time_intervals.stop is None
        ):
            log.info(
                "Time intervals not defined. Extract light curve on datasets GTIs."
            )
            time_intervals = None
        else:
            time_intervals = [
                (t1, t2)
                for t1, t2 in zip(
                    lc_settings.time_intervals.start, lc_settings.time_intervals.stop
                )
            ]

        light_curve_estimator = LightCurveEstimator(
            time_intervals=time_intervals,
            energy_edges=energy_edges,
            source=lc_settings.source,
            fit=self.fit,
            n_jobs=self.config.general.n_jobs,
            **lc_settings.parameters,
        )
        lc = light_curve_estimator.run(datasets=self.datasets)
        self.light_curve = lc
        log.info(
            "\n{}".format(
                self.light_curve.to_table(format="lightcurve", sed_type="flux")
            )
        )

    def update_config(self, config):
        self.config = self.config.update(config=config)

    @staticmethod
    def _create_wcs_geometry(wcs_geom_settings, axes):
        """Create the WCS geometry."""
        geom_params = {}
        skydir_settings = wcs_geom_settings.skydir
        if skydir_settings.lon is not None:
            skydir = SkyCoord(
                skydir_settings.lon, skydir_settings.lat, frame=skydir_settings.frame
            )
            geom_params["skydir"] = skydir

        if skydir_settings.frame in ["icrs", "galactic"]:
            geom_params["frame"] = skydir_settings.frame
        else:
            raise ValueError(
                f"Incorrect skydir frame: expect 'icrs' or 'galactic'. Got {skydir_settings.frame}"
            )

        geom_params["axes"] = axes
        geom_params["binsz"] = wcs_geom_settings.binsize
        width = wcs_geom_settings.width.width.to("deg").value
        height = wcs_geom_settings.width.height.to("deg").value
        geom_params["width"] = (width, height)

        return WcsGeom.create(**geom_params)

    @staticmethod
    def _create_region_geometry(on_region_settings, axes, offset):
        """Create the region geometry."""
        on_lon = on_region_settings.lon
        on_lat = on_region_settings.lat
        pointing = SkyCoord(on_lon, on_lat, frame=on_region_settings.frame)
        on_center = pointing.directional_offset_by(
            position_angle=pointing.dec, 
            separation=offset)
        on_region = CircleSkyRegion(on_center, on_region_settings.radius)
        return RegionGeom.create(region=on_region, axes=axes)


    def _create_geometry(self):
        """Create the geometry."""
        log.debug("Creating geometry.")
        datasets_settings = self.config.datasets
        geom_settings = datasets_settings.geom
        observations_settings = self.config.observations
        offset = observations_settings.parameters.offset
        axes = [self._make_energy_axis(geom_settings.axes.energy)]
        if datasets_settings.type == "3d":
            geom = self._create_wcs_geometry(geom_settings.wcs, axes)
        elif datasets_settings.type == "1d":
            geom = self._create_region_geometry(
                datasets_settings.on_region, axes, offset)
        else:
            raise ValueError(
                f"Incorrect dataset type. Expect '1d' or '3d'. Got {datasets_settings.type}."
            )
        return geom

    def _create_reference_dataset(self, name=None):
        """Create the reference dataset for the current analysis."""
        log.debug("Creating target Dataset.")
        geom = self._create_geometry()

        geom_settings = self.config.datasets.geom
        geom_irf = dict(energy_axis_true=None, binsz_irf=None)
        if geom_settings.axes.energy_true.min is not None:
            geom_irf["energy_axis_true"] = self._make_energy_axis(
                geom_settings.axes.energy_true, name="energy_true"
            )
        if geom_settings.wcs.binsize_irf is not None:
            geom_irf["binsz_irf"] = geom_settings.wcs.binsize_irf.to("deg").value

        if self.config.datasets.type == "1d":
            return SpectrumDataset.create(geom, name=name, **geom_irf)
        else:
            return MapDataset.create(geom, name=name, **geom_irf)

    def _create_dataset_maker(self):
        """Create the Dataset Maker."""
        log.debug("Creating the target Dataset Maker.")

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
        """Create the SafeMaskMaker."""
        log.debug("Creating the mask_safe Maker.")

        safe_mask_selection = self.config.datasets.safe_mask.methods
        safe_mask_settings = self.config.datasets.safe_mask.parameters
        return SafeMaskMaker(methods=safe_mask_selection, **safe_mask_settings)

    def _create_background_maker(self):
        """Create the Background maker."""
        log.info("Creating the background Maker.")

        datasets_settings = self.config.datasets
        bkg_maker_config = {}
        if datasets_settings.background.exclusion:
            path = make_path(datasets_settings.background.exclusion)
            exclusion_mask = Map.read(path)
            exclusion_mask.data = exclusion_mask.data.astype(bool)
            bkg_maker_config["exclusion_mask"] = exclusion_mask
        bkg_maker_config.update(datasets_settings.background.parameters)

        bkg_method = datasets_settings.background.method

        bkg_maker = None
        if bkg_method == "fov_background":
            log.debug(f"Creating FoVBackgroundMaker with arguments {bkg_maker_config}")
            bkg_maker = FoVBackgroundMaker(**bkg_maker_config)
        elif bkg_method == "ring":
            bkg_maker = RingBackgroundMaker(**bkg_maker_config)
            log.debug(f"Creating RingBackgroundMaker with arguments {bkg_maker_config}")
            if datasets_settings.geom.axes.energy.nbins > 1:
                raise ValueError(
                    "You need to define a single-bin energy geometry for your dataset."
                )
        elif bkg_method == "reflected":
            bkg_maker = ReflectedRegionsBackgroundMaker(**bkg_maker_config)
            log.debug(
                f"Creating ReflectedRegionsBackgroundMaker with arguments {bkg_maker_config}"
            )
        else:
            log.warning("No background maker set. Check configuration.")
        return bkg_maker

    def compute_wstat(self):
        # Class to compute statistics for Poisson distributed variable with unknown background.
        
        self.wstat = WStatCountsStatistic(
            n_on=sum(self.spectrum_dataset_onoff.counts.data), 
            n_off=sum(self.spectrum_dataset_onoff.counts_off.data), 
            alpha=self.config.datasets.alpha)

    def map_making(self):
        """Make maps and datasets for 3d analysis"""
        datasets_settings = self.config.datasets
        observations_settings = self.config.observations
        target_settings = self.config.target
        ctao_perf = self._ctao_perf
        pointing = dict_to_skcoord(target_settings.position)
        observation = self._create_observation(
            pointing, 
            observations_settings.parameters.livetime, 
            ctao_perf.irfs, 
            ctao_perf.obs_loc
        )
        log.info("Creating reference dataset and makers.")

        dataset_empty = self._create_spectrum_dataset_empty(
            self._create_geometry(), 
            self._make_energy_axis(datasets_settings.geom.axes.energy_true)
        )

        maker = self._create_dataset_maker()
        safe_maker = self._create_safe_mask_maker()
        dataset = maker.run(dataset_empty, observation)
        dataset = safe_maker.run(dataset, observation)

        log.info("Set the model on the dataset, and fake.")

        dataset.models = Model.from_dict(target_settings.model)
        dataset.fake(random_state=42)
        self.dataset_map = dataset

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        log.info("Reducing spectrum datasets.")
        datasets_settings = self.config.datasets
        dataset_maker = self._create_dataset_maker()
        safe_mask_maker = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()

        reference = self._create_reference_dataset()

        datasets = []
        for obs in progress_bar(self.observations, desc="Observations"):
            log.debug(f"Processing observation {obs.obs_id}")
            dataset = dataset_maker.run(reference.copy(), obs)
            if bkg_maker is not None:
                dataset = bkg_maker.run(dataset, obs)
                if dataset.counts_off is None:
                    log.debug(
                        f"No OFF region found for observation {obs.obs_id}. Discarding."
                    )
                    continue
            dataset = safe_mask_maker.run(dataset, obs)
            log.debug(dataset)
            datasets.append(dataset)
        self.datasets = Datasets(datasets)

        if datasets_settings.stack:
            stacked = self.datasets.stack_reduce(name="stacked")
            self.datasets = Datasets([stacked])
            
            
    @staticmethod    
    def _create_spectrum_dataset_onoff(dataset, acceptance, acceptance_off):
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
    def _create_observation(pointing, livetime, irfs, location):
        """Create an observation."""
        return Observation.create(
            pointing=pointing,
            livetime=livetime,
            irfs=irfs,
            location=location,
        )

    def estimate_sensitivity(self):
        sensitivity_settings = self.config.sensitivity
        sensitivity = SensitivityEstimator(
            spectrum=Model.from_dict(self.config.target.model).spectral_model,
            gamma_min=sensitivity_settings.gamma_min, 
            n_sigma=sensitivity_settings.n_sigma, 
            bkg_syst_fraction=sensitivity_settings.bkg_syst_fraction
        )
        self.sens = sensitivity
        self.sensitivity_table = sensitivity.run(self.spectrum_dataset_onoff)
        

    @staticmethod
    def _create_spectrum_dataset_empty(geom, energy_axis_true, name="obs-0"):
        """Create a MapDataset object with zero filled maps."""
        return SpectrumDataset.create(
            geom=geom, 
            energy_axis_true=energy_axis_true,
            name=name,
        )
    
    @staticmethod
    def _create_center(pointing, offset):
        return pointing.directional_offset_by(position_angle=pointing.dec, separation=offset)

    @staticmethod
    def _make_energy_axis(config_axis_energy, per_decade=True):
        return MapAxis.from_energy_bounds(        
            energy_min=config_axis_energy.min, 
            energy_max=config_axis_energy.max, 
            nbin=config_axis_energy.nbins, 
            per_decade=per_decade, 
            name=config_axis_energy.name,
            )


# In[1]:


from astropy.io.fits.verify import VerifyWarning
import warnings



# In[3]:


class CounterpartsAnalysis:
    """Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `~gammapy.analysis.counterparts.CounterpartsAnalysisConfig`
        Configuration options following `CounterpartsAnalysisConfig` schema.
    """
    all = []
    def __init__(self, config):
        self.config = config
        self.catalogs = None
        self.datasets = None
#         self.sources = self.config.roi.sources
        self.sources = None
        self.models = None
        self.pulsars = None
        self.dict_roi = None
        self.df_roi = None
        CounterpartsAnalysis.all.append(self)
        
    @property
    def config(self):
        """Analysis configuration as an `~feupy.analysis.CounterpartsAnalysisConfig` object."""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, CounterpartsAnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be CounterpartsAnalysisConfig")
            
    def run(self):
        self._get_datasets()
        self._get_dict_roi()
        self._get_df_roi()
        
    def _get_datasets(self):
        """
        Select a catalog subset (only sources within a region of interest)
        """
        _catalogs = self.config.roi.catalogs
        self.pulsars = self.config.roi.pulsars

        datasets = Datasets() # global datasets object
        models = Models()  # global models object
        sources = [] # global sources object
        catalogs = []
        n_sources = 0 # number of sources
        n_flux_points = 0 # number of flux points tables
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            for catalog in _catalogs:
                indexes = []
                cat_tag = catalog.tag
                for source in catalog:
                    n_sources += 1   
                    source_name = source.name 
                    index = source.row_index
                    if cat_tag == PULSARTAG:
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

                            if any([self.config.e_ref_min !=  None, self.config.e_ref_max !=  None]):
                                dataset = cut_energy_table_fp(dataset, self.config.e_ref_min, self.config.e_ref_max) 

                            n_flux_points += 1
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
            self.models = models
            self.sources = sources
            self.catalogs = catalogs

            print(f"Total number of gamma sources: {len(self.sources)}")
            print(f"Total number of flux points tables: {n_flux_points}")
            print(f"Total number of pulsars: {len(self.pulsars)}")

#     def sensitivity_estimator(self):

             
    def _get_dict_roi(self):
        _dict_roi = {}

        roi_pos = self.config.roi.target.position 
        radius_roi = self.config.roi.radius 

        _sources = self.sources.copy()
        _sources.extend(self.pulsars)
        for index, source in enumerate(_sources):
            source_pos = source.position
            sep = source.position.separation(roi_pos).deg
            if index < len(self.datasets):
                name = self.datasets[index].name
            else: name = source.name
            _dict_roi[name] = {
                'position': source_pos,
                'separation':sep
            }

        self.dict_roi = _dict_roi
        
    def _get_df_roi(self):
        _dict = self.dict_roi

        df = pd.DataFrame()
        df["Source name"] = _dict.keys()
        df_ra = []
        df_dec = []
        df_sep = []

        for index, name in enumerate(_dict.keys()):
            df_ra.append(_dict[name]["position"].ra.deg)
            df_dec.append(_dict[name]["position"].dec.deg)
            df_sep.append(_dict[name]["separation"])

        df["RA(deg)"] = df_ra
        df["dec.(deg)"] = df_dec
        df["Sep.(deg)"] = df_sep
        self.df_roi = df
        
    def create_analysis_name(self): 
        """ ... """
        ss = f"{self.config.target.name}"
        ss += "_roi_{:.2f}".format(self.config.roi.radius).replace(' ', '')
        if self.config.e_ref_min is None: ss += ""
        else: ss += "_e_ref_min_{}".format(energy_unit_format(self.config.e_ref_min).replace(' ', ''))
        if self.config.e_ref_max is None: ss += ""
        else: ss += "_e_ref_max_{}".format(energy_unit_format(self.config.e_ref_max).replace(' ', ''))
        return ss
    
    def create_analysis_path(self): 
        """ ... """
        return Path(f"analysis_counterparts/{self.create_analysis_name()}")

    def write_datasets(self, overwrite=True, path_file=None):
        """Write Datasets and Models to YAML file.

            Parameters
            ----------
            overwrite : bool, optional
                Overwrite existing file. Default is True.  
            """
        
        if path_file is None:
            path_file = Path(f"{self.create_analysis_path()}/datasets")
        write_datasets(self.datasets, path_file, overwrite)
    
    def read_datasets(self, path_file=None):
        """Read Datasets and Models from YAML file."""

        if path_file is None:
            path_file = Path(f"{self.create_analysis_path()}/datasets")
        return read_datasets(path_file)


# In[3]:


class CTAObservationAnalysis:
    """Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `~gammapy.analysis.counterparts.CTAObservationAnalysisConfig`
        Configuration options following `CTAObservationAnalysisConfig` schema.
    """
    all = []
    def __init__(self, config):
        self.config = config
        self.dataset_onoff = None
        self.wstat = None
        self.sens = None
        self.sensitivity_table = None
        self.fit_result_params = None 
        self.fit_result_joint = None
        self.fpe = None
        self.flux_points = None
        

        CTAObservationAnalysis.all.append(self)

        
    @property
    def config(self):
        """Analysis configuration as an `~feupy.analysis.CTAObservationAnalysisConfig` object."""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, CTAObservationAnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be CTAObservationAnalysisConfig")
            
#     def run(self):
#         self._make_energy_axes()
#         self._get_dict_roi()
#         self._get_df_roi()
        
#     def _make_energy_axes(self):
        
#         """
#         Select a catalog subset (only sources within a region of interest)
#         """

#         datasets = Datasets() # global datasets object
#         models = Models()  # global models object
#         sources = [] # global sources object
#         pulsars = [] # global pulsars object
#         n_sources = 0 # number of sources
#         n_flux_points = 0 # number of flux points tables
        
#         for catalog in self.catalogs:
#             cat_tag = catalog.tag
#             for source in catalog:
#                 n_sources += 1   
#                 source_name = source.name            
#                 if cat_tag == PULSARTAG:
#                     pass
#                 else:
#                     try:
#                         flux_points = source.flux_points

#                         spectral_model = source.spectral_model()
#                         spectral_model_tag = spectral_model.tag[1]

#                         if cat_tag == 'gamma-cat' or cat_tag == 'hgps':
#                             dataset_name = f'{source_name}: {cat_tag}'
#                         else: dataset_name = source_name

#                         file_name = name_to_txt(dataset_name)

#                         model = SkyModel(
#                             name=f"{file_name}_{spectral_model_tag}",
#                             spectral_model=spectral_model,
#                             datasets_names=dataset_name
#                         )

#                         dataset = FluxPointsDataset(
#                             models=model,
#                             data=flux_points, 
#                             name=dataset_name   
#                         )

#                         if any([self.config.e_ref_min !=  None, self.config.e_ref_max !=  None]):
#                             dataset = cut_energy_table_fp(dataset, self.config.e_ref_min, self.config.e_ref_max) 

#                         n_flux_points += 1
#                         models.append(model)  # Add the model to models()
#                         datasets.append(dataset)
#                         sources.append(source)
#                     except Exception as error:
#                         # By this way we can know about the type of error occurring
#                         print(f'The error is: ({source_name}) {error}') 

#         datasets.models = models
#         self.datasets = datasets
#         self.models = models
#         self.sources = sources
#         print(f"Total number of gammapy sources: {len(self.sources)}")
#         print(f"Total number of flux points tables: {n_flux_points}")
#         print(f"Total number of pulsars: {len(self.pulsars)}")
             
#     def _get_dict_roi(self):
#         _dict_roi = {}

#         roi_pos = self.config.roi.target.position 
#         radius_roi = self.config.roi.radius 

#         _sources = self.sources.copy()
#         _sources.extend(self.pulsars)
#         for index, source in enumerate(_sources):
#             source_pos = source.position
#             sep = source.position.separation(roi_pos).deg
#             if index < len(self.datasets):
#                 name = self.datasets[index].name
#             else: name = source.name
#             _dict_roi[name] = {
#                 'position': source_pos,
#                 'separation':sep
#             }

#         self.dict_roi = _dict_roi
        
#     def _get_df_roi(self):
#         _dict = self.dict_roi

#         df = pd.DataFrame()
#         df["Source name"] = _dict.keys()
#         df_ra = []
#         df_dec = []
#         df_sep = []

#         for index, name in enumerate(_dict.keys()):
#             df_ra.append(_dict[name]["position"].ra.deg)
#             df_dec.append(_dict[name]["position"].dec.deg)
#             df_sep.append(_dict[name]["separation"])

#         df["RA(deg)"] = df_ra
#         df["dec.(deg)"] = df_dec
#         df["Sep.(deg)"] = df_sep
#         self.df_roi = df
        
#     def create_analysis_name(self): 
#         """ ... """
#         ss = f"{self.config.target.name}"
#         ss += "_roi_{:.2f}".format(self.config.roi.radius).replace(' ', '')
#         if self.config.e_ref_min is None: ss += ""
#         else: ss += "_e_ref_min_{}".format(energy_unit_format(self.config.e_ref_min).replace(' ', ''))
#         if self.config.e_ref_max is None: ss += ""
#         else: ss += "_e_ref_max_{}".format(energy_unit_format(self.config.e_ref_max).replace(' ', ''))
#         return ss
    
#     def create_analysis_path(self): 
#         """ ... """
#         return Path(f"analysis_counterparts/{self.create_analysis_name()}")

#     def write_datasets(self, overwrite=True, path_file=None):
#         """Write Datasets and Models to YAML file.

#             Parameters
#             ----------
#             overwrite : bool, optional
#                 Overwrite existing file. Default is True.  
#             """
        
#         if path_file is None:
#             path_file = Path(f"{self.create_analysis_path()}/datasets")
#         write_datasets(self.datasets, path_file, overwrite)
    
#     def read_datasets(self, path_file=None):
#         """Read Datasets and Models from YAML file."""

#         if path_file is None:
#             path_file = Path(f"{self.create_analysis_path()}/datasets")
#         return read_datasets(path_file)


# In[2]:


# # To save only the models
# models_3fhl.write("3fhl_models.yaml", overwrite=True)

# # To save datasets and models
# datasets.write(
#     filename="datasets-gc.yaml", filename_models="models_gc.yaml", overwrite=True
# )

# # To read only models
# models = Models.read("3fhl_models.yaml")
# print(models)

# # To read datasets with models
# datasets_read = Datasets.read("datasets-gc.yaml", filename_models="models_gc.yaml")
# print(datasets_read)


# In[ ]:





# In[7]:


def test_analysis_confg():
    return CounterpartsAnalysisConfig(
        "LHAASO J1825-1326", 
        276.45* u.Unit('deg'), 
        -13.45* u.Unit('deg'),
        1* u.Unit('deg'),
        1* u.Unit('erg')
    )


# In[10]:





# In[ ]:




