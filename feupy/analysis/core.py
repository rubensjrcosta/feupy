# Licensed under a 3-clause BSD style license - see LICENSE
"""High-level interfaces for ROI and CTAO analyses."""

import html
import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.data import FixedPointingInfo, Observation, Observations
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.estimators import FluxPoints, FluxPointsEstimator, SensitivityEstimator
from gammapy.makers import (
    FoVBackgroundMaker,
    ReflectedRegionsBackgroundMaker,
    RingBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel, Models, SkyModel
from gammapy.utils.scripts import make_path
from regions import CircleSkyRegion

from feupy.analysis.config import CTAOAnalysisConfig, ROIAnalysisConfig
from feupy.catalogs.fermi import get_flux_points_2PC, get_flux_points_3PC
from feupy.catalogs.hawc import get_flux_points_2hwc, get_flux_points_3hwc
from feupy.catalogs.lhaaso import get_flux_points_1lhaaso
from feupy.catalogs.utils import get_catalog_tag, load_catalogs
from feupy.catalogs.veritas import generate_unique_name
from feupy.core.sources import Sources
from feupy.irf import CTAOIRFManager
from feupy.utils.coordinates import convert_pos_config_to_skycoord
from feupy.utils.datasets import (
    cut_energy_flux_points_datasets,
)
from feupy.utils.tables.io import read_table, write_table

__all__ = ["ROIAnalysis", "CTAOAnalysis"]

log = logging.getLogger(__name__)


class ROIAnalysis:
    """Config-driven high level simulation interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `ROIAnalysisConfig`
        Configuration options following `ROIAnalysisConfig` schema
    """

    def __init__(self, config):
        self.config = config
        self.config.set_logging()
        self.datasets = None
        self.sources = None
        self._has_fp = None
        self.catalog = None

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

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
        """Simulation configuration (`ROIAnalysisConfig`)"""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = ROIAnalysisConfig(**value)
        elif isinstance(value, ROIAnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or ROIAnalysisConfig.")

    def _get_catalogs(self):
        """Load source catalogs and filter sources within the ROI."""
        catalogs = []
        config_settings = self.config
        all_catalogs = load_catalogs()
        position = convert_pos_config_to_skycoord(config_settings.roi)
        radius = config_settings.roi.radius

        for catalog in all_catalogs:
            separation = position.separation(catalog.positions)
            mask_roi = separation < radius

            if len(catalog[mask_roi].table):
                catalogs.append(catalog[mask_roi])
        return catalogs

    def _get_sources(self):
        """Retrieve sources from catalogs within the ROI."""
        sources = []
        for catalog in self._get_catalogs():
            sources.extend(catalog)
        self.sources = Sources(sources)

    def _get_catalog_roi(self):
        """
        Compute separations of sources from the target position and return an Astropy Table.

        Returns
        -------
        result_table : `~astropy.table.Table`
            Table with columns for source name, catalog, RA, Dec, and separation from target.
        """
        names, ras, decs, separations, catalogs, has_fp = [], [], [], [], [], []
        sources = self.sources
        config_settings = self.config
        target_position = convert_pos_config_to_skycoord(config_settings.roi)

        for source in sources:
            source_name = source.name
            source_pos = source.position
            ra, dec = source_pos.ra.deg, source_pos.dec.deg
            catalog = get_catalog_tag(source)
            sep = source_pos.separation(target_position).deg

            names.append(source_name)
            ras.append(ra)
            decs.append(dec)
            catalogs.append(catalog)
            separations.append(sep)
            has_fp.append(self._has_fp[source.name])

        result_table = Table()
        result_table["index"] = list(range(len(names)))
        result_table["source_name"] = names
        result_table["source_label"] = sources.labels
        result_table["catalog"] = catalogs
        result_table["ra"] = ras * u.deg
        result_table["dec"] = decs * u.deg
        result_table["separation"] = separations * u.deg
        result_table["has_fp"] = has_fp

        for column in result_table.colnames:
            if column.startswith(("ra", "dec", "sep")):
                result_table[column].format = ".3f"

        # Add meta (columns descriptions)
        result_table.meta["description"] = {
            "index": "Unique identifier for each source.",
            "source_name": "Name of the source as listed in the catalog.",
            "source_label": "User-defined label for the source, often used for plotting.",
            "has_fp": "Boolean flag indicating if flux points are available (True/False).",
            "catalog": "Catalog from which the source originates.",
            "ra": "Right Ascension (RA) of the source in degrees, formatted to 3 decimal places.",
            "dec": "Declination (Dec) of the source in degrees, formatted to 3 decimal places.",
            "separation": "Angular separation from a reference position in degrees, formatted to 3 decimal places.",
        }
        self.catalog = result_table

    def run(self):
        """Run the ROI analysis, initializing datasets."""
        self._get_sources()
        self._get_flux_points_datasets()
        self._get_catalog_roi()

    def _get_flux_points_datasets(self):
        """Generate FluxPointsDataset objects from sources within the ROI."""
        models = Models()
        datasets = Datasets()
        dict_fp = {}
        sources = self.sources
        log.info("Generating FluxPointsDataset.")

        for index, source in enumerate(sources):
            label = sources.labels[index]
            tag = get_catalog_tag(source)
            if tag == "psrcat":
                dict_fp[source.name] = False
                log.info("Flux points unavailable for ATNF Pulsar Catalog.")
                continue

            try:
                which = None

                if tag == "2PC":
                    spectral_model = self._create_spectral_model(source, which)
                    flux_points = get_flux_points_2PC(source)
                    dict_fp[source.name] = True
                    self._create_flux_points_dataset(
                        f"PSR {label}", models, datasets, spectral_model, flux_points
                    )

                elif tag == "3PC":
                    spectral_model = self._create_spectral_model(source, which)
                    flux_points = get_flux_points_3PC(source)
                    dict_fp[source.name] = True
                    self._create_flux_points_dataset(
                        f"{label}", models, datasets, spectral_model, flux_points
                    )

                elif tag == "3hwc":
                    spectral_model = self._create_spectral_model(source, which)
                    flux_points = get_flux_points_3hwc(source)
                    dict_fp[source.name] = True
                    self._create_flux_points_dataset(
                        f"{label}", models, datasets, spectral_model, flux_points
                    )

                elif tag == "2hwc":
                    for model_type in ["point", "extended"]:
                        spectral_model = self._create_spectral_model(source, model_type)
                        flux_points = get_flux_points_2hwc(source, model_type)
                        dict_fp[source.name] = True
                        self._create_flux_points_dataset(
                            f"{label} ({model_type})",
                            models,
                            datasets,
                            spectral_model,
                            flux_points,
                        )

                elif tag == "1LHAASO":
                    for model_type in ["KM2A", "WCDA"]:
                        spectral_model = self._create_spectral_model(source, model_type)
                        flux_points = get_flux_points_1lhaaso(source, model_type)
                        dict_fp[source.name] = True
                        self._create_flux_points_dataset(
                            f"{label} ({model_type})",
                            models,
                            datasets,
                            spectral_model,
                            flux_points,
                        )

                elif tag == "vtscat":
                    unique_names = [source.name]
                    for flux_points in source.flux_points():
                        reference_id = flux_points.meta["reference_id"]
                        model_name = generate_unique_name(
                            source.name, reference_id, unique_names
                        )
                        unique_names.append(model_name)
                        model = flux_points.reference_model
                        dict_fp[source.name] = True
                        spectral_model = model.spectral_model
                        label = model_name
                        self._create_flux_points_dataset(
                            label, models, datasets, spectral_model, flux_points
                        )

                else:
                    spectral_model = source.spectral_model()
                    flux_points = source.flux_points
                    dict_fp[source.name] = True
                    self._create_flux_points_dataset(
                        label, models, datasets, spectral_model, flux_points
                    )

            except Exception as error:
                dict_fp[source.name] = False
                log.info(f"Unable to create dataset for {source.name}. Error: {error}")

        self.datasets = datasets
        self.datasets.models = models
        self._has_fp = dict_fp

    @staticmethod
    def _create_spectral_model(source, which=None):
        """Helper function to create a FluxPointsDataset for a source."""
        return source.spectral_model(which=which) if which else source.spectral_model()

    def _create_flux_points_dataset(
        self, name, models, datasets, spectral_model, flux_points
    ):
        """Helper function to create a FluxPointsDataset for a source."""

        e_ref_min = self.config.energy_range.min
        e_ref_max = self.config.energy_range.max
        log.info("Getting FluxPointsDataset.")

        model = SkyModel(name=name, spectral_model=spectral_model, datasets_names=name)
        dataset = FluxPointsDataset(models=model, data=flux_points, name=name)

        if any([e_ref_min is not None, e_ref_max is not None]):
            dataset = cut_energy_flux_points_datasets(dataset, e_ref_min, e_ref_max)

        models.append(model)
        datasets.append(dataset)
        log.info(f"Dataset created: {dataset}")

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
        elif isinstance(models, (DatasetModels, list)):
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            models.extend(self.datasets.models)

        self.datasets.models = models

        log.info(models)

    def read_models(self, path, extend=True):
        """Read models from YAML file.

        Parameters
        ----------
        path : str
        Path to the model file.
        extend : bool, optional
        Extend the exiting models on the datasets or replace them.
        Default is True.
        """
        path = make_path(path)
        models = Models.read(path)
        self.set_models(models, extend=extend)
        log.info(f"Models loaded from {path}.")

    def read_sources(self, path):
        """Read sources from YAML file.

        Parameters
        ----------
        path : str
        Path to the model file.
        extend : bool, optional
        Extend the exiting sources on the datasets or replace them.
        Default is True.
        """
        path = make_path(path)
        _sources = Sources()
        _sources.read(path)
        self.sources = Sources(_sources)
        log.info(f"sources loaded from {path}.")

    def write_sources(self, overwrite=True):
        """Write sources to YAML file.

        File name is taken from the configuration file.
        """
        filename_sources = self.config.general.sources_file
        if filename_sources is not None:
            self.sources.write(filename_sources, overwrite=overwrite)
            log.info(f"sources loaded from {filename_sources}.")
        else:
            raise RuntimeError("Missing sources_file in config.general")

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
        overwrite : bool, optional
            Overwrite existing file. Default is True.
        write_covariance : bool, optional
            Save covariance or not. Default is True.
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


class CTAOAnalysis:
    """Config-driven high level analysis interface (ON/OFF 1D only)."""

    def __init__(self, config):
        self.config = config
        self.config.set_logging()

        self.observations = Observations()
        self.datasets = None
        self.spectrum_dataset = None

        self.fit = Fit()
        self.fit_result = None
        self.flux_points = None
        self.table_sens = None

        self.irf_manager = CTAOIRFManager()

    # =====================
    # Config
    # =====================
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = CTAOAnalysisConfig(**value)
        elif isinstance(value, CTAOAnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or CTAOAnalysisConfig.")

    def update_config(self, config):
        self.config = self.config.update(config=config)

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)

    def set_models(self, models, extend=True):
        """Set models on datasets.

        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string.
        extend : bool, optional
            Extend the exiting models on the datasets or replace them.
            Default is True.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info("Reading model.")
        if isinstance(models, str):
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, (DatasetModels, list)):
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
            Path to the model file.
        extend : bool, optional
            Extend the exiting models on the datasets or replace them.
            Default is True.
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

    # =====================
    # Observation
    # =====================
    def simulate_observation(self, obs_id=0):

        if obs_id in self.observations.ids:
            raise ValueError("Observation ids must be unique")

        obs_cfg = self.config.observation
        on_region = self.config.datasets.on_region

        on_center = SkyCoord(on_region.lon, on_region.lat, frame=on_region.frame)

        pointing_pos = self._create_pointing_position(
            on_center,
            obs_cfg.position_angle,
            obs_cfg.offset,
        )

        pointing = self._create_pointing(pointing_pos)

        irf_data = self.irf_manager.get_irf(obs_cfg.required_irfs)

        observation = Observation.create(
            pointing=pointing,
            livetime=obs_cfg.livetime,
            irfs=irf_data["irf"],
            location=irf_data["obs_location"],
            obs_id=obs_id,
        )

        self.observations.append(observation)
        log.info(f"Observation {obs_id} created")

    @staticmethod
    def _create_pointing_position(position, position_angle, separation):
        return position.directional_offset_by(position_angle, separation)

    @staticmethod
    def _create_pointing(pointing_position):
        return FixedPointingInfo(fixed_icrs=pointing_position.icrs)

    # =====================
    # Dataset
    # =====================
    def get_spectrum_dataset(self, model=None, obs_id=0, random_state=42):

        if not self.observations:
            raise RuntimeError("No observations defined")

        if self.config.datasets.type != "1d":
            raise ValueError("Only 1D ON/OFF supported")

        self._spectrum_extraction(model, obs_id, random_state)

    def _spectrum_extraction(self, model=None, obs_id=0, random_state=42):

        obs = self.observations[obs_id]

        obs_cfg = self.config.observation
        ds_cfg = self.config.datasets

        dataset = self._create_dataset_maker().run(
            self._create_reference_dataset(str(obs.obs_id)), obs
        )

        if not ds_cfg.containment_correction:
            containment = ds_cfg.containment
            offset = obs_cfg.offset
            energy_axis = self._make_energy_axis(ds_cfg.geom.axes.energy)
            on_region_radius = ds_cfg.on_region.radius

            dataset.exposure *= containment
            log.info(
                f"\nCorrected exposure (containment: {containment}%):\n{dataset}\n"
            )

            on_radii = obs.psf.containment_radius(
                energy_true=energy_axis.center, offset=offset, fraction=containment
            )

            factor = (1 - np.cos(on_radii)) / (1 - np.cos(on_region_radius))
            dataset.background *= factor.value.reshape((-1, 1, 1))
            log.info(
                f"\nCorrected background (containment: {containment}%):\n{dataset}\n"
            )

        bkg_maker = self._create_background_maker()
        if bkg_maker:
            dataset = bkg_maker.run(dataset, obs)

        dataset = self._create_safe_mask_maker().run(dataset, obs)

        if model:
            dataset.models = model
            dataset.fake(random_state=random_state)

        self.spectrum_dataset = dataset

    # =====================
    # ON/OFF
    # =====================
    def get_datasets(self):

        if self.spectrum_dataset is None:
            raise RuntimeError("No spectrum dataset")

        self._run_on_off()

    def _run_on_off(self):

        cfg = self.config

        dataset_on_off = self._create_dataset_on_off(
            self.spectrum_dataset,
            cfg.datasets.on_off.acceptance,
            cfg.datasets.on_off.acceptance_off,
        )

        datasets = Datasets()

        for i in range(cfg.statistics.n_obs):
            dataset_on_off.fake(
                random_state=i,
                npred_background=self.spectrum_dataset.npred_background(),
            )

            ds = dataset_on_off.copy(name=f"obs-{i}")
            ds.meta_table["OBS_ID"] = [i]
            datasets.append(ds)

        self.datasets = datasets

        if cfg.datasets.stack:
            self.datasets = Datasets([self.datasets.stack_reduce(name="stacked")])

    @staticmethod
    def _create_dataset_on_off(dataset, acceptance, acceptance_off):

        ds = SpectrumDatasetOnOff.from_spectrum_dataset(
            dataset=dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
        )

        ds.fake(
            random_state="random-seed",
            npred_background=dataset.npred_background(),
        )

        return ds

    # =====================
    # Makers
    # =====================
    def _create_dataset_maker(self):
        cfg = self.config.datasets
        return SpectrumDatasetMaker(
            selection=cfg.map_selection,
            use_region_center=cfg.use_region_center,
            containment_correction=cfg.containment_correction,
        )

    def _create_safe_mask_maker(self):
        return SafeMaskMaker(
            methods=self.config.datasets.safe_mask.methods,
            **self.config.datasets.safe_mask.parameters,
        )

    def _create_background_maker(self):

        cfg = self.config.datasets.background

        if cfg.method == "reflected":
            return ReflectedRegionsBackgroundMaker(**cfg.parameters)
        if cfg.method == "ring":
            return RingBackgroundMaker(**cfg.parameters)
        if cfg.method == "fov_background":
            return FoVBackgroundMaker(**cfg.parameters)

        return None

    # =====================
    # Geometry
    # =====================
    def _create_reference_dataset(self, name=None):
        return SpectrumDataset.create(self._create_geometry(), name=name)

    def _create_geometry(self):

        axis = self._make_energy_axis(self.config.datasets.geom.axes.energy)

        center = SkyCoord(
            self.config.datasets.on_region.lon,
            self.config.datasets.on_region.lat,
            frame=self.config.datasets.on_region.frame,
        )

        region = CircleSkyRegion(center, self.config.datasets.on_region.radius)

        return RegionGeom.create(region=region, axes=[axis])

    # =====================
    # Fit & Estimators
    # =====================
    def run_fit(self):

        if not self.datasets:
            raise RuntimeError("No datasets")

        self.fit_result = self.fit.run(datasets=self.datasets)

    def get_flux_points(self):

        cfg = self.config.flux_points

        estimator = FluxPointsEstimator(
            energy_edges=self._make_energy_axis(cfg.energy).edges,
            source=cfg.source,
            fit=self.fit,
        )

        fp = estimator.run(self.datasets)

        self.flux_points = FluxPointsDataset(
            data=fp,
            models=self.models[cfg.source],
        )

    # =====================
    # Sensitivity
    # =====================
    def compute_sensitivity(self):

        cfg = self.config

        dataset_on_off = self._create_dataset_on_off(
            self.spectrum_dataset,
            cfg.datasets.on_off.acceptance,
            cfg.datasets.on_off.acceptance_off,
        )

        estimator = SensitivityEstimator(
            gamma_min=cfg.sensitivity.gamma_min,
            n_sigma=cfg.sensitivity.n_sigma,
            bkg_syst_fraction=cfg.sensitivity.bkg_syst_fraction,
        )

        table = estimator.run(dataset_on_off)

        # Integral sensitivity
        table_img = estimator.run(dataset_on_off.to_image())

        flux_points = FluxPoints.from_table(
            table_img,
            sed_type="e2dnde",
            reference_model=estimator.spectral_model,
        )

        int_sens = np.squeeze(flux_points.flux.quantity)

        table.meta = self._get_table_meta()
        table.meta["INT_SENS"] = f"{int_sens:.2e}"

        self.table_sens = table

    # =====================
    # Utils
    # =====================

    # File Handling Methods
    def write_table_sensitivity(self, overwrite=True):
        """Write sensitivity table to file in configured format."""

        if self.table_sens is None:
            raise RuntimeError("Missing table_sens")

        path_file = self.config.sensitivity.data_path
        file_name = self.get_file_name()
        file_format = self.config.sensitivity.table_format

        full_name = f"{file_name}.{file_format}"

        write_table(
            self.table_sens,
            path_file,
            full_name,
            overwrite=overwrite,
        )

        log.info(f"Table ({full_name}) stored to {path_file}.")

    def read_table_sensitivity(self):
        """Read sensitivity table from file based on configuration."""

        path_file = self.config.sensitivity.data_path
        file_name = self.get_file_name()
        file_format = self.config.sensitivity.table_format

        full_name = f"{file_name}.{file_format}"

        try:
            return read_table(path_file, full_name)

        except Exception as error:
            log.error(f"Error reading sensitivity table: {error}")
            raise

    @staticmethod
    def _make_energy_axis(axis, name="energy"):

        if axis.min is None or axis.max is None:
            return None

        return MapAxis.from_bounds(
            name=name,
            lo_bnd=axis.min.value,
            hi_bnd=axis.max.to_value(axis.min.unit),
            nbin=axis.nbins,
            unit=axis.min.unit,
            interp="log",
        )

    def _get_table_meta(self):

        obs = self.config.observation
        irfs = obs.required_irfs

        irf_data = self.irf_manager.get_irf(irfs)

        return {
            "ONRADIUS": f"{self.config.datasets.on_region.radius.to('deg').value} deg",
            "OFFSET": obs.offset.to_string(),
            "LIVETIME": obs.livetime.to_string(),
            "IRF_NAME": irf_data["name"],
            "IRF_LABEL": irf_data["label"],
            "IRF_ARR": irfs[0] if len(irfs) > 0 else "",
            "IRF_AZ": irfs[1] if len(irfs) > 1 else "",
            "IRF_ZEN": irfs[2] if len(irfs) > 2 else "",
            "IRF_LT": irfs[3] if len(irfs) > 3 else "",
        }

    def get_file_name(self):

        obs = self.config.observation
        irf_name = self.irf_manager.get_irf(obs.required_irfs)["name"]

        livetime = obs.livetime.to_string().replace(" ", "")

        return f"sens_{irf_name}_livetime{livetime}"
